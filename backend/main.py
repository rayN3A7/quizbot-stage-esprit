"""
API FastAPI de QuizBot — couche backend exposée au frontend (Streamlit ou autre).

Authentification : JWT (voir auth.py). Chaque endpoint métier est protégé
par rôle via `Depends(require_role(...))`.

Endpoints :
  POST /auth/register             -> inscription (professeur ou étudiant)
  POST /auth/login                -> connexion, retourne un jeton JWT
  GET  /auth/me                   -> profil de l'utilisateur connecté

  POST /documents/upload          -> [professeur] téléverse + indexe un support de cours
  GET  /documents                 -> [professeur] liste des documents indexés

  POST /quizzes/generate          -> [professeur] génère un quiz via le pipeline RAG
  GET  /quizzes/{quiz_id}         -> [authentifié] récupère un quiz (si publié, ou si professeur)
  POST /quizzes/{quiz_id}/publish -> [professeur] publie un quiz pour les étudiants
  GET  /quizzes                   -> [authentifié] liste des quiz
  POST /quizzes/{quiz_id}/submit  -> [étudiant] soumission des réponses + correction
  GET  /quizzes/{quiz_id}/results -> [professeur] carnet de notes et analyse des questions
  GET  /quizzes/{quiz_id}/export/pdf   -> [professeur]
  GET  /quizzes/{quiz_id}/export/json  -> [professeur]
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Union

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm

from . import storage
from .analytics import gradebook
from .auth import authenticate_user, create_access_token, get_current_user, hash_password, require_role
from .config import settings
from .database import init_db
from .export import export_quiz_json, export_quiz_pdf
from .grading import grade_quiz
from .ingestion import EmptyDocumentError, UnsupportedFileTypeError, process_document
from .models import (
    DocumentInfo, Quiz, QuizConfig, QuizResult, Role, StudentQuiz, SubmissionRequest,
    Token, UserCreate, UserInDB, UserPublic,
)
from .quiz_generator import generate_quiz
from .semantic_map import build_semantic_map, document_chunk_texts, overlay_performance
from .quiz_agent import generate_quiz_agentic
from .vectorstore import get_vector_store

app = FastAPI(
    title="QuizBot API",
    description="Assistant pédagogique intelligent — génération automatique de questionnaires via RAG",
    version="1.1.0",
)

professor_only = require_role(Role.PROFESSOR)
student_only = require_role(Role.STUDENT)
any_role = require_role(Role.PROFESSOR, Role.STUDENT)

# Crée les tables de la base de données (utilisateurs) dès le chargement du
# module, garantissant qu'elles existent avant toute requête (y compris avec
# TestClient(app) utilisé sans context manager dans les tests).
init_db()


@app.get("/health")
def health():
    info = {"status": "ok", "llm_provider": settings.LLM_PROVIDER}
    # Le fournisseur local peut se rabattre sur un modèle plus petit si la VRAM
    # est insuffisante : on expose le modèle RÉELLEMENT chargé, pas celui demandé.
    # Ce détail est facultatif : /health doit rester disponible même si le
    # module LLM est absent, obsolète ou en erreur — c'est précisément l'endpoint
    # qu'on interroge pour diagnostiquer une panne.
    if settings.LLM_PROVIDER.lower() == "local":
        try:
            from .llm_client import LocalLLMProvider
            loaded = getattr(LocalLLMProvider, "_model_name", None)
            info["model"] = loaded or f"{settings.LOCAL_LLM_MODEL} (non chargé)"
        except Exception as e:
            info["model"] = f"indisponible ({type(e).__name__})"
    return info


# --------------------------------------------------------------------------- #
# Authentification
# --------------------------------------------------------------------------- #

@app.post("/auth/register", response_model=UserPublic, status_code=201)
def register(payload: UserCreate):
    if storage.username_exists(payload.username):
        raise HTTPException(400, "Ce nom d'utilisateur est déjà pris.")
    if len(payload.password) < 6:
        raise HTTPException(400, "Le mot de passe doit contenir au moins 6 caractères.")

    if payload.role == Role.PROFESSOR and settings.PROFESSOR_SIGNUP_CODE:
        if payload.professor_code != settings.PROFESSOR_SIGNUP_CODE:
            raise HTTPException(
                403,
                "Code d'inscription professeur invalide. "
                "Contactez votre administrateur pour l'obtenir.",
            )

    user = UserInDB(
        id=storage.new_user_id(),
        username=payload.username,
        full_name=payload.full_name,
        role=payload.role,
        hashed_password=hash_password(payload.password),
    )
    storage.save_user(user)
    return UserPublic(**user.model_dump(exclude={"hashed_password"}))


@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            401, "Nom d'utilisateur ou mot de passe incorrect.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(user.username, user.role)
    return Token(access_token=access_token, role=user.role, username=user.username)


@app.get("/auth/me", response_model=UserPublic)
def read_current_user(current_user: UserInDB = Depends(get_current_user)):
    return UserPublic(**current_user.model_dump(exclude={"hashed_password"}))


# --------------------------------------------------------------------------- #
# Documents (Module Enseignant — étape 1)
# --------------------------------------------------------------------------- #

@app.post("/documents/upload", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(professor_only),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".pptx"):
        raise HTTPException(400, "Formats acceptés : PDF ou PPTX uniquement.")

    document_id = storage.new_document_id()
    dest_path = settings.UPLOADS_DIR / f"{document_id}{suffix}"
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = process_document(dest_path)
        store = get_vector_store()
        num_indexed = store.index_document(document_id, chunks)
    except UnsupportedFileTypeError as e:
        raise HTTPException(400, str(e))
    except EmptyDocumentError as e:
        raise HTTPException(422, str(e))

    info = DocumentInfo(
        id=document_id, filename=file.filename, num_chunks=num_indexed,
        uploaded_by=current_user.username,
    )
    storage.save_document_info(info)
    return info


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(current_user: UserInDB = Depends(professor_only)):
    return storage.list_documents()


@app.get("/documents/{document_id}/map")
def document_semantic_map(
    document_id: str,
    with_performance: bool = False,
    current_user: UserInDB = Depends(professor_only),
):
    """Carte sémantique du document : chunks projetés en 2D.

    Avec `with_performance=true`, les résultats des étudiants sont superposés :
    chaque point porte le score moyen obtenu sur les questions issues de ce
    passage, ce qui fait apparaître les régions du cours mal comprises.
    """
    if storage.get_document_info(document_id) is None:
        raise HTTPException(404, "Document introuvable.")

    try:
        semantic_map = build_semantic_map(document_id)
    except ValueError as e:
        raise HTTPException(422, str(e))

    if with_performance:
        # Seule la première tentative de chaque étudiant compte : les suivantes
        # sont de l'entraînement et gonfleraient les scores.
        semantic_map = overlay_performance(
            semantic_map,
            storage.with_source_excerpts(storage.list_results(first_attempts_only=True)),
            chunk_texts=document_chunk_texts(document_id),
        )
    return semantic_map


# --------------------------------------------------------------------------- #
# Quiz generation (Module Enseignant — étape 2)
# --------------------------------------------------------------------------- #

@app.post("/quizzes/generate", response_model=Quiz)
def generate_quiz_endpoint(config: QuizConfig, current_user: UserInDB = Depends(professor_only)):
    doc_info = storage.get_document_info(config.document_id)
    if doc_info is None:
        raise HTTPException(404, "Document introuvable. Téléversez-le d'abord.")

    generator = generate_quiz_agentic if config.use_verification_agent else generate_quiz
    try:
        quiz = generator(config.document_id, doc_info.filename, config)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # LLMError et erreurs fournisseur
        raise HTTPException(502, f"Erreur lors de la génération par le LLM : {e}")

    quiz.created_by = current_user.username
    storage.save_quiz(quiz)
    return quiz


def _visible_quiz(quiz: Quiz, user: UserInDB) -> Quiz | StudentQuiz:
    # Un étudiant a un JWT valide et peut appeler l'API directement : masquer
    # les réponses dans l'interface ne les protège pas.
    return quiz if user.role == Role.PROFESSOR else StudentQuiz.from_quiz(quiz)


@app.get("/quizzes/{quiz_id}", response_model=Union[Quiz, StudentQuiz])
def get_quiz_endpoint(quiz_id: str, current_user: UserInDB = Depends(any_role)):
    quiz = storage.get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(404, "Quiz introuvable.")
    if not quiz.published and current_user.role != Role.PROFESSOR:
        raise HTTPException(403, "Ce quiz n'est pas encore publié.")
    return _visible_quiz(quiz, current_user)


@app.get("/quizzes", response_model=list[Union[Quiz, StudentQuiz]])
def list_quizzes_endpoint(published_only: bool = False, current_user: UserInDB = Depends(any_role)):
    # Un étudiant ne doit jamais voir les quiz non publiés, quel que soit le paramètre.
    force_published_only = published_only or current_user.role == Role.STUDENT
    quizzes = storage.list_quizzes(published_only=force_published_only)
    return [_visible_quiz(q, current_user) for q in quizzes]


@app.post("/quizzes/{quiz_id}/publish", response_model=Quiz)
def publish_quiz(quiz_id: str, current_user: UserInDB = Depends(professor_only)):
    quiz = storage.get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(404, "Quiz introuvable.")
    quiz.published = True
    storage.save_quiz(quiz)
    return quiz


# --------------------------------------------------------------------------- #
# Module Étudiant — auto-évaluation
# --------------------------------------------------------------------------- #

@app.post("/quizzes/{quiz_id}/submit", response_model=QuizResult)
def submit_answers(
    quiz_id: str,
    submission: SubmissionRequest,
    current_user: UserInDB = Depends(student_only),
):
    quiz = storage.get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(404, "Quiz introuvable.")
    if not quiz.published:
        raise HTTPException(403, "Ce quiz n'est pas encore publié.")

    # Le nom de l'étudiant provient du compte authentifié, jamais du corps de
    # la requête : impossible de soumettre des réponses sous une autre identité.
    student_name = current_user.full_name or current_user.username
    result = grade_quiz(quiz, student_name, submission.answers)
    result.student_username = current_user.username
    storage.save_result(result)
    return result


@app.get("/quizzes/{quiz_id}/results")
def quiz_results(quiz_id: str, current_user: UserInDB = Depends(professor_only)):
    """Carnet de notes : toutes les copies remises, numérotées par tentative ; seule
    la première tentative de chaque étudiant compte dans les statistiques et dans
    l'analyse des questions (voir analytics.py)."""
    quiz = storage.get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(404, "Quiz introuvable.")
    return gradebook(quiz, storage.list_results(quiz_id))


# --------------------------------------------------------------------------- #
# Module Export
# --------------------------------------------------------------------------- #

@app.get("/quizzes/{quiz_id}/export/json")
def export_json_endpoint(
    quiz_id: str, include_answers: bool = True,
    current_user: UserInDB = Depends(professor_only),
):
    quiz = storage.get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(404, "Quiz introuvable.")
    path = export_quiz_json(quiz, include_answers=include_answers)
    return FileResponse(path, filename=path.name, media_type="application/json")


@app.get("/quizzes/{quiz_id}/export/pdf")
def export_pdf_endpoint(
    quiz_id: str, include_answers: bool = True,
    current_user: UserInDB = Depends(professor_only),
):
    quiz = storage.get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(404, "Quiz introuvable.")
    path = export_quiz_pdf(quiz, include_answers=include_answers)
    return FileResponse(path, filename=path.name, media_type="application/pdf")
