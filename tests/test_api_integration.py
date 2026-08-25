"""
Test d'intégration bout-en-bout de l'API FastAPI, authentification comprise.

La base vectorielle (ChromaDB + sentence-transformers) et le modèle
d'embedding sont simulés (mock) afin que ce test s'exécute rapidement et
sans dépendance lourde (torch) ; le LLM utilise le fournisseur "mock" natif
de QuizBot, ce qui teste réellement le pipeline RAG "retrieval -> prompt ->
génération -> parsing -> stockage -> correction -> export" ainsi que le
contrôle d'accès par rôle (professeur / étudiant).
"""
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


class FakeVectorStore:
    """Simule ChromaDB : stocke les chunks en mémoire et renvoie les k premiers."""

    def __init__(self):
        self.data = {}

    def index_document(self, document_id, chunks):
        self.data[document_id] = chunks
        return len(chunks)

    def query(self, document_id, query_text, top_k=6):
        chunks = self.data.get(document_id, [])
        return [{"text": c.text, "metadata": {}, "distance": 0.1} for c in chunks[:top_k]]


def _fake_embed_text(text: str):
    vec = np.zeros(64)
    for w in text.lower().split():
        vec[hash(w) % 64] += 1
    return vec.tolist()


@pytest.fixture
def client():
    fake_store = FakeVectorStore()
    with patch("backend.main.get_vector_store", return_value=fake_store), \
         patch("backend.quiz_generator.get_vector_store", return_value=fake_store), \
         patch("backend.grading.embed_text", side_effect=_fake_embed_text):
        from backend.main import app
        yield TestClient(app)


def _auth_headers(client, username, role, password="secret123", full_name=""):
    """Enregistre (si besoin) et connecte un utilisateur, retourne ses headers Bearer."""
    client.post("/auth/register", json={
        "username": username, "password": password, "full_name": full_name, "role": role,
    })
    r = client.post("/auth/login", data={"username": username, "password": password})
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_register_and_login(client):
    r = client.post("/auth/register", json={
        "username": "prof_test", "password": "secret123", "full_name": "Prof Test", "role": "professeur",
    })
    assert r.status_code == 201
    assert r.json()["role"] == "professeur"
    assert "hashed_password" not in r.json()

    r = client.post("/auth/login", data={"username": "prof_test", "password": "secret123"})
    assert r.status_code == 200
    assert r.json()["role"] == "professeur"

    r = client.post("/auth/login", data={"username": "prof_test", "password": "wrong"})
    assert r.status_code == 401


def test_duplicate_username_rejected(client):
    client.post("/auth/register", json={"username": "dup", "password": "secret123", "role": "etudiant"})
    r = client.post("/auth/register", json={"username": "dup", "password": "secret123", "role": "professeur"})
    assert r.status_code == 400


def test_short_password_rejected(client):
    r = client.post("/auth/register", json={"username": "shortpw", "password": "123", "role": "etudiant"})
    assert r.status_code == 400


def test_endpoints_require_authentication(client):
    r = client.get("/documents")
    assert r.status_code == 401
    r = client.get("/quizzes")
    assert r.status_code == 401


def test_student_cannot_access_professor_endpoints(client):
    stud_headers = _auth_headers(client, "stud_only", "etudiant")
    r = client.get("/documents", headers=stud_headers)
    assert r.status_code == 403
    r = client.post("/quizzes/generate", headers=stud_headers, json={"document_id": "x", "num_questions": 3})
    assert r.status_code == 403


def test_full_teacher_and_student_flow(client, sample_pdf):
    prof_headers = _auth_headers(client, "prof_flow", "professeur", full_name="Prof Flow")
    stud_headers = _auth_headers(client, "stud_flow", "etudiant", full_name="Student Flow")

    # 1. Téléversement (professeur uniquement)
    with open(sample_pdf, "rb") as f:
        r = client.post("/documents/upload", headers=prof_headers,
                         files={"file": ("cours.pdf", f, "application/pdf")})
    assert r.status_code == 200
    doc = r.json()
    assert doc["num_chunks"] >= 1
    assert doc["uploaded_by"] == "prof_flow"

    # 2. Génération du quiz (RAG complet avec LLM mock)
    r = client.post("/quizzes/generate", headers=prof_headers, json={
        "document_id": doc["id"], "title": "Quiz Auto", "num_questions": 4,
        "question_type": "mélange", "difficulty": "moyen",
    })
    assert r.status_code == 200
    quiz = r.json()
    assert len(quiz["questions"]) == 4
    assert quiz["published"] is False
    assert quiz["created_by"] == "prof_flow"

    # 3. Un quiz non publié n'est pas visible/soumissible pour les étudiants
    r = client.get(f"/quizzes/{quiz['id']}", headers=stud_headers)
    assert r.status_code == 403
    r = client.post(f"/quizzes/{quiz['id']}/submit", headers=stud_headers, json={
        "quiz_id": quiz["id"], "answers": [],
    })
    assert r.status_code == 403

    # 4. Publication (professeur uniquement)
    r = client.post(f"/quizzes/{quiz['id']}/publish", headers=stud_headers)
    assert r.status_code == 403
    r = client.post(f"/quizzes/{quiz['id']}/publish", headers=prof_headers)
    assert r.status_code == 200
    assert r.json()["published"] is True

    # 5. Soumission des réponses (toutes correctes) — le nom vient du compte connecté
    answers = []
    for q in quiz["questions"]:
        if q["type"] == "qcm":
            answers.append({"question_id": q["id"], "answer": str(q["correct_choice_index"])})
        else:
            answers.append({"question_id": q["id"], "answer": q["reference_answer"]})

    # Un professeur ne peut pas soumettre de réponses (rôle étudiant requis)
    r = client.post(f"/quizzes/{quiz['id']}/submit", headers=prof_headers, json={
        "quiz_id": quiz["id"], "answers": answers,
    })
    assert r.status_code == 403

    r = client.post(f"/quizzes/{quiz['id']}/submit", headers=stud_headers, json={
        "quiz_id": quiz["id"], "answers": answers,
    })
    assert r.status_code == 200
    result = r.json()
    assert result["percentage"] == 100.0
    assert result["student_name"] == "Student Flow"

    # 6. Exports (professeur uniquement)
    r = client.get(f"/quizzes/{quiz['id']}/export/pdf", headers=stud_headers)
    assert r.status_code == 403
    r = client.get(f"/quizzes/{quiz['id']}/export/pdf", headers=prof_headers)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"

    r = client.get(f"/quizzes/{quiz['id']}/export/json", headers=prof_headers)
    assert r.status_code == 200


def test_generate_quiz_unknown_document_returns_404(client):
    prof_headers = _auth_headers(client, "prof_404", "professeur")
    r = client.post("/quizzes/generate", headers=prof_headers, json={"document_id": "unknown", "num_questions": 3})
    assert r.status_code == 404


def test_upload_rejects_unsupported_format(client, tmp_path):
    prof_headers = _auth_headers(client, "prof_badfmt", "professeur")
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello")
    with open(bad_file, "rb") as f:
        r = client.post("/documents/upload", headers=prof_headers,
                         files={"file": ("notes.txt", f, "text/plain")})
    assert r.status_code == 400


def test_professor_signup_code_enforced(client):
    with patch("backend.main.settings.PROFESSOR_SIGNUP_CODE", "letmein"):
        r = client.post("/auth/register", json={
            "username": "prof_needs_code", "password": "secret123", "role": "professeur",
        })
        assert r.status_code == 403

        r = client.post("/auth/register", json={
            "username": "prof_with_code", "password": "secret123", "role": "professeur",
            "professor_code": "letmein",
        })
        assert r.status_code == 201

        # Un étudiant n'a pas besoin du code
        r = client.post("/auth/register", json={
            "username": "student_no_code_needed", "password": "secret123", "role": "etudiant",
        })
        assert r.status_code == 201
