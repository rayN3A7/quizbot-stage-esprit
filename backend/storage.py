"""
Persistance.

Documents / quiz / résultats : fichiers JSON légers, suffisants pour un
projet de stage (voir README pour la piste d'évolution vers une base
complète si besoin).

Utilisateurs (authentification) : vraie base de données relationnelle via
SQLAlchemy (SQLite par défaut, PostgreSQL en changeant simplement
DATABASE_URL) — voir database.py et db_models.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .config import settings
from .models import DocumentInfo, Quiz, QuizResult, UserInDB

DOCUMENTS_INDEX = settings.DATA_DIR / "documents.json"


def _read_documents_index() -> dict:
    if DOCUMENTS_INDEX.exists():
        return json.loads(DOCUMENTS_INDEX.read_text(encoding="utf-8"))
    return {}


def _write_documents_index(data: dict) -> None:
    DOCUMENTS_INDEX.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def new_document_id() -> str:
    return uuid4().hex[:10]


def save_document_info(info: DocumentInfo) -> None:
    data = _read_documents_index()
    data[info.id] = json.loads(info.model_dump_json())
    _write_documents_index(data)


def get_document_info(document_id: str) -> Optional[DocumentInfo]:
    data = _read_documents_index()
    if document_id not in data:
        return None
    return DocumentInfo(**data[document_id])


def list_documents() -> list[DocumentInfo]:
    data = _read_documents_index()
    return [DocumentInfo(**v) for v in data.values()]


def save_quiz(quiz: Quiz) -> None:
    path = settings.QUIZZES_DIR / f"{quiz.id}.json"
    path.write_text(quiz.model_dump_json(indent=2), encoding="utf-8")


def get_quiz(quiz_id: str) -> Optional[Quiz]:
    path = settings.QUIZZES_DIR / f"{quiz_id}.json"
    if not path.exists():
        return None
    return Quiz(**json.loads(path.read_text(encoding="utf-8")))


def list_quizzes(published_only: bool = False) -> list[Quiz]:
    quizzes = []
    for path in settings.QUIZZES_DIR.glob("*.json"):
        quiz = Quiz(**json.loads(path.read_text(encoding="utf-8")))
        if not published_only or quiz.published:
            quizzes.append(quiz)
    return sorted(quizzes, key=lambda q: q.created_at, reverse=True)


def save_result(result: QuizResult) -> None:
    results_dir = settings.DATA_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"{result.quiz_id}_{result.student_name}_{result.submitted_at.timestamp():.0f}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def list_results(quiz_id: Optional[str] = None) -> list[dict]:
    """Retourne les résultats enregistrés, éventuellement filtrés par quiz.

    Retourne des dictionnaires bruts plutôt que des QuizResult : la carte
    sémantique n'a besoin que de `graded_answers`, et d'anciens fichiers
    peuvent avoir un schéma légèrement différent — un parsing strict les
    ferait échouer sans bénéfice.
    """
    results_dir = settings.DATA_DIR / "results"
    if not results_dir.exists():
        return []
    out: list[dict] = []
    for path in results_dir.glob("*.json"):
        try:
            out_data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if quiz_id is None or out_data.get("quiz_id") == quiz_id:
            out.append(out_data)
    return out


# --------------------------------------------------------------------------- #
# Utilisateurs (authentification) — stockés dans une vraie base de données
# relationnelle (SQLite/PostgreSQL via SQLAlchemy), pas en JSON.
# --------------------------------------------------------------------------- #

from .database import get_session
from .db_models import UserORM


def _to_pydantic(row: UserORM) -> UserInDB:
    return UserInDB(
        id=row.id, username=row.username, full_name=row.full_name,
        role=row.role, created_at=row.created_at, hashed_password=row.hashed_password,
    )


def new_user_id() -> str:
    return uuid4().hex[:10]


def username_exists(username: str) -> bool:
    with get_session() as db:
        return db.query(UserORM).filter(UserORM.username == username.lower()).first() is not None


def save_user(user: UserInDB) -> None:
    with get_session() as db:
        row = db.get(UserORM, user.id)
        if row is None:
            row = UserORM(id=user.id, username=user.username.lower())
        row.full_name = user.full_name
        row.role = user.role.value if hasattr(user.role, "value") else user.role
        row.hashed_password = user.hashed_password
        row.created_at = user.created_at
        db.add(row)
        db.commit()


def get_user(username: str) -> Optional[UserInDB]:
    with get_session() as db:
        row = db.query(UserORM).filter(UserORM.username == username.lower()).first()
        return _to_pydantic(row) if row else None


def list_users() -> list[UserInDB]:
    with get_session() as db:
        return [_to_pydantic(row) for row in db.query(UserORM).all()]
