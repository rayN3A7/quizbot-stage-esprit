"""
Connexion et session SQLAlchemy.

Base de données relationnelle réelle (SQLite par défaut, un seul fichier
`data/quizbot.db`, aucun serveur à installer). Le changement vers PostgreSQL
ou MySQL en production se fait uniquement en changeant DATABASE_URL dans
.env — aucun code ci-dessous ni ailleurs n'a besoin d'être modifié.
"""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

# `check_same_thread=False` est nécessaire uniquement pour SQLite (un seul
# fichier accédé depuis plusieurs threads, ex. FastAPI + Streamlit) ; ignoré
# par les autres moteurs (PostgreSQL, MySQL...).
_connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


@contextmanager
def get_session() -> Session:
    """Fournit une session SQLAlchemy et garantit sa fermeture (usage : `with get_session() as db:`)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Crée les tables si elles n'existent pas encore. Appelé au démarrage de l'API."""
    from . import db_models  # noqa: F401  (assure l'enregistrement des modèles auprès de Base)
    Base.metadata.create_all(bind=engine)
