"""Tests du stockage des utilisateurs en base de données réelle (SQLAlchemy)."""
from backend import storage
from backend.auth import hash_password
from backend.models import Role, UserInDB


def _make_user(username="dbuser", role=Role.STUDENT) -> UserInDB:
    return UserInDB(
        id=storage.new_user_id(), username=username, full_name="DB Test User",
        role=role, hashed_password=hash_password("secret123"),
    )


def test_save_and_get_user_roundtrip():
    user = _make_user("roundtrip_user")
    storage.save_user(user)

    fetched = storage.get_user("roundtrip_user")
    assert fetched is not None
    assert fetched.username == "roundtrip_user"
    assert fetched.role == Role.STUDENT
    assert fetched.hashed_password == user.hashed_password


def test_username_lookup_is_case_insensitive():
    storage.save_user(_make_user("CaseTest"))
    assert storage.get_user("casetest") is not None
    assert storage.username_exists("CASETEST") is True


def test_get_unknown_user_returns_none():
    assert storage.get_user("does_not_exist_at_all") is None


def test_save_user_upserts_existing_row():
    user = _make_user("upsert_user", role=Role.STUDENT)
    storage.save_user(user)

    # Même id, mais rôle/mot de passe modifiés : doit mettre à jour, pas dupliquer.
    user.role = Role.PROFESSOR
    user.hashed_password = hash_password("newpassword")
    storage.save_user(user)

    fetched = storage.get_user("upsert_user")
    assert fetched.role == Role.PROFESSOR
    assert fetched.hashed_password == user.hashed_password

    matches = [u for u in storage.list_users() if u.username == "upsert_user"]
    assert len(matches) == 1


def test_persisted_users_survive_new_session():
    # Vérifie que les données sont bien écrites sur disque (SQLite), pas juste en mémoire.
    storage.save_user(_make_user("persisted_user"))
    from backend.database import get_session
    from backend.db_models import UserORM

    with get_session() as db:
        row = db.query(UserORM).filter(UserORM.username == "persisted_user").first()
        assert row is not None
