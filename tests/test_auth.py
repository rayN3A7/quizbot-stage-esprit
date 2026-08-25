"""Tests unitaires du module d'authentification (hachage, JWT)."""
import time

import jwt
import pytest

from backend.auth import (
    AuthError, create_access_token, decode_access_token, hash_password, verify_password,
)
from backend.config import settings
from backend.models import Role


def test_hash_password_is_not_plaintext():
    hashed = hash_password("mysecret123")
    assert hashed != "mysecret123"
    assert verify_password("mysecret123", hashed)
    assert not verify_password("wrongpassword", hashed)


def test_same_password_hashes_differently_each_time():
    # bcrypt utilise un sel aléatoire : deux hachages du même mot de passe diffèrent.
    h1 = hash_password("samepassword")
    h2 = hash_password("samepassword")
    assert h1 != h2
    assert verify_password("samepassword", h1)
    assert verify_password("samepassword", h2)


def test_create_and_decode_access_token():
    token = create_access_token("rayen", Role.PROFESSOR)
    data = decode_access_token(token)
    assert data.username == "rayen"
    assert data.role == Role.PROFESSOR


def test_decode_invalid_token_raises():
    with pytest.raises(AuthError):
        decode_access_token("not.a.valid.token")


def test_decode_expired_token_raises():
    import datetime
    expired_payload = {
        "sub": "rayen", "role": "professeur",
        "exp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=1),
    }
    expired_token = jwt.encode(expired_payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    with pytest.raises(AuthError):
        decode_access_token(expired_token)


def test_decode_token_wrong_secret_raises():
    token = jwt.encode(
        {"sub": "rayen", "role": "professeur"}, "a-completely-different-secret", algorithm="HS256",
    )
    with pytest.raises(AuthError):
        decode_access_token(token)
