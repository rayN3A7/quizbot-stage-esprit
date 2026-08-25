"""
Authentification : hachage des mots de passe (bcrypt) et jetons JWT.

Deux rôles sont supportés : "professeur" et "etudiant" (voir Role dans
models.py). Les dépendances FastAPI `get_current_user` et `require_role`
permettent de protéger chaque endpoint selon le rôle attendu.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from . import storage
from .config import settings
from .models import Role, TokenData, UserInDB

# L'URL "tokenUrl" sert uniquement à la documentation Swagger (/docs) pour
# savoir où obtenir un jeton ; l'authentification réelle se fait via
# POST /auth/login.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


class AuthError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Mots de passe
# --------------------------------------------------------------------------- #

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


# --------------------------------------------------------------------------- #
# JWT
# --------------------------------------------------------------------------- #

def create_access_token(username: str, role: Role) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "role": role.value, "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise AuthError("Le jeton a expiré, merci de vous reconnecter.")
    except jwt.InvalidTokenError:
        raise AuthError("Jeton d'authentification invalide.")

    username = payload.get("sub")
    role = payload.get("role")
    if not username or not role:
        raise AuthError("Jeton d'authentification invalide.")
    return TokenData(username=username, role=Role(role))


# --------------------------------------------------------------------------- #
# Dépendances FastAPI
# --------------------------------------------------------------------------- #

def authenticate_user(username: str, password: str) -> UserInDB | None:
    user = storage.get_user(username)
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> UserInDB:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Impossible de valider les identifiants.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token is None:
        raise credentials_error
    try:
        token_data = decode_access_token(token)
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = storage.get_user(token_data.username)
    if user is None:
        raise credentials_error
    return user


def require_role(*allowed_roles: Role):
    """Fabrique une dépendance FastAPI qui n'autorise que les rôles donnés."""

    async def _checker(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Accès réservé au(x) rôle(s) : {', '.join(r.value for r in allowed_roles)}.",
            )
        return current_user

    return _checker
