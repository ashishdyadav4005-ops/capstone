"""JSON Web Token (JWT) encoding and verification using PyJWT."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from jwt.exceptions import PyJWTError

from src.common.config import get_config

ALGORITHM = "HS256"


def get_jwt_secret() -> str:
    """Retrieve secret key from configuration."""
    config = get_config()
    return getattr(config.auth, "jwt_secret_key", "dev-secret-key-please-change-in-production-12345678")


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    now = datetime.now(UTC)
    if expires_delta:
        expire = now + expires_delta
    else:
        config = get_config()
        expire_minutes = getattr(config.auth, "access_token_expire_minutes", 60)
        expire = now + timedelta(minutes=expire_minutes)

    to_encode.update({
        "exp": expire,
        "iat": now,
        "token_type": "access",
    })
    secret_key = get_jwt_secret()
    return jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)


def create_refresh_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT refresh token."""
    to_encode = data.copy()
    now = datetime.now(UTC)
    if expires_delta:
        expire = now + expires_delta
    else:
        # Default 7 days
        expire = now + timedelta(days=7)

    to_encode.update({
        "exp": expire,
        "iat": now,
        "token_type": "refresh",
    })
    secret_key = get_jwt_secret()
    return jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a signed JWT token."""
    secret_key = get_jwt_secret()
    try:
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
        return payload
    except PyJWTError as e:
        raise ValueError(f"Invalid or expired token: {e}") from e
