"""FastAPI Authentication and Authorization dependencies."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.auth.jwt import decode_token
from src.auth.repository import UserRecord, UserRepository
from src.auth.roles import Permission, UserRole, has_permission

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


@lru_cache
def get_user_repository() -> UserRepository:
    """Provide singleton UserRepository."""
    return UserRepository()


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    repo: UserRepository = Depends(get_user_repository),
) -> UserRecord:
    """Validate Bearer JWT token, verify account status, and return authenticated user."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_token(token)
        username: str | None = payload.get("sub")
        token_type: str | None = payload.get("token_type")

        if not username or token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token payload.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    user = repo.get_by_username(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    if repo.is_locked(user):
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Account is temporarily locked until {user.locked_until} due to failed login attempts.",
        )

    return user


def require_roles(roles: list[UserRole] | UserRole) -> Callable[[UserRecord], UserRecord]:
    """Dependency factory restricting endpoint access to specific authorized roles."""
    allowed = {roles} if isinstance(roles, UserRole) else set(roles)

    def role_checker(current_user: UserRecord = Depends(get_current_user)) -> UserRecord:
        if current_user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: Role '{current_user.role.value}' is not authorized. Required: {[r.value for r in allowed]}",
            )
        return current_user

    return role_checker


def require_permission(permission: Permission) -> Callable[[UserRecord], UserRecord]:
    """Dependency factory restricting access based on granular permission."""
    def permission_checker(current_user: UserRecord = Depends(get_current_user)) -> UserRecord:
        if not has_permission(current_user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: User lacking required permission '{permission.value}'",
            )
        return current_user

    return permission_checker
