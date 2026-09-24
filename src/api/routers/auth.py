"""Authentication and User Management Router."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.audit.logger import AuditLogger
from src.audit.models import AuditEventType
from src.audit.repository import AuditRepository
from src.auth.dependencies import (
    get_current_user,
    get_user_repository,
    require_roles,
)
from src.auth.jwt import create_access_token, create_refresh_token, decode_token
from src.auth.repository import UserRecord, UserRepository
from src.auth.roles import UserRole, get_role_permissions
from src.auth.security import hash_password, verify_password

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication & RBAC"])


def get_audit_logger() -> AuditLogger:
    """Provide singleton AuditLogger."""
    return AuditLogger(repository=AuditRepository())


class LoginRequest(BaseModel):
    """User credentials for login."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT Token credentials and user metadata response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    full_name: str
    role: UserRole
    permissions: list[str]


class RefreshTokenRequest(BaseModel):
    """Token refresh payload."""

    refresh_token: str


class UserProfileResponse(BaseModel):
    """Authenticated user profile metadata."""

    user_id: str
    username: str
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    permissions: list[str]


class CreateUserRequest(BaseModel):
    """Admin payload to create a new user."""

    username: str = Field(min_length=3, max_length=30)
    email: str
    full_name: str
    password: str = Field(min_length=6)
    role: UserRole


@router.post("/login", response_model=TokenResponse)
def login(
    req: LoginRequest,
    repo: UserRepository = Depends(get_user_repository),
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> TokenResponse:
    """Authenticate with username and password, return JWT access and refresh tokens."""
    username = req.username.strip().lower()
    user = repo.get_by_username(username)

    if user is None:
        try:
            audit_logger.log_security_event(
                event_type=AuditEventType.USER_LOGIN_FAILED,
                actor_username=username,
                actor_role="unknown",
                details={"reason": "User not found"},
            )
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if account is locked
    if repo.is_locked(user):
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Account is temporarily locked until {user.locked_until} due to consecutive failed attempts.",
        )

    # Verify password
    if not verify_password(req.password, user.hashed_password):
        failed_count, is_locked = repo.record_login_failure(username)
        if is_locked:
            try:
                audit_logger.log_security_event(
                    event_type=AuditEventType.USER_LOCKED_OUT,
                    actor_username=username,
                    actor_role=user.role.value,
                    details={"failed_attempts": failed_count, "lockout_minutes": repo.lockout_duration_minutes},
                )
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account locked for {repo.lockout_duration_minutes} minutes after {failed_count} failed attempts.",
            )
        try:
            audit_logger.log_security_event(
                event_type=AuditEventType.USER_LOGIN_FAILED,
                actor_username=username,
                actor_role=user.role.value,
                details={"failed_attempts": failed_count},
            )
        except Exception:
            pass
        remaining = max(0, repo.max_failed_logins - failed_count)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid credentials. {remaining} attempt(s) remaining before lockout.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has been deactivated. Please contact an administrator.",
        )

    # Reset failure counter and update last login
    repo.record_login_success(username)

    try:
        audit_logger.log_security_event(
            event_type=AuditEventType.USER_LOGIN_SUCCESS,
            actor_username=user.username,
            actor_role=user.role.value,
            details={"user_id": user.user_id},
        )
    except Exception:
        pass

    # Generate JWT tokens
    token_payload = {
        "sub": user.username,
        "user_id": user.user_id,
        "role": user.role.value,
    }
    access_tok = create_access_token(token_payload)
    refresh_tok = create_refresh_token(token_payload)

    perms = [p.value for p in get_role_permissions(user.role)]

    return TokenResponse(
        access_token=access_tok,
        refresh_token=refresh_tok,
        token_type="bearer",
        user_id=user.user_id,
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        permissions=perms,
    )


@router.post("/refresh", response_model=dict[str, str])
def refresh_token(
    req: RefreshTokenRequest,
    repo: UserRepository = Depends(get_user_repository),
) -> dict[str, str]:
    """Exchange a valid refresh token for a new access token."""
    try:
        payload = decode_token(req.refresh_token)
        username = payload.get("sub")
        token_type = payload.get("token_type")

        if not username or token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token.",
            )

        user = repo.get_by_username(username)
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User inactive or not found.",
            )

        new_access_token = create_access_token({
            "sub": user.username,
            "user_id": user.user_id,
            "role": user.role.value,
        })
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        ) from e


@router.get("/me", response_model=UserProfileResponse)
def get_current_user_profile(
    current_user: UserRecord = Depends(get_current_user),
) -> UserProfileResponse:
    """Return the profile and authorized permissions of the currently authenticated user."""
    perms = [p.value for p in get_role_permissions(current_user.role)]
    return UserProfileResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        permissions=perms,
    )


@router.get("/users", response_model=list[dict])
def list_users(
    _admin: UserRecord = Depends(require_roles(UserRole.ADMIN)),
    repo: UserRepository = Depends(get_user_repository),
) -> list[dict]:
    """Admin-only endpoint to list all system users."""
    users = repo.list_users()
    return [u.to_dict() for u in users]


@router.post("/users", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_user(
    req: CreateUserRequest,
    admin: UserRecord = Depends(require_roles(UserRole.ADMIN)),
    repo: UserRepository = Depends(get_user_repository),
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> dict:
    """Admin-only endpoint to create a new authorized user."""
    existing = repo.get_by_username(req.username)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{req.username}' already exists.",
        )

    user = UserRecord(
        user_id=f"USR_{uuid.uuid4().hex[:8].upper()}",
        username=req.username,
        email=req.email,
        full_name=req.full_name,
        hashed_password=hash_password(req.password),
        role=req.role,
        is_active=True,
    )
    repo.create_user(user)

    try:
        audit_logger.log_security_event(
            event_type=AuditEventType.USER_CREATED,
            actor_username=admin.username,
            actor_role=admin.role.value,
            details={"created_user_id": user.user_id, "created_username": user.username, "role": user.role.value},
        )
    except Exception:
        pass

    return user.to_dict()


@router.post("/users/{user_id}/unlock", response_model=dict)
def unlock_user(
    user_id: str,
    admin: UserRecord = Depends(require_roles(UserRole.ADMIN)),
    repo: UserRepository = Depends(get_user_repository),
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> dict:
    """Admin-only endpoint to manually unlock a locked user account."""
    success = repo.unlock_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"User ID '{user_id}' not found.")

    try:
        audit_logger.log_security_event(
            event_type=AuditEventType.USER_UNLOCKED,
            actor_username=admin.username,
            actor_role=admin.role.value,
            details={"unlocked_user_id": user_id},
        )
    except Exception:
        pass

    return {"status": "unlocked", "user_id": user_id}
