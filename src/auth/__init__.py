"""Auth package: RBAC, password security, JWT tokens, and account lockout."""

from src.auth.dependencies import (
    get_current_user,
    get_user_repository,
    require_permission,
    require_roles,
)
from src.auth.jwt import create_access_token, create_refresh_token, decode_token
from src.auth.repository import UserRecord, UserRepository
from src.auth.roles import (
    ROLE_PERMISSIONS,
    Permission,
    UserRole,
    get_role_permissions,
    has_permission,
)
from src.auth.security import hash_password, verify_password

__all__ = [
    "UserRole",
    "Permission",
    "ROLE_PERMISSIONS",
    "get_role_permissions",
    "has_permission",
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "UserRecord",
    "UserRepository",
    "get_current_user",
    "get_user_repository",
    "require_roles",
    "require_permission",
]
