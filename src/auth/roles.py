"""Role-Based Access Control (RBAC) definitions and permissions matrix."""

from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    """The four distinct authorized roles for BDS-39 decision portal."""

    ADMIN = "admin"
    ANALYST = "analyst"
    MANAGER = "manager"
    GOVERNANCE = "governance"


class Permission(StrEnum):
    """Granular system permissions."""

    # Pricing & Simulation permissions
    PRICING_READ = "pricing:read"
    SIMULATION_RUN = "simulation:run"
    OPTIMIZATION_RUN = "optimization:run"

    # Recommendation lifecycle permissions
    RECOMMENDATION_VIEW = "recommendation:view"
    RECOMMENDATION_REVIEW = "recommendation:review"
    RECOMMENDATION_OVERRIDE = "recommendation:override"
    RECOMMENDATION_APPROVE = "recommendation:approve"
    RECOMMENDATION_REJECT = "recommendation:reject"
    PRICE_PUBLISH = "price:publish"
    EMERGENCY_ROLLBACK = "price:rollback"

    # Audit, Governance, and Admin permissions
    AUDIT_VIEW = "audit:view"
    FAIRNESS_AUDIT = "fairness:audit"
    USER_MANAGE = "user:manage"
    SYSTEM_CONFIG = "system:config"


# Role-to-Permissions Mapping Matrix
ROLE_PERMISSIONS: dict[UserRole, set[Permission]] = {
    UserRole.ADMIN: {
        Permission.PRICING_READ,
        Permission.SIMULATION_RUN,
        Permission.OPTIMIZATION_RUN,
        Permission.RECOMMENDATION_VIEW,
        Permission.RECOMMENDATION_REVIEW,
        Permission.RECOMMENDATION_OVERRIDE,
        Permission.RECOMMENDATION_APPROVE,
        Permission.RECOMMENDATION_REJECT,
        Permission.PRICE_PUBLISH,
        Permission.EMERGENCY_ROLLBACK,
        Permission.AUDIT_VIEW,
        Permission.FAIRNESS_AUDIT,
        Permission.USER_MANAGE,
        Permission.SYSTEM_CONFIG,
    },
    UserRole.ANALYST: {
        Permission.PRICING_READ,
        Permission.SIMULATION_RUN,
        Permission.OPTIMIZATION_RUN,
        Permission.RECOMMENDATION_VIEW,
        Permission.RECOMMENDATION_REVIEW,
        Permission.FAIRNESS_AUDIT,
    },
    UserRole.MANAGER: {
        Permission.PRICING_READ,
        Permission.SIMULATION_RUN,
        Permission.OPTIMIZATION_RUN,
        Permission.RECOMMENDATION_VIEW,
        Permission.RECOMMENDATION_REVIEW,
        Permission.RECOMMENDATION_OVERRIDE,
        Permission.RECOMMENDATION_APPROVE,
        Permission.RECOMMENDATION_REJECT,
        Permission.PRICE_PUBLISH,
        Permission.EMERGENCY_ROLLBACK,
        Permission.AUDIT_VIEW,
    },
    UserRole.GOVERNANCE: {
        Permission.PRICING_READ,
        Permission.RECOMMENDATION_VIEW,
        Permission.AUDIT_VIEW,
        Permission.FAIRNESS_AUDIT,
        Permission.EMERGENCY_ROLLBACK,
    },
}


def get_role_permissions(role: UserRole) -> set[Permission]:
    """Retrieve all permissions granted to a given role."""
    return ROLE_PERMISSIONS.get(role, set())


def has_permission(role: UserRole, permission: Permission) -> bool:
    """Check if a specific role possesses a permission."""
    return permission in get_role_permissions(role)
