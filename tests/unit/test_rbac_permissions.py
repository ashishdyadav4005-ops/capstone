"""Unit tests for Role-Based Access Control (RBAC) permissions matrix."""

from __future__ import annotations

from src.auth.roles import (
    Permission,
    UserRole,
    get_role_permissions,
    has_permission,
)


class TestRbacPermissions:
    """Test suite for role permission mappings and authorization checks."""

    def test_all_four_roles_defined(self) -> None:
        """Verify all 4 required BDS-39 roles exist."""
        roles = [r.value for r in UserRole]
        assert "admin" in roles
        assert "analyst" in roles
        assert "manager" in roles
        assert "governance" in roles

    def test_admin_has_full_permissions(self) -> None:
        """Verify admin possesses administrative and operational permissions."""
        admin_perms = get_role_permissions(UserRole.ADMIN)
        assert Permission.USER_MANAGE in admin_perms
        assert Permission.SYSTEM_CONFIG in admin_perms
        assert Permission.AUDIT_VIEW in admin_perms
        assert Permission.PRICE_PUBLISH in admin_perms

    def test_analyst_permissions_contain_simulations(self) -> None:
        """Verify analyst can run simulations and optimizations, but cannot approve or publish."""
        analyst_perms = get_role_permissions(UserRole.ANALYST)
        assert Permission.SIMULATION_RUN in analyst_perms
        assert Permission.OPTIMIZATION_RUN in analyst_perms
        assert Permission.PRICING_READ in analyst_perms

        # Analyst cannot override or publish
        assert Permission.PRICE_PUBLISH not in analyst_perms
        assert Permission.RECOMMENDATION_APPROVE not in analyst_perms
        assert Permission.USER_MANAGE not in analyst_perms

    def test_manager_permissions_contain_approvals_and_overrides(self) -> None:
        """Verify manager can override, approve, reject, and publish prices."""
        manager_perms = get_role_permissions(UserRole.MANAGER)
        assert Permission.RECOMMENDATION_APPROVE in manager_perms
        assert Permission.RECOMMENDATION_OVERRIDE in manager_perms
        assert Permission.RECOMMENDATION_REJECT in manager_perms
        assert Permission.PRICE_PUBLISH in manager_perms
        assert Permission.EMERGENCY_ROLLBACK in manager_perms

        # Manager cannot manage users
        assert Permission.USER_MANAGE not in manager_perms

    def test_governance_permissions_contain_audits_and_rollback(self) -> None:
        """Verify governance role can audit logs and trigger emergency rollbacks."""
        gov_perms = get_role_permissions(UserRole.GOVERNANCE)
        assert Permission.AUDIT_VIEW in gov_perms
        assert Permission.FAIRNESS_AUDIT in gov_perms
        assert Permission.EMERGENCY_ROLLBACK in gov_perms

        # Governance cannot run raw optimization or publish prices
        assert Permission.OPTIMIZATION_RUN not in gov_perms
        assert Permission.PRICE_PUBLISH not in gov_perms

    def test_has_permission_helper(self) -> None:
        """Verify has_permission helper function accuracy."""
        assert has_permission(UserRole.ADMIN, Permission.USER_MANAGE)
        assert not has_permission(UserRole.ANALYST, Permission.USER_MANAGE)
        assert has_permission(UserRole.MANAGER, Permission.RECOMMENDATION_APPROVE)
        assert not has_permission(UserRole.GOVERNANCE, Permission.OPTIMIZATION_RUN)
