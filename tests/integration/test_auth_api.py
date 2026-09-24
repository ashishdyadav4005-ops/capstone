"""Integration tests for Authentication & RBAC API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.auth.repository import UserRecord, UserRepository
from src.auth.roles import UserRole
from src.auth.security import hash_password

client = TestClient(app)


@pytest.fixture(autouse=True)
def seed_test_users() -> None:
    """Ensure standard test users are available in SQLite for API tests."""
    repo = UserRepository()
    users = [
        ("admin", "Admin@12345", UserRole.ADMIN),
        ("analyst", "Analyst@12345", UserRole.ANALYST),
        ("manager", "Manager@12345", UserRole.MANAGER),
        ("governance", "Governance@12345", UserRole.GOVERNANCE),
    ]
    for uname, pwd, role in users:
        existing = repo.get_by_username(uname)
        if existing is None:
            u = UserRecord(
                user_id=f"USR_{uname.upper()}",
                username=uname,
                email=f"{uname}@company.internal",
                full_name=f"Test {uname.capitalize()}",
                hashed_password=hash_password(pwd),
                role=role,
                is_active=True,
            )
            repo.create_user(u)
        else:
            # Reset failure count and lockout
            repo.record_login_success(uname)


class TestAuthApi:
    """Integration test suite for Auth API."""

    def test_login_success_all_four_roles(self) -> None:
        """Verify successful login and token generation for all 4 roles."""
        creds = [
            ("admin", "Admin@12345", "admin"),
            ("analyst", "Analyst@12345", "analyst"),
            ("manager", "Manager@12345", "manager"),
            ("governance", "Governance@12345", "governance"),
        ]

        for username, password, expected_role in creds:
            res = client.post("/api/v1/auth/login", json={"username": username, "password": password})
            assert res.status_code == 200, f"Login failed for {username}: {res.text}"
            data = res.json()
            assert "access_token" in data
            assert "refresh_token" in data
            assert data["token_type"] == "bearer"
            assert data["role"] == expected_role
            assert len(data["permissions"]) > 0

    def test_login_invalid_password_returns_401(self) -> None:
        """Verify that wrong password returns 401 and warns about remaining attempts."""
        res = client.post("/api/v1/auth/login", json={"username": "analyst", "password": "WrongPassword"})
        assert res.status_code == 401
        assert "remaining before lockout" in res.json()["detail"]

    def test_get_current_user_profile_me(self) -> None:
        """Verify /me profile endpoint with valid and missing tokens."""
        # 1. Login as Analyst
        login_res = client.post("/api/v1/auth/login", json={"username": "analyst", "password": "Analyst@12345"})
        token = login_res.json()["access_token"]

        # 2. Access /me with Bearer token
        res_me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res_me.status_code == 200
        data_me = res_me.json()
        assert data_me["username"] == "analyst"
        assert data_me["role"] == "analyst"
        assert "simulation:run" in data_me["permissions"]

        # 3. Access /me without token -> 401 Unauthorized
        res_unauth = client.get("/api/v1/auth/me")
        assert res_unauth.status_code == 401

    def test_token_refresh_flow(self) -> None:
        """Verify refresh token exchange for a new access token."""
        login_res = client.post("/api/v1/auth/login", json={"username": "manager", "password": "Manager@12345"})
        refresh_tok = login_res.json()["refresh_token"]

        res_refresh = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_tok})
        assert res_refresh.status_code == 200
        data = res_refresh.json()
        assert "access_token" in data

    def test_admin_endpoints_rbac_enforcement(self) -> None:
        """Verify that /users is accessible to admin but forbidden (403) to analyst."""
        # 1. Analyst token
        analyst_tok = client.post("/api/v1/auth/login", json={"username": "analyst", "password": "Analyst@12345"}).json()["access_token"]
        res_forbidden = client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {analyst_tok}"})
        assert res_forbidden.status_code == 403

        # 2. Admin token
        admin_tok = client.post("/api/v1/auth/login", json={"username": "admin", "password": "Admin@12345"}).json()["access_token"]
        res_admin = client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {admin_tok}"})
        assert res_admin.status_code == 200
        users_list = res_admin.json()
        assert len(users_list) >= 4

    def test_admin_create_user_and_unlock(self) -> None:
        """Verify admin user creation and unlock endpoint."""
        admin_tok = client.post("/api/v1/auth/login", json={"username": "admin", "password": "Admin@12345"}).json()["access_token"]

        new_user_payload = {
            "username": "new_analyst_user",
            "email": "newanalyst@company.internal",
            "full_name": "New Analyst",
            "password": "NewUser@12345",
            "role": "analyst",
        }
        res_create = client.post(
            "/api/v1/auth/users",
            json=new_user_payload,
            headers={"Authorization": f"Bearer {admin_tok}"},
        )
        assert res_create.status_code in [201, 400] # 201 created or 400 if already exists

        # Unlock user endpoint
        repo = UserRepository()
        created = repo.get_by_username("new_analyst_user")
        if created:
            res_unlock = client.post(
                f"/api/v1/auth/users/{created.user_id}/unlock",
                headers={"Authorization": f"Bearer {admin_tok}"},
            )
            assert res_unlock.status_code == 200
            assert res_unlock.json()["status"] == "unlocked"
