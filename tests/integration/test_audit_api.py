"""Integration tests for Audit Trail API, Cryptographic Verification, and Governance Endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.auth.jwt import create_access_token
from src.auth.repository import UserRecord, UserRepository
from src.auth.roles import UserRole
from src.auth.security import hash_password


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


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
            repo.record_login_success(uname)


@pytest.fixture
def auth_headers():
    """Generate auth headers for each persona role."""
    def _make_header(role: UserRole):
        username = role.value
        token = create_access_token({
            "sub": username,
            "user_id": f"USR_{role.value.upper()}",
            "role": role.value,
        })
        return {"Authorization": f"Bearer {token}"}

    return _make_header


class TestAuditAndGovernanceApi:
    """Integration test suite for audit and governance routes."""

    def test_audit_logs_rbac_access(self, client: TestClient, auth_headers):
        """Verify role-based authorization rules on audit logs endpoint."""
        # 1. Unauthenticated -> 401
        res_unauth = client.get("/api/v1/audit/logs")
        assert res_unauth.status_code == 401

        # 2. Analyst -> 403 (Analyst is not allowed to inspect audit logs)
        res_analyst = client.get("/api/v1/audit/logs", headers=auth_headers(UserRole.ANALYST))
        assert res_analyst.status_code == 403

        # 3. Governance -> 200 OK
        res_gov = client.get("/api/v1/audit/logs", headers=auth_headers(UserRole.GOVERNANCE))
        assert res_gov.status_code == 200
        assert isinstance(res_gov.json(), list)

        # 4. Admin -> 200 OK
        res_admin = client.get("/api/v1/audit/logs", headers=auth_headers(UserRole.ADMIN))
        assert res_admin.status_code == 200

        # 5. Manager -> 200 OK
        res_mgr = client.get("/api/v1/audit/logs", headers=auth_headers(UserRole.MANAGER))
        assert res_mgr.status_code == 200

    def test_audit_verification_endpoint(self, client: TestClient, auth_headers):
        """Verify cryptographic chain verification endpoint returns valid integrity status."""
        headers = auth_headers(UserRole.GOVERNANCE)
        res = client.get("/api/v1/audit/verify", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert "is_valid" in data
        assert data["is_valid"] is True
        assert "total_entries" in data
        assert "error_message" in data

    def test_audit_summary_endpoint(self, client: TestClient, auth_headers):
        """Verify audit summary endpoint metadata."""
        headers = auth_headers(UserRole.ADMIN)
        res = client.get("/api/v1/audit/summary", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert "total_records" in data
        assert isinstance(data["total_records"], int)

    def test_governance_fairness_and_model_card_endpoints(self, client: TestClient, auth_headers):
        """Verify governance endpoints return proper structured reports."""
        gov_headers = auth_headers(UserRole.GOVERNANCE)

        # Fairness report
        res_fair = client.get("/api/v1/governance/fairness-report", headers=gov_headers)
        assert res_fair.status_code == 200
        fair_data = res_fair.json()
        assert "overall_compliant" in fair_data
        assert "max_observed_disparity_pct" in fair_data
        assert "products" in fair_data

        # Model card
        res_card = client.get("/api/v1/governance/model-card", headers=gov_headers)
        assert res_card.status_code == 200
        card_data = res_card.json()
        assert card_data["cross_fitting_folds"] == 5
        assert "ground_truth_bias_comparison" in card_data
        assert "sensitivity_stress_tests" in card_data

    def test_pricing_lifecycle_triggers_audit_chain_entries(self, client: TestClient, auth_headers):
        """Verify recommendation optimization, approval, override, and rollback append to audit chain."""
        # 1. Run optimization batch
        opt_req = {
            "items": [
                {
                    "item_id": "AUDIT_TEST_01",
                    "product_id": "PROD_001",
                    "customer_group": "premium",
                    "location_id": "us-east-1",
                    "base_price": 100.0,
                    "base_demand": 50.0,
                    "unit_cost": 60.0,
                    "elasticity": -1.5,
                    "capacity": 100.0,
                }
            ],
            "objective_type": "profit",
        }
        res_opt = client.post("/api/v1/pricing/optimize", json=opt_req)
        assert res_opt.status_code == 200

        # Get the generated recommendation ID
        recs_res = client.get("/api/v1/recommendations?product_id=PROD_001&customer_group=premium")
        assert recs_res.status_code == 200
        recs = recs_res.json()
        assert len(recs) > 0
        rec_id = recs[0]["recommendation_id"]

        # 2. Approve recommendation
        res_app = client.post(
            f"/api/v1/recommendations/{rec_id}/approve",
            json={"actor": "manager", "notes": "Approved for testing"},
        )
        assert res_app.status_code == 200

        # 3. Publish recommendation
        res_pub = client.post(
            f"/api/v1/recommendations/{rec_id}/publish",
            json={"actor": "system", "notes": "Published to catalog"},
        )
        assert res_pub.status_code == 200

        # 4. Emergency rollback
        res_rb = client.post(
            f"/api/v1/recommendations/{rec_id}/rollback",
            json={"actor": "manager", "notes": "Emergency rollback test"},
        )
        assert res_rb.status_code == 200

        # 5. Verify cryptographic audit chain is valid and includes all actions
        gov_headers = auth_headers(UserRole.GOVERNANCE)
        res_verify = client.get("/api/v1/audit/verify", headers=gov_headers)
        assert res_verify.status_code == 200
        v_data = res_verify.json()
        assert v_data["is_valid"] is True

        # 6. Check audit logs list
        res_logs = client.get("/api/v1/audit/logs?limit=10", headers=gov_headers)
        assert res_logs.status_code == 200
        logs = res_logs.json()
        event_types = [entry["event_type"] for entry in logs]
        assert "PRICE_APPROVED" in event_types or "EMERGENCY_ROLLBACK" in event_types
