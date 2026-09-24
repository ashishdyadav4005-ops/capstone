"""Integration tests for Prometheus metrics exporter and Drift Monitoring REST API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.auth.jwt import create_access_token
from src.auth.repository import UserRecord, UserRepository
from src.auth.roles import UserRole
from src.auth.security import hash_password
from src.monitoring.metrics import PROMETHEUS_CONTENT_TYPE


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


@pytest.fixture
def auth_headers():
    """Generate auth headers for each persona role."""
    def _make_header(role: UserRole):
        token = create_access_token({
            "sub": role.value,
            "user_id": f"USR_{role.value.upper()}",
            "role": role.value,
        })
        return {"Authorization": f"Bearer {token}"}

    return _make_header


class TestMonitoringApi:
    """Integration test suite for monitoring and observability endpoints."""

    def test_root_metrics_endpoint(self, client: TestClient):
        """Verify root /metrics endpoint exposes valid Prometheus plain-text metrics."""
        res = client.get("/metrics")
        assert res.status_code == 200
        assert PROMETHEUS_CONTENT_TYPE.split(";")[0] in res.headers["content-type"]
        body = res.text
        assert "# TYPE pricing_requests_total counter" in body
        assert "# TYPE audit_chain_valid gauge" in body
        assert "# TYPE optimizer_solve_duration_seconds histogram" in body

    def test_monitoring_router_metrics_endpoint(self, client: TestClient):
        """Verify /api/v1/monitoring/metrics endpoint returns Prometheus telemetry."""
        res = client.get("/api/v1/monitoring/metrics")
        assert res.status_code == 200
        assert "pricing_requests_total" in res.text

    def test_monitoring_health_endpoint(self, client: TestClient):
        """Verify /api/v1/monitoring/health provides collector status."""
        res = client.get("/api/v1/monitoring/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["metrics_exporter"] == "active"
        assert data["drift_engine"] == "active"
        assert data["audit_chain_valid"] is True

    def test_drift_check_endpoint(self, client: TestClient, auth_headers):
        """Verify on-demand drift detection execution and response structure."""
        headers = auth_headers(UserRole.ANALYST)

        payload = {
            "features": ["price", "quantity", "competitor_price"],
            "psi_warning_threshold": 0.10,
            "psi_critical_threshold": 0.25,
        }

        res = client.post("/api/v1/monitoring/drift/check", json=payload, headers=headers)
        assert res.status_code == 200
        data = res.json()

        assert "overall_status" in data
        assert data["overall_status"] in ["STABLE", "WARNING", "CRITICAL"]
        assert len(data["features"]) == 3
        assert data["reference_sample_size"] > 0
        assert data["current_sample_size"] > 0

        # Verify individual feature drift result
        price_feat = next(f for f in data["features"] if f["feature_name"] == "price")
        assert "psi_score" in price_feat
        assert "wasserstein_distance" in price_feat
        assert "ks_statistic" in price_feat

    def test_drift_latest_endpoint(self, client: TestClient, auth_headers):
        """Verify /api/v1/monitoring/drift/latest returns cached or evaluated report."""
        headers = auth_headers(UserRole.GOVERNANCE)
        res = client.get("/api/v1/monitoring/drift/latest", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert "overall_status" in data
        assert len(data["features"]) > 0

    def test_drift_check_unauthenticated_forbidden(self, client: TestClient):
        """Verify unauthenticated requests to drift check are rejected with 401."""
        res = client.post("/api/v1/monitoring/drift/check", json={})
        assert res.status_code == 401
