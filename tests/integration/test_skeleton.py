"""Integration smoke tests verifying repository skeleton and package structure."""

import importlib

from fastapi.testclient import TestClient

from src.api.main import app


def test_package_imports():
    """Verify all subpackages in src/ can be imported without errors."""
    packages = [
        "src.common",
        "src.data",
        "src.models",
        "src.pricing",
        "src.api",
        "src.auth",
        "src.audit",
        "src.monitoring",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None


def test_fastapi_root_and_health_endpoints():
    """Verify FastAPI root and healthcheck endpoints respond with status code 200."""
    client = TestClient(app)

    # Root endpoint
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data

    # Health endpoint
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "healthy"
