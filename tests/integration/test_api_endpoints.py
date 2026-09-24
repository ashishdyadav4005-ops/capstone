"""Integration tests for FastAPI REST endpoints using TestClient."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestApiEndpoints:
    """Integration test suite for all FastAPI REST endpoints."""

    def test_health_and_root_endpoints(self) -> None:
        """Verify root and health check endpoints."""
        # 1. Root
        r_root = client.get("/")
        assert r_root.status_code == 200
        data_root = r_root.json()
        assert data_root["status"] == "online"
        assert "version" in data_root

        # 2. Health
        r_health = client.get("/health")
        assert r_health.status_code == 200
        data_health = r_health.json()
        assert data_health["status"] in ["healthy", "degraded"]
        assert data_health["database"] == "connected"

        # 3. API v1 Health
        r_v1_health = client.get("/api/v1/health")
        assert r_v1_health.status_code == 200

    def test_catalog_endpoint(self) -> None:
        """Verify catalog discovery endpoint."""
        res = client.get("/api/v1/pricing/catalog")
        assert res.status_code == 200
        data = res.json()
        assert len(data["products"]) >= 5
        assert "PROD_001" in [p["product_id"] for p in data["products"]]
        assert "budget" in data["customer_groups"]

    def test_pricing_curves_endpoint(self) -> None:
        """Verify single counterfactual demand curve generation."""
        payload = {
            "product_id": "PROD_001",
            "customer_group": "regular",
            "location_id": "LOC_URBAN",
            "base_price": 100.0,
            "base_demand": 50.0,
            "unit_cost": 60.0,
            "grid_points": 40,
        }
        res = client.post("/api/v1/pricing/curves", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["product_id"] == "PROD_001"
        assert len(data["points"]) >= 40
        assert data["optimal_profit_price"] > 0
        assert "expected_demand" in data["points"][0]

    def test_multi_segment_curves_endpoint(self) -> None:
        """Verify multi-segment batch curves generation."""
        payload = {
            "product_id": "PROD_001",
            "customer_groups": ["budget", "regular", "premium"],
            "base_price": 100.0,
            "unit_cost": 60.0,
            "grid_points": 30,
        }
        res = client.post("/api/v1/pricing/curves/multi-segment", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "curves" in data
        assert "budget" in data["curves"]
        assert "regular" in data["curves"]

    def test_what_if_simulate_endpoint(self) -> None:
        """Verify what-if scenario simulation endpoint."""
        payload = {
            "product_id": "PROD_001",
            "customer_group": "regular",
            "base_price": 100.0,
            "base_demand": 50.0,
            "unit_cost": 60.0,
            "scenario": {
                "name": "Competitor Price War",
                "competitor_price_change_pct": -0.15,
                "cross_price_elasticity": 0.35,
            },
        }
        res = client.post("/api/v1/pricing/simulate", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["scenario_applied"] == "Competitor Price War"
        assert len(data["evaluations"]) >= 5
        assert data["best_compliant_strategy"] != ""

    def test_optimize_and_lifecycle_flow(self) -> None:
        """Verify end-to-end flow: optimize portfolio -> list recs -> review -> override -> approve -> publish -> rollback."""
        # 1. Run Optimization
        opt_payload = {
            "items": [
                {
                    "item_id": "PROD_001_budget_URBAN",
                    "product_id": "PROD_001",
                    "customer_group": "budget",
                    "location_id": "LOC_URBAN",
                    "base_price": 100.0,
                    "base_demand": 60.0,
                    "unit_cost": 60.0,
                },
                {
                    "item_id": "PROD_001_regular_URBAN",
                    "product_id": "PROD_001",
                    "customer_group": "regular",
                    "location_id": "LOC_URBAN",
                    "base_price": 100.0,
                    "base_demand": 50.0,
                    "unit_cost": 60.0,
                },
            ],
            "objective_type": "expected_profit",
            "max_price_change_pct": 0.15,
        }
        res_opt = client.post("/api/v1/pricing/optimize", json=opt_payload)
        assert res_opt.status_code == 200
        data_opt = res_opt.json()
        assert len(data_opt["recommendations"]) == 2
        assert data_opt["guardrail_compliant"] is True

        # 2. List recommendations from database
        res_list = client.get("/api/v1/recommendations?product_id=PROD_001")
        assert res_list.status_code == 200
        recs = res_list.json()
        assert len(recs) >= 2

        target_rec = recs[0]
        rec_id = target_rec["recommendation_id"]
        assert target_rec["state"] == "GENERATED"

        # 3. Transition to UNDER_REVIEW
        res_review = client.post(f"/api/v1/recommendations/{rec_id}/review", json={"actor": "analyst_alice", "notes": "Checking competitors"})
        assert res_review.status_code == 200
        assert res_review.json()["state"] == "UNDER_REVIEW"

        # 4. Apply Manager Price Override
        res_override = client.post(
            f"/api/v1/recommendations/{rec_id}/override",
            json={"override_price": 112.0, "reason": "Strategic pricing alignment with retail partner", "reviewed_by": "manager_bob"},
        )
        assert res_override.status_code == 200
        override_data = res_override.json()
        assert override_data["final_price"] == 112.0
        assert override_data["override_price"] == 112.0

        # 5. Approve recommendation
        res_approve = client.post(f"/api/v1/recommendations/{rec_id}/approve", json={"actor": "manager_bob", "notes": "Approved override"})
        assert res_approve.status_code == 200
        assert res_approve.json()["state"] == "APPROVED"

        # 6. Publish recommendation
        res_publish = client.post(f"/api/v1/recommendations/{rec_id}/publish", json={"actor": "system_deployer"})
        assert res_publish.status_code == 200
        assert res_publish.json()["state"] == "PUBLISHED"

        # 7. Emergency Rollback
        res_rollback = client.post(f"/api/v1/recommendations/{rec_id}/rollback", json={"actor": "ops_oncall", "notes": "Rolling back for maintenance"})
        assert res_rollback.status_code == 200
        rollback_data = res_rollback.json()
        assert rollback_data["state"] == "ROLLED_BACK"
        assert rollback_data["final_price"] == rollback_data["base_price"] == 100.0
