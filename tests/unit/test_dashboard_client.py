"""Unit tests for Streamlit Dashboard ApiClient and UI Component Helpers."""

from __future__ import annotations

import pytest

from dashboard.api_client import ApiClient


@pytest.fixture
def api_client():
    """Create a test ApiClient instance."""
    return ApiClient(base_url="http://127.0.0.1:8000")


class TestDashboardApiClient:
    """Test suite for Streamlit Dashboard ApiClient."""

    def test_client_initialization_and_catalog(self, api_client: ApiClient):
        """Verify API client initializes and loads product catalog."""
        catalog = api_client.get_catalog()
        assert len(catalog) == 5
        assert catalog[0]["product_id"] == "PROD_001"
        assert catalog[0]["base_price"] == 100.0

    def test_login_flow(self, api_client: ApiClient):
        """Verify login authentication returns valid user profile and token."""
        # 1. Invalid credentials
        res_fail = api_client.login("unknown_user", "wrong_pwd")
        assert res_fail["success"] is False

        # 2. Valid credentials
        res_ok = api_client.login("analyst", "Analyst@12345")
        assert res_ok["success"] is True
        data = res_ok["data"]
        assert data["username"] == "analyst"
        assert data["role"] == "analyst"
        assert "access_token" in data

    def test_generate_curve(self, api_client: ApiClient):
        """Verify counterfactual demand curve generation for dashboard visualizer."""
        curve = api_client.generate_curve(
            product_id="PROD_001",
            customer_group="regular",
            base_price=100.0,
            base_demand=50.0,
            unit_cost=60.0,
            grid_points=20,
        )
        assert "points" in curve
        assert len(curve["points"]) >= 20
        assert curve["optimal_profit_price"] > 0
        assert curve["profit_lift_pct"] >= 0

    def test_simulate_scenario(self, api_client: ApiClient):
        """Verify what-if scenario simulation and multi-strategy benchmarks."""
        res = api_client.simulate_scenario(
            scenario_type="competitor_price_drop",
            product_id="PROD_001",
            competitor_price_change_pct=-0.20,
        )
        assert "strategies_evaluated" in res
        assert len(res["strategies_evaluated"]) >= 4

    def test_portfolio_optimization_and_lifecycle_actions(self, api_client: ApiClient):
        """Verify portfolio optimization, recommendations creation, approvals, overrides, and rollbacks."""
        items = [
            {
                "item_id": "DASH_TEST_01",
                "product_id": "PROD_001",
                "customer_group": "business",
                "location_id": "us-east-1",
                "base_price": 100.0,
                "base_demand": 50.0,
                "unit_cost": 60.0,
                "capacity": 100.0,
                "elasticity": -1.5,
                "elasticity_std_error": 0.1,
            }
        ]

        # 1. Optimize portfolio
        opt_res = api_client.optimize_portfolio(items=items, objective_type="profit")
        assert opt_res["total_profit_lift_pct"] > 0

        # 2. List recommendations
        recs = api_client.list_recommendations(product_id="PROD_001", customer_group="business")
        assert len(recs) > 0
        rec_id = recs[0]["recommendation_id"]

        # 3. Manager override price
        overridden = api_client.override_price(
            rec_id=rec_id,
            override_price=108.0,
            reason="Dashboard test override",
            actor="manager",
        )
        assert overridden["final_price"] == 108.0
        assert overridden["state"] == "UNDER_REVIEW"

        # 4. Approve recommendation
        approved = api_client.approve_recommendation(rec_id=rec_id, actor="manager")
        assert approved["state"] == "APPROVED"

        # 5. Publish recommendation
        published = api_client.publish_recommendation(rec_id=rec_id, actor="system")
        assert published["state"] == "PUBLISHED"

        # 6. Emergency rollback
        rolled_back = api_client.rollback_recommendation(rec_id=rec_id, actor="manager")
        assert rolled_back["state"] == "ROLLED_BACK"
        assert rolled_back["final_price"] == 100.0

    def test_audit_and_governance_features(self, api_client: ApiClient):
        """Verify cryptographic audit trail verification, fairness report, and model card."""
        # Audit logs & verification
        logs = api_client.get_audit_logs(limit=10)
        assert isinstance(logs, list)

        v_res = api_client.verify_audit_chain()
        assert v_res["is_valid"] is True

        # Fairness report
        fair_res = api_client.get_fairness_report()
        assert "overall_compliant" in fair_res
        assert len(fair_res["products"]) == 5

        # Model card
        card = api_client.get_model_card()
        assert "Double Machine Learning" in card["model_name"]
