"""Unit tests for API Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    OptimizationItemInput,
    OptimizationRequest,
    OverrideRequest,
    RecommendationRecord,
    RecommendationState,
    ScenarioShockSchema,
    SingleCurveRequest,
)


class TestApiSchemas:
    """Test suite for API request and response models."""

    def test_single_curve_request_defaults_and_validation(self) -> None:
        """Verify default values and positive price constraints."""
        req = SingleCurveRequest()
        assert req.product_id == "PROD_001"
        assert req.base_price == 100.0
        assert req.grid_points == 50

        # Invalid non-positive price
        with pytest.raises(ValidationError):
            SingleCurveRequest(base_price=-10.0)

        with pytest.raises(ValidationError):
            SingleCurveRequest(grid_points=2) # Minimum is 5

    def test_scenario_shock_schema_bounds(self) -> None:
        """Verify scenario shock bounds validation."""
        sc = ScenarioShockSchema(competitor_price_change_pct=-0.15, demand_surge_multiplier=1.5)
        assert sc.competitor_price_change_pct == -0.15
        assert sc.demand_surge_multiplier == 1.5

        # Extreme shock bounds
        with pytest.raises(ValidationError):
            ScenarioShockSchema(competitor_price_change_pct=-0.95) # Beyond -0.80

    def test_override_request_validation(self) -> None:
        """Verify manager override request schema enforces reason length."""
        req = OverrideRequest(override_price=110.0, reason="Strategic competitive adjustment")
        assert req.override_price == 110.0

        # Reason too short
        with pytest.raises(ValidationError):
            OverrideRequest(override_price=110.0, reason="no")

    def test_optimization_request_schema(self) -> None:
        """Verify optimization request validation."""
        item = OptimizationItemInput(
            item_id="PROD_001_budget",
            product_id="PROD_001",
            customer_group="budget",
            location_id="LOC_URBAN",
            base_price=100.0,
            base_demand=50.0,
            unit_cost=60.0,
        )
        opt_req = OptimizationRequest(items=[item], objective_type="expected_profit")
        assert len(opt_req.items) == 1
        assert opt_req.objective_type == "expected_profit"

    def test_recommendation_record_states(self) -> None:
        """Verify RecommendationRecord schema supports all lifecycle states."""
        rec = RecommendationRecord(
            recommendation_id="REC_123",
            item_id="PROD_001_budget",
            product_id="PROD_001",
            customer_group="budget",
            location_id="LOC_URBAN",
            base_price=100.0,
            recommended_price=115.0,
            final_price=115.0,
            unit_cost=60.0,
            expected_demand=48.0,
            expected_revenue=5520.0,
            expected_profit=2640.0,
            margin_pct=47.83,
            state=RecommendationState.GENERATED,
            created_at="2026-09-22T10:00:00Z",
            updated_at="2026-09-22T10:00:00Z",
        )
        assert rec.state == RecommendationState.GENERATED
        assert rec.final_price == 115.0
