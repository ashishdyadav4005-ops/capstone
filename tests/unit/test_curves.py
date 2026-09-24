"""Unit tests for Counterfactual Demand Curve Generator."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.causal_dml import DoubleMachineLearningEstimator, ElasticityEstimate
from src.pricing.curves import DemandCurveGenerator, DemandCurveResult


@pytest.fixture
def mock_causal_model() -> DoubleMachineLearningEstimator:
    """Provide a mock DML estimator with known elasticity estimates."""
    model = DoubleMachineLearningEstimator()
    model.overall_elasticity = -1.50
    model.elasticities = {
        "PROD_001_budget": ElasticityEstimate(
            segment_key="PROD_001_budget",
            product_id="PROD_001",
            customer_group="budget",
            point_estimate=-2.20,
            std_error=0.10,
            ci_lower_95=-2.40,
            ci_upper_95=-2.00,
            t_statistic=-22.0,
            p_value=1e-10,
            sample_size=1000,
        ),
        "PROD_001_regular": ElasticityEstimate(
            segment_key="PROD_001_regular",
            product_id="PROD_001",
            customer_group="regular",
            point_estimate=-1.50,
            std_error=0.08,
            ci_lower_95=-1.66,
            ci_upper_95=-1.34,
            t_statistic=-18.75,
            p_value=1e-10,
            sample_size=1000,
        ),
        "PROD_001_premium": ElasticityEstimate(
            segment_key="PROD_001_premium",
            product_id="PROD_001",
            customer_group="premium",
            point_estimate=-0.80,
            std_error=0.06,
            ci_lower_95=-0.92,
            ci_upper_95=-0.68,
            t_statistic=-13.33,
            p_value=1e-10,
            sample_size=1000,
        ),
    }
    return model


class TestDemandCurveGenerator:
    """Test suite for DemandCurveGenerator."""

    def test_demand_curve_generation_basic(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify basic demand curve creation, points count, and baseline point inclusion."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model, default_grid_points=50)
        res = generator.generate_curve(
            product_id="PROD_001",
            customer_group="regular",
            location_id="LOC_URBAN",
            base_price=100.0,
            base_demand=50.0,
            unit_cost=60.0,
            capacity=100.0,
        )

        assert isinstance(res, DemandCurveResult)
        assert res.product_id == "PROD_001"
        assert res.customer_group == "regular"
        assert res.elasticity == -1.50
        assert not res.is_cold_start
        assert len(res.points) >= 50

        # Check baseline point is present
        baseline_pts = [p for p in res.points if p.is_baseline]
        assert len(baseline_pts) == 1
        base_pt = baseline_pts[0]
        assert np.isclose(base_pt.price, 100.0)
        assert np.isclose(base_pt.expected_demand, 50.0, atol=0.1)
        assert np.isclose(base_pt.expected_revenue, 5000.0, atol=10.0)
        assert np.isclose(base_pt.expected_profit, 2000.0, atol=10.0)

    def test_confidence_intervals_bounds(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify that demand, revenue, and profit confidence bounds enclose expected values."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)
        res = generator.generate_curve(
            product_id="PROD_001",
            customer_group="budget",
            base_price=100.0,
            base_demand=60.0,
            unit_cost=50.0,
        )

        for pt in res.points:
            # Demand bounds: Q_low <= Q_exp <= Q_high
            assert pt.demand_ci_lower <= pt.expected_demand + 1e-4
            assert pt.expected_demand <= pt.demand_ci_upper + 1e-4

            # Revenue bounds: Rev_low <= Rev_exp <= Rev_high
            assert pt.revenue_ci_lower <= pt.expected_revenue + 1e-4
            assert pt.expected_revenue <= pt.revenue_ci_upper + 1e-4

            # Profit bounds when price >= cost
            if pt.price >= res.unit_cost:
                assert pt.profit_ci_lower <= pt.expected_profit + 1e-4
                assert pt.expected_profit <= pt.profit_ci_upper + 1e-4

    def test_demand_monotonicity_with_negative_elasticity(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify that counterfactual demand strictly decreases as price increases when elasticity < 0."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)
        res = generator.generate_curve(
            product_id="PROD_001",
            customer_group="regular",
            base_price=100.0,
            base_demand=50.0,
        )

        prices = [p.price for p in res.points]
        demands = [p.expected_demand for p in res.points]

        # Check monotonic decrease
        for i in range(len(demands) - 1):
            assert demands[i] >= demands[i + 1], f"Demand at P={prices[i]} ({demands[i]}) < P={prices[i+1]} ({demands[i+1]})"

    def test_revenue_and_profit_optimal_points(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify that optimal revenue and profit points are identified correctly."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)
        res = generator.generate_curve(
            product_id="PROD_001",
            customer_group="regular", # elasticity = -1.5
            base_price=100.0,
            base_demand=50.0,
            unit_cost=60.0,
            min_price=50.0,
            max_price=250.0,
            grid_points=100,
        )

        rev_opt_pts = [p for p in res.points if p.is_revenue_optimal]
        prof_opt_pts = [p for p in res.points if p.is_profit_optimal]

        assert len(rev_opt_pts) == 1
        assert len(prof_opt_pts) == 1

        # Check theoretical profit optimal: P* = Cost * (theta / (theta + 1)) = 60 * (-1.5 / -0.5) = 180.0
        assert res.theoretical_optimal_profit_price == 180.0
        assert np.isclose(res.optimal_profit_price, 180.0, atol=3.0)
        assert res.max_expected_profit > res.baseline_profit

    def test_guardrail_boundary_flags(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify that margin floor, price volatility, and capacity guardrails set correct boolean flags."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)
        res = generator.generate_curve(
            product_id="PROD_001",
            customer_group="regular",
            base_price=100.0,
            base_demand=50.0,
            unit_cost=70.0, # Margin floor at 1.15 * 70 = 80.5
            capacity=60.0, # Demand exceeding 60 triggers capacity violation
            margin_floor_pct=0.15,
            max_volatility_pct=0.15, # Volatility window [85, 115]
            min_price=60.0,
            max_price=140.0,
            grid_points=80,
        )

        for pt in res.points:
            # Margin check
            if pt.price < 80.5 - 1e-3:
                assert not pt.within_margin_guardrail
            else:
                assert pt.within_margin_guardrail

            # Volatility check
            if pt.price < 85.0 - 1e-3 or pt.price > 115.0 + 1e-3:
                assert not pt.within_volatility_guardrail
            else:
                assert pt.within_volatility_guardrail

            # Capacity check
            if pt.expected_demand > 60.0 + 1e-3:
                assert not pt.within_capacity_guardrail
            else:
                assert pt.within_capacity_guardrail

    def test_cold_start_fallback(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify that unseen product-group segments trigger cold start with expanded confidence intervals."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model, cold_start_se_multiplier=2.5)
        res = generator.generate_curve(
            product_id="PROD_999", # Unseen product
            customer_group="unknown_group",
            base_price=100.0,
            base_demand=50.0,
        )

        assert res.is_cold_start
        assert res.elasticity == -1.50 # Global fallback
        # Standard error should be expanded by 2.5x
        assert res.elasticity_std_error >= 0.50

    def test_multi_segment_curves(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify multi-segment batch curve generation and summary table output."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)
        multi = generator.generate_multi_segment_curves(
            product_id="PROD_001",
            customer_groups=["budget", "regular", "premium"],
            base_price=100.0,
            unit_cost=60.0,
        )

        assert len(multi.curves) == 3
        assert "budget" in multi.curves
        assert "regular" in multi.curves
        assert "premium" in multi.curves

        df_summary = multi.summary_table()
        assert len(df_summary) == 3
        assert "optimal_profit_price" in df_summary.columns
        assert "revenue_lift_pct" in df_summary.columns

        # Budget segment (elasticity -2.2) should have lower optimal price than Premium segment (elasticity -0.8)
        budget_opt = multi.curves["budget"].optimal_profit_price
        regular_opt = multi.curves["regular"].optimal_profit_price
        assert budget_opt <= regular_opt

    def test_dataframe_and_dict_serialization(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify export to DataFrame and JSON-compatible dict."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)
        res = generator.generate_curve(
            product_id="PROD_001",
            customer_group="regular",
            base_price=100.0,
            base_demand=50.0,
        )

        df = res.to_dataframe()
        assert len(df) == len(res.points)
        assert "expected_demand" in df.columns
        assert "expected_profit" in df.columns

        d = res.to_dict()
        assert d["product_id"] == "PROD_001"
        assert len(d["points"]) == len(res.points)

    def test_invalid_inputs_raise_errors(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify that negative prices, zero demands, and negative costs raise ValueError."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)

        with pytest.raises(ValueError, match="Base price must be strictly positive"):
            generator.generate_curve(product_id="PROD_001", base_price=0.0)

        with pytest.raises(ValueError, match="Base demand must be strictly positive"):
            generator.generate_curve(product_id="PROD_001", base_demand=-5.0)

        with pytest.raises(ValueError, match="Unit cost cannot be negative"):
            generator.generate_curve(product_id="PROD_001", unit_cost=-10.0)

        with pytest.raises(ValueError, match="Grid points must be >= 2"):
            generator.generate_curve(product_id="PROD_001", grid_points=1)
