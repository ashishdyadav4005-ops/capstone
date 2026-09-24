"""Unit tests for What-If Pricing Scenario Simulator and Multi-Strategy Comparison."""

from __future__ import annotations

import pytest

from src.models.causal_dml import DoubleMachineLearningEstimator, ElasticityEstimate
from src.pricing.curves import DemandCurveGenerator
from src.pricing.simulator import (
    PricingScenario,
    PricingScenarioSimulator,
    StrategyComparisonResult,
)


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


class TestPricingScenarioSimulator:
    """Test suite for PricingScenarioSimulator."""

    def test_simulate_scenario_competitor_price_drop(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify that competitor price drop reduces baseline demand via cross-price elasticity."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)
        simulator = PricingScenarioSimulator(curve_generator=generator)

        scenario = PricingScenario(
            name="Competitor Price War",
            competitor_price_change_pct=-0.20, # -20% competitor drop
            cross_price_elasticity=0.50, # Demand drops by 0.5 * 20% = 10%
        )

        curve, shock = simulator.simulate_scenario(
            product_id="PROD_001",
            customer_group="regular",
            base_price=100.0,
            base_demand=50.0,
            unit_cost=60.0,
            scenario=scenario,
        )

        assert shock["scenario_name"] == "Competitor Price War"
        assert shock["original_base_demand"] == 50.0
        assert shock["adjusted_base_demand"] < 50.0
        assert curve.base_demand == shock["adjusted_base_demand"]

    def test_simulate_scenario_cost_inflation(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify that unit cost inflation raises unit cost and adjusts margin floor."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)
        simulator = PricingScenarioSimulator(curve_generator=generator)

        scenario = PricingScenario(
            name="Cost Inflation",
            unit_cost_change_pct=0.25, # +25% cost surge
        )

        curve, shock = simulator.simulate_scenario(
            product_id="PROD_001",
            customer_group="regular",
            base_price=100.0,
            base_demand=50.0,
            unit_cost=60.0,
            scenario=scenario,
        )

        assert shock["original_unit_cost"] == 60.0
        assert shock["adjusted_unit_cost"] == 75.0 # 60 * 1.25
        assert curve.unit_cost == 75.0

    def test_simulate_scenario_demand_surge(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify that seasonal holiday demand surge scales expected demand appropriately."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)
        simulator = PricingScenarioSimulator(curve_generator=generator)

        scenario = PricingScenario(
            name="Holiday Surge",
            demand_surge_multiplier=1.60, # +60% demand
        )

        curve, shock = simulator.simulate_scenario(
            product_id="PROD_001",
            customer_group="regular",
            base_price=100.0,
            base_demand=50.0,
            unit_cost=60.0,
            scenario=scenario,
        )

        assert shock["adjusted_base_demand"] == 80.0 # 50 * 1.6
        assert curve.base_demand == 80.0

    def test_compare_strategies_metrics_and_guardrails(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify side-by-side strategy evaluation and guardrail compliance."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)
        simulator = PricingScenarioSimulator(curve_generator=generator)

        res = simulator.compare_strategies(
            product_id="PROD_001",
            customer_group="regular",
            base_price=100.0,
            base_demand=50.0,
            unit_cost=60.0,
            capacity=100.0,
            margin_floor_pct=0.15, # Floor = 69.0
            max_volatility_pct=0.15, # Bounds = [85, 115]
        )

        assert isinstance(res, StrategyComparisonResult)
        assert len(res.evaluations) >= 5

        strat_names = [e.strategy_name for e in res.evaluations]
        assert "Baseline (Current)" in strat_names
        assert "Cost-Plus Markup (30%)" in strat_names
        assert "Causal Unconstrained (Revenue Max)" in strat_names
        assert "Causal Unconstrained (Profit Max)" in strat_names
        assert "Causal Guardrailed (Feasible)" in strat_names

        # Verify Causal Guardrailed is within volatility bounds [85, 115]
        guardrailed_ev = next(e for e in res.evaluations if e.strategy_name == "Causal Guardrailed (Feasible)")
        assert 85.0 <= guardrailed_ev.recommended_price <= 115.0
        assert guardrailed_ev.recommended_price >= 69.0
        assert guardrailed_ev.is_guardrail_compliant

        # Check summary table DataFrame
        df_summary = res.summary_table()
        assert len(df_summary) == len(res.evaluations)
        assert "revenue_lift_pct" in df_summary.columns
        assert "profit_lift_pct" in df_summary.columns

    def test_customer_group_impact_fairness(self, mock_causal_model: DoubleMachineLearningEstimator) -> None:
        """Verify fairness analysis across customer segments and disparity threshold checking."""
        generator = DemandCurveGenerator(causal_model=mock_causal_model)
        simulator = PricingScenarioSimulator(curve_generator=generator)

        # Scenario 1: Loose disparity threshold (25%) -> Compliant
        impact_loose = simulator.evaluate_customer_group_impact(
            product_id="PROD_001",
            customer_groups=["budget", "regular", "premium"],
            base_price=100.0,
            unit_cost=60.0,
            max_disparity_pct=0.25,
        )

        assert "budget" in impact_loose.group_prices
        assert "regular" in impact_loose.group_prices
        assert "premium" in impact_loose.group_prices
        assert impact_loose.max_price_disparity_pct >= 0.0

        # Scenario 2: Strict disparity threshold (1%) -> Non-compliant flag
        impact_strict = simulator.evaluate_customer_group_impact(
            product_id="PROD_001",
            customer_groups=["budget", "regular", "premium"],
            base_price=100.0,
            unit_cost=60.0,
            max_disparity_pct=0.01,
        )

        assert not impact_strict.is_fairness_compliant
        df_metrics = impact_strict.to_dataframe()
        assert len(df_metrics) == 3
        assert "consumer_surplus_delta_proxy" in df_metrics.columns
