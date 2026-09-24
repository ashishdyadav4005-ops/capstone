"""Integration tests for the dynamic pricing simulation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.data.generator import SyntheticDataGenerator
from src.models.causal_dml import DoubleMachineLearningEstimator
from src.pricing.curves import DemandCurveGenerator
from src.pricing.simulator import PricingScenario, PricingScenarioSimulator


@pytest.fixture(scope="module")
def trained_dml_model(tmp_path_factory: pytest.TempPathFactory) -> DoubleMachineLearningEstimator:
    """Load pre-trained model or quickly train a lightweight DML model on synthetic data."""
    root_model_path = Path("data/models/causal_dml_model.pkl")
    if root_model_path.exists():
        return joblib.load(root_model_path)

    # Train a fast synthetic model for testing
    gen = SyntheticDataGenerator(n_days=60, random_seed=42)
    df = gen.generate_dataset()
    df["log_price"] = df["price"].apply(lambda p: float(pd.Series([p]).pipe(lambda s: pd.np.log(s)).iloc[0] if hasattr(pd, "np") else 0.0))
    import numpy as np
    df["log_price"] = np.log(df["price"])
    df["log_quantity"] = np.log(df["quantity"])
    df["rolling_demand_7d"] = df["quantity"]
    df["rolling_demand_14d"] = df["quantity"]
    df["rolling_price_mean_7d"] = df["price"]
    df["price_to_rolling_mean"] = 1.0

    model = DoubleMachineLearningEstimator(n_splits=3, y_max_iter=30, t_max_iter=30, random_state=42)
    confounders = ["cost", "competitor_price", "is_weekend", "is_holiday", "rolling_demand_7d"]
    model.fit(df, confounder_cols=confounders)
    return model


class TestPricingPipelineIntegration:
    """End-to-end integration tests for pricing curves and simulator."""

    def test_full_pricing_pipeline_flow(
        self,
        trained_dml_model: DoubleMachineLearningEstimator,
        tmp_path: Path,
    ) -> None:
        """Verify complete pipeline: DML model -> Demand Curves -> What-If Simulation -> Artifact Export."""
        generator = DemandCurveGenerator(causal_model=trained_dml_model, default_grid_points=40)
        simulator = PricingScenarioSimulator(curve_generator=generator)

        # 1. Multi-segment demand curve generation
        multi = generator.generate_multi_segment_curves(
            product_id="PROD_001",
            customer_groups=["budget", "regular", "premium", "business"],
            location_id="LOC_URBAN",
            base_price=100.0,
            unit_cost=55.0,
            capacity=150.0,
            grid_points=40,
        )

        assert len(multi.curves) == 4
        for grp, curve in multi.curves.items():
            assert curve.product_id == "PROD_001"
            assert curve.customer_group == grp
            assert curve.optimal_profit_price > 0
            assert len(curve.points) >= 40
            # Test JSON exportability
            d = curve.to_dict()
            json_str = json.dumps(d)
            assert len(json_str) > 100

        # 2. What-If Scenario Comparisons
        scenarios = [
            PricingScenario(name="Competitor War", competitor_price_change_pct=-0.15),
            PricingScenario(name="Supply Inflation", unit_cost_change_pct=0.20),
            PricingScenario(name="Demand Surge", demand_surge_multiplier=1.40),
        ]

        for sc in scenarios:
            comp_res = simulator.compare_strategies(
                product_id="PROD_001",
                customer_group="regular",
                base_price=100.0,
                base_demand=50.0,
                unit_cost=55.0,
                scenario=sc,
            )
            assert len(comp_res.evaluations) >= 5
            assert comp_res.best_compliant_strategy != ""

        # 3. Fairness Impact
        impact = simulator.evaluate_customer_group_impact(
            product_id="PROD_001",
            customer_groups=["budget", "regular", "premium", "business"],
            base_price=100.0,
            unit_cost=55.0,
            max_disparity_pct=0.15,
        )
        assert isinstance(impact.max_price_disparity_pct, float)
        assert len(impact.group_metrics) == 4
