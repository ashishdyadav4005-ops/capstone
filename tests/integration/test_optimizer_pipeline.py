"""Integration tests for the full constrained pricing optimization pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from src.data.generator import SyntheticDataGenerator
from src.models.causal_dml import DoubleMachineLearningEstimator
from src.pricing.optimizer import (
    ConstrainedPriceOptimizer,
    OptimizationConfig,
    SegmentPricingItem,
)


@pytest.fixture(scope="module")
def trained_dml_model() -> DoubleMachineLearningEstimator:
    """Load pre-trained model or quickly train a lightweight DML model on synthetic data."""
    root_model_path = Path("data/models/causal_dml_model.pkl")
    if root_model_path.exists():
        return joblib.load(root_model_path)

    gen = SyntheticDataGenerator(n_days=60, random_seed=42)
    df = gen.generate_dataset()
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


class TestOptimizerPipelineIntegration:
    """End-to-end integration tests for dynamic pricing optimization."""

    def test_full_catalog_joint_optimization(
        self,
        trained_dml_model: DoubleMachineLearningEstimator,
    ) -> None:
        """Verify joint optimization across products, customer groups, and locations."""
        products = ["PROD_001", "PROD_002", "PROD_003"]
        groups = ["budget", "regular", "premium", "business"]
        locations = ["LOC_URBAN", "LOC_SUBURBAN"]

        base_prices = {"PROD_001": 100.0, "PROD_002": 50.0, "PROD_003": 200.0}
        base_costs = {"PROD_001": 60.0, "PROD_002": 30.0, "PROD_003": 120.0}
        base_demands = {"budget": 60.0, "regular": 50.0, "premium": 35.0, "business": 25.0}

        items: list[SegmentPricingItem] = []
        for prod in products:
            for loc in locations:
                for grp in groups:
                    est = trained_dml_model.get_elasticity(prod, grp)
                    items.append(
                        SegmentPricingItem(
                            item_id=f"{prod}_{grp}_{loc}",
                            product_id=prod,
                            customer_group=grp,
                            location_id=loc,
                            base_price=base_prices[prod],
                            base_demand=base_demands[grp],
                            unit_cost=base_costs[prod],
                            elasticity=est.point_estimate,
                            elasticity_std_error=est.std_error,
                            capacity=120.0,
                        )
                    )

        assert len(items) == 24 # 3 products * 2 locations * 4 groups

        optimizer = ConstrainedPriceOptimizer()
        cfg = OptimizationConfig(
            objective_type="expected_profit",
            max_price_change_pct=0.15,
            min_margin_pct_over_cost=0.15,
            max_customer_group_disparity_pct=0.12,
            max_location_disparity_pct=0.10,
        )

        res = optimizer.optimize(items, config_override=cfg)

        assert res.status in ["OPTIMAL", "FEASIBLE_FALLBACK"]
        assert len(res.recommendations) == 24
        assert res.total_recommended_profit > 0
        assert res.solve_time_ms > 0

        # Guardrails check
        assert res.guardrail_check is not None
        assert res.guardrail_check.critical_violations == 0

        # Export table
        df_summary = res.summary_table()
        assert len(df_summary) == 24
        assert "recommended_price" in df_summary.columns
        assert "margin_pct" in df_summary.columns

        # Verify JSON export
        recs_data = [r.to_dict() for r in res.recommendations]
        json_str = json.dumps(recs_data)
        assert len(json_str) > 500
