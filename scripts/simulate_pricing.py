"""CLI Driver for Counterfactual Demand Curve Generation and What-If Pricing Simulations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure root workspace directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd

from src.common.logger import get_logger
from src.pricing.curves import DemandCurveGenerator
from src.pricing.simulator import (
    PricingScenario,
    PricingScenarioSimulator,
)

logger = get_logger("simulate_pricing")


def run_simulation() -> None:
    """Load model, generate counterfactual curves, run what-if scenarios, and save artifacts."""
    logger.info("Starting Counterfactual Demand Curves & Simulation Pipeline...")

    model_path = ROOT_DIR / "data" / "models" / "causal_dml_model.pkl"
    output_dir = ROOT_DIR / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    causal_model = None
    if model_path.exists():
        logger.info("Loading pre-trained Causal DML model...", extra={"path": str(model_path)})
        causal_model = joblib.load(model_path)
    else:
        logger.warning("No pre-trained model found at data/models/causal_dml_model.pkl, using fallback estimator.")

    # 1. Initialize Generator and Simulator
    generator = DemandCurveGenerator(causal_model=causal_model, default_grid_points=50)
    simulator = PricingScenarioSimulator(curve_generator=generator)

    # 2. Generate Multi-Segment Counterfactual Demand Curves for PROD_001
    sample_product = "PROD_001"
    base_price = 100.0
    unit_cost = 60.0
    capacity = 120.0

    print("\n" + "=" * 80)
    print(f"1. COUNTERFACTUAL DEMAND CURVES: Product {sample_product} (P0=${base_price}, Cost=${unit_cost})")
    print("=" * 80)

    multi_curves = generator.generate_multi_segment_curves(
        product_id=sample_product,
        customer_groups=["budget", "regular", "premium", "business"],
        location_id="LOC_URBAN",
        base_price=base_price,
        unit_cost=unit_cost,
        capacity=capacity,
        grid_points=50,
    )

    summary_df = multi_curves.summary_table()
    print(summary_df.to_string(index=False))

    # 3. Simulate What-If Market Scenarios
    scenarios = [
        PricingScenario(
            name="Baseline",
            description="Normal market conditions",
        ),
        PricingScenario(
            name="Competitor Price War (-15%)",
            description="Competitor slashes price by 15%, causing demand substitution",
            competitor_price_change_pct=-0.15,
            cross_price_elasticity=0.35,
        ),
        PricingScenario(
            name="Supply Chain Cost Inflation (+20%)",
            description="Raw material cost jumps 20%, lifting margin floor",
            unit_cost_change_pct=0.20,
        ),
        PricingScenario(
            name="Holiday Peak Surge (+50% Demand)",
            description="Seasonal holiday demand spike",
            demand_surge_multiplier=1.50,
        ),
        PricingScenario(
            name="High Sensitivity Spike",
            description="Consumers become more price-sensitive due to macro inflation",
            elasticity_shock=-0.35,
        ),
    ]

    print("\n" + "=" * 80)
    print("2. WHAT-IF SCENARIO STRATEGY COMPARISONS")
    print("=" * 80)

    all_comparisons = []
    for sc in scenarios:
        comp_res = simulator.compare_strategies(
            product_id=sample_product,
            customer_group="regular",
            location_id="LOC_URBAN",
            base_price=base_price,
            base_demand=50.0,
            unit_cost=unit_cost,
            capacity=capacity,
            scenario=sc,
        )
        print(f"\n>>> Scenario: {sc.name.upper()} ({sc.description})")
        comp_df = comp_res.summary_table()
        comp_df["scenario"] = sc.name
        all_comparisons.append(comp_df)
        print(comp_df.to_string(index=False))
        print(f"   * Best Revenue Strategy: {comp_res.best_revenue_strategy}")
        print(f"   * Best Profit Strategy:  {comp_res.best_profit_strategy}")
        print(f"   * Best Compliant Target: {comp_res.best_compliant_strategy}")

    # 4. Fairness and Customer Group Impact Analysis
    print("\n" + "=" * 80)
    print("3. CUSTOMER GROUP FAIRNESS & DISPARITY ANALYSIS")
    print("=" * 80)

    impact = simulator.evaluate_customer_group_impact(
        product_id=sample_product,
        customer_groups=["budget", "regular", "premium", "business"],
        location_id="LOC_URBAN",
        base_price=base_price,
        unit_cost=unit_cost,
        max_disparity_pct=0.12,
    )
    print(impact.to_dataframe().to_string(index=False))
    print(f"\nMax Price Disparity: {impact.max_price_disparity_pct:.2f}% (Threshold: {impact.disparity_threshold_pct:.2f}%)")
    print(f"Fairness Compliant: {'PASS [YES]' if impact.is_fairness_compliant else 'FAIL [DISPARITY EXCEEDS 12%]'}")

    # 5. Persist Simulation Artifacts
    sim_output_json = output_dir / "simulation_curves.json"
    curves_dict = {
        grp: curve.to_dict() for grp, curve in multi_curves.curves.items()
    }
    with open(sim_output_json, "w", encoding="utf-8") as f:
        json.dump(curves_dict, f, indent=2)

    scenario_csv = output_dir / "scenario_comparisons.csv"
    pd.concat(all_comparisons, ignore_index=True).to_csv(scenario_csv, index=False)

    logger.info("Simulation completed successfully. Artifacts saved.", extra={
        "curves_json": str(sim_output_json),
        "scenario_csv": str(scenario_csv),
    })


if __name__ == "__main__":
    run_simulation()
