"""CLI Driver for Constrained Dynamic Pricing Optimization under Guardrails."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure root workspace directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib

from src.common.logger import get_logger
from src.pricing.optimizer import (
    ConstrainedPriceOptimizer,
    OptimizationConfig,
    SegmentPricingItem,
)

logger = get_logger("run_optimizer")


def run_optimizer() -> None:
    """Load model, assemble catalog items, solve constrained optimization, and save artifacts."""
    logger.info("Starting Constrained Dynamic Pricing Optimization Pipeline...")

    model_path = ROOT_DIR / "data" / "models" / "causal_dml_model.pkl"
    output_dir = ROOT_DIR / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    causal_model = None
    if model_path.exists():
        logger.info("Loading pre-trained Causal DML model...", extra={"path": str(model_path)})
        causal_model = joblib.load(model_path)

    # 1. Assemble Catalog Items from Data or Defaults
    products = ["PROD_001", "PROD_002", "PROD_003", "PROD_004", "PROD_005"]
    customer_groups = ["budget", "regular", "premium", "business"]
    locations = ["LOC_URBAN", "LOC_SUBURBAN", "LOC_RURAL", "LOC_ONLINE"]

    base_costs = {
        "PROD_001": 60.0,
        "PROD_002": 30.0,
        "PROD_003": 120.0,
        "PROD_004": 15.0,
        "PROD_005": 85.0,
    }
    base_prices = {
        "PROD_001": 100.0,
        "PROD_002": 50.0,
        "PROD_003": 200.0,
        "PROD_004": 25.0,
        "PROD_005": 140.0,
    }
    base_demands = {
        "budget": 65.0,
        "regular": 50.0,
        "premium": 35.0,
        "business": 25.0,
    }

    items: list[SegmentPricingItem] = []

    for prod in products:
        for loc in locations:
            for grp in customer_groups:
                item_id = f"{prod}_{grp}_{loc}"
                p0 = base_prices[prod]
                c0 = base_costs[prod]
                q0 = base_demands[grp]

                # Extract elasticity from DML model if available
                if causal_model is not None:
                    est = causal_model.get_elasticity(prod, grp)
                    elasticity = est.point_estimate
                    se = est.std_error
                else:
                    elasticity = -1.50
                    se = 0.10

                items.append(
                    SegmentPricingItem(
                        item_id=item_id,
                        product_id=prod,
                        customer_group=grp,
                        location_id=loc,
                        base_price=p0,
                        base_demand=q0,
                        unit_cost=c0,
                        elasticity=elasticity,
                        elasticity_std_error=se,
                        capacity=100.0,
                    )
                )

    logger.info(f"Assembled {len(items)} product-segment items for joint optimization.")

    # 2. Run Constrained Optimization (Expected Profit Objective)
    optimizer = ConstrainedPriceOptimizer()
    cfg = OptimizationConfig(
        objective_type="expected_profit",
        risk_aversion=0.05,
        max_price_change_pct=0.15,
        min_margin_pct_over_cost=0.15,
        max_customer_group_disparity_pct=0.12,
        max_location_disparity_pct=0.10,
    )

    opt_result = optimizer.optimize(items, config_override=cfg)

    # 3. Print Results Summary
    print("\n" + "=" * 80)
    print("CONSTRAINED DYNAMIC PRICING OPTIMIZATION REPORT (SciPy SLSQP + Guardrails)")
    print("=" * 80)
    print(f"Solver Status:        {opt_result.status} ({opt_result.solver_message})")
    print(f"Solve Time:           {opt_result.solve_time_ms:.2f} ms")
    print(f"Total Baseline Rev:   ${opt_result.total_baseline_revenue:,.2f}")
    print(f"Total Optimized Rev:  ${opt_result.total_recommended_revenue:,.2f} ({opt_result.total_revenue_lift_pct:+.2f}%)")
    print(f"Total Baseline Prof:  ${opt_result.total_baseline_profit:,.2f}")
    print(f"Total Optimized Prof: ${opt_result.total_recommended_profit:,.2f} ({opt_result.total_profit_lift_pct:+.2f}%)")

    if opt_result.guardrail_check:
        print(f"\nGuardrail Compliance: {'PASS [100% COMPLIANT]' if opt_result.guardrail_check.is_compliant else 'FLAGGED'}")
        print(f"Total Violations:     {opt_result.guardrail_check.total_violations}")
        if opt_result.guardrail_check.violations:
            for v in opt_result.guardrail_check.violations[:5]:
                print(f"  - [{v.severity}] {v.rule_name} on {v.item_identifier}: {v.violation_message}")

    # Display sample recommendations table (first 12 items)
    df_summary = opt_result.summary_table()
    print("\nSample Recommendations (first 12 of 80 segment-locations):")
    print(df_summary.head(12).to_string(index=False))

    # 4. Save Artifacts
    opt_json_path = output_dir / "optimized_prices.json"
    recs_data = [r.to_dict() for r in opt_result.recommendations]
    with open(opt_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "status": opt_result.status,
            "solve_time_ms": opt_result.solve_time_ms,
            "total_recommended_revenue": opt_result.total_recommended_revenue,
            "total_recommended_profit": opt_result.total_recommended_profit,
            "total_revenue_lift_pct": opt_result.total_revenue_lift_pct,
            "total_profit_lift_pct": opt_result.total_profit_lift_pct,
            "recommendations": recs_data,
        }, f, indent=2)

    report_csv_path = output_dir / "optimization_report.csv"
    df_summary.to_csv(report_csv_path, index=False)

    logger.info("Optimization completed successfully.", extra={
        "json_artifact": str(opt_json_path),
        "csv_artifact": str(report_csv_path),
    })


if __name__ == "__main__":
    run_optimizer()
