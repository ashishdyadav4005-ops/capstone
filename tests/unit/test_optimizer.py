"""Unit tests for Constrained Dynamic Price Optimizer."""

from __future__ import annotations

import numpy as np

from src.pricing.optimizer import (
    ConstrainedPriceOptimizer,
    OptimizationConfig,
    OptimizationResult,
    SegmentPricingItem,
)


class TestConstrainedPriceOptimizer:
    """Test suite for ConstrainedPriceOptimizer."""

    def test_single_item_profit_maximization_under_volatility(self) -> None:
        """Verify that optimizer pushes price towards optimal profit while respecting the volatility ceiling."""
        optimizer = ConstrainedPriceOptimizer()
        item = SegmentPricingItem(
            item_id="PROD_001_regular",
            product_id="PROD_001",
            customer_group="regular",
            location_id="LOC_URBAN",
            base_price=100.0,
            base_demand=50.0,
            unit_cost=60.0,
            elasticity=-1.50, # Theoretical unconstrained optimum = 60 * (-1.5 / -0.5) = 180.0
            elasticity_std_error=0.08,
            capacity=100.0,
        )

        cfg = OptimizationConfig(
            objective_type="expected_profit",
            max_price_change_pct=0.15, # Max price $115.0
            min_margin_pct_over_cost=0.15,
        )

        res = optimizer.optimize([item], config_override=cfg)

        assert isinstance(res, OptimizationResult)
        assert res.status in ["OPTIMAL", "FEASIBLE_FALLBACK"]
        assert len(res.recommendations) == 1

        rec = res.recommendations[0]
        # Should be pushed to volatility ceiling $115.0
        assert np.isclose(rec.recommended_price, 115.0, atol=0.5)
        assert rec.profit_lift_pct > 0.0
        assert rec.is_margin_compliant
        assert rec.is_volatility_compliant
        assert rec.is_capacity_compliant

    def test_margin_floor_enforcement(self) -> None:
        """Verify that optimizer strictly enforces profit margin floor over unit cost."""
        optimizer = ConstrainedPriceOptimizer()
        # Cost = $80.0 -> Margin floor (15%) = $92.0
        # Base price = $85.0
        item = SegmentPricingItem(
            item_id="PROD_002_low_margin",
            product_id="PROD_002",
            customer_group="budget",
            location_id="LOC_URBAN",
            base_price=85.0,
            base_demand=60.0,
            unit_cost=80.0,
            elasticity=-2.50, # High elasticity pushes price down
            elasticity_std_error=0.10,
        )

        cfg = OptimizationConfig(
            min_margin_pct_over_cost=0.15,
            max_price_change_pct=0.20,
        )

        res = optimizer.optimize([item], config_override=cfg)
        rec = res.recommendations[0]

        # Recommended price must be >= 80 * 1.15 = 92.0
        assert rec.recommended_price >= 91.99
        assert rec.is_margin_compliant

    def test_capacity_constraint_throttles_demand(self) -> None:
        """Verify that tight capacity forces price higher to compress demand within capacity limits."""
        optimizer = ConstrainedPriceOptimizer()
        item = SegmentPricingItem(
            item_id="PROD_003_tight_cap",
            product_id="PROD_003",
            customer_group="regular",
            location_id="LOC_URBAN",
            base_price=100.0,
            base_demand=80.0,
            unit_cost=40.0,
            elasticity=-1.80,
            elasticity_std_error=0.08,
            capacity=45.0, # Capacity ceiling is 45 units (baseline is 80 units)
        )

        cfg = OptimizationConfig(
            max_price_change_pct=0.40, # Allow room to throttle demand
            max_capacity_utilization=0.98,
        )

        res = optimizer.optimize([item], config_override=cfg)
        rec = res.recommendations[0]

        # Price must increase to suppress demand <= 45 * 0.98 = 44.1
        assert rec.recommended_price > 100.0
        assert rec.expected_demand <= (45.0 * 0.98 + 1.0)
        assert rec.is_capacity_compliant

    def test_customer_group_fairness_coupling(self) -> None:
        """Verify that joint optimization couples customer group prices within 12% disparity."""
        optimizer = ConstrainedPriceOptimizer()
        items = [
            SegmentPricingItem(
                item_id="PROD_001_budget",
                product_id="PROD_001",
                customer_group="budget",
                location_id="LOC_URBAN",
                base_price=100.0,
                base_demand=60.0,
                unit_cost=50.0,
                elasticity=-2.80, # Highly elastic
                elasticity_std_error=0.08,
            ),
            SegmentPricingItem(
                item_id="PROD_001_premium",
                product_id="PROD_001",
                customer_group="premium",
                location_id="LOC_URBAN",
                base_price=100.0,
                base_demand=40.0,
                unit_cost=50.0,
                elasticity=-0.60, # Inelastic
                elasticity_std_error=0.08,
            ),
        ]

        cfg = OptimizationConfig(
            max_price_change_pct=0.20,
            max_customer_group_disparity_pct=0.12, # Strict 12% gap
        )

        res = optimizer.optimize(items, config_override=cfg)

        p_budget = next(r.recommended_price for r in res.recommendations if r.customer_group == "budget")
        p_premium = next(r.recommended_price for r in res.recommendations if r.customer_group == "premium")

        # Disparity = (p_premium - p_budget) / p_budget <= 12%
        disparity = abs(p_premium - p_budget) / min(p_budget, p_premium)
        assert disparity <= 0.125, f"Disparity {disparity * 100:.2f}% exceeded 12% ceiling"

    def test_expected_revenue_objective(self) -> None:
        """Verify optimization using expected revenue objective."""
        optimizer = ConstrainedPriceOptimizer()
        item = SegmentPricingItem(
            item_id="PROD_001_rev",
            product_id="PROD_001",
            customer_group="regular",
            location_id="LOC_URBAN",
            base_price=100.0,
            base_demand=50.0,
            unit_cost=60.0,
            elasticity=-0.80, # Inelastic -> higher price yields higher revenue
            elasticity_std_error=0.08,
        )

        cfg = OptimizationConfig(
            objective_type="expected_revenue",
            max_price_change_pct=0.15,
        )

        res = optimizer.optimize([item], config_override=cfg)
        assert res.objective_type == "expected_revenue"
        assert res.total_recommended_revenue > res.total_baseline_revenue

    def test_empty_items_returns_zero_result(self) -> None:
        """Verify that passing an empty list returns a clean, zeroed result without error."""
        optimizer = ConstrainedPriceOptimizer()
        res = optimizer.optimize([])

        assert res.status == "OPTIMAL"
        assert len(res.recommendations) == 0
        assert res.total_recommended_revenue == 0.0
