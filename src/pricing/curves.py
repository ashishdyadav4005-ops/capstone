"""Counterfactual Demand Curves and Revenue/Profit Trajectory Generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.causal_dml import DoubleMachineLearningEstimator, ElasticityEstimate


@dataclass
class DemandPoint:
    """Evaluated price point along a counterfactual demand curve."""

    price: float
    expected_demand: float
    demand_ci_lower: float
    demand_ci_upper: float
    expected_revenue: float
    revenue_ci_lower: float
    revenue_ci_upper: float
    expected_profit: float
    profit_ci_lower: float
    profit_ci_upper: float
    margin_pct: float
    price_change_pct: float
    demand_change_pct: float
    point_elasticity: float
    is_revenue_optimal: bool = False
    is_profit_optimal: bool = False
    is_baseline: bool = False
    within_margin_guardrail: bool = True
    within_volatility_guardrail: bool = True
    within_capacity_guardrail: bool = True


@dataclass
class DemandCurveResult:
    """Full counterfactual demand curve output for a specific product-segment context."""

    product_id: str
    customer_group: str
    location_id: str
    base_price: float
    base_demand: float
    unit_cost: float
    elasticity: float
    elasticity_std_error: float
    elasticity_ci_95: tuple[float, float]
    is_cold_start: bool
    points: list[DemandPoint]
    optimal_revenue_price: float
    max_expected_revenue: float
    optimal_profit_price: float
    max_expected_profit: float
    baseline_revenue: float
    baseline_profit: float
    revenue_lift_pct: float
    profit_lift_pct: float
    theoretical_optimal_profit_price: float | None
    guardrails_applied: dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert curve points to a pandas DataFrame."""
        records = [asdict(p) for p in self.points]
        return pd.DataFrame(records)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a dictionary for API/JSON responses."""
        res = asdict(self)
        return res


@dataclass
class MultiSegmentCurveResult:
    """Grouped demand curves across multiple customer segments or locations."""

    product_id: str
    curves: dict[str, DemandCurveResult]

    def summary_table(self) -> pd.DataFrame:
        """Generate a comparison summary table across all evaluated segments."""
        rows = []
        for segment_key, curve in self.curves.items():
            rows.append({
                "segment": segment_key,
                "customer_group": curve.customer_group,
                "location_id": curve.location_id,
                "elasticity": curve.elasticity,
                "elasticity_ci_95": f"[{curve.elasticity_ci_95[0]:.2f}, {curve.elasticity_ci_95[1]:.2f}]",
                "base_price": curve.base_price,
                "base_demand": round(curve.base_demand, 1),
                "baseline_revenue": round(curve.baseline_revenue, 2),
                "baseline_profit": round(curve.baseline_profit, 2),
                "optimal_revenue_price": curve.optimal_revenue_price,
                "max_expected_revenue": round(curve.max_expected_revenue, 2),
                "revenue_lift_pct": round(curve.revenue_lift_pct, 2),
                "optimal_profit_price": curve.optimal_profit_price,
                "max_expected_profit": round(curve.max_expected_profit, 2),
                "profit_lift_pct": round(curve.profit_lift_pct, 2),
                "is_cold_start": curve.is_cold_start,
            })
        return pd.DataFrame(rows)


class DemandCurveGenerator:
    """Generates counterfactual demand, revenue, and profit curves using causal elasticity estimates."""

    def __init__(
        self,
        causal_model: DoubleMachineLearningEstimator | None = None,
        default_grid_points: int = 50,
        default_price_range: tuple[float, float] = (-0.25, 0.25),
        cold_start_se_multiplier: float = 2.0,
    ):
        self.causal_model = causal_model
        self.default_grid_points = default_grid_points
        self.default_price_range = default_price_range
        self.cold_start_se_multiplier = cold_start_se_multiplier

    def generate_curve(
        self,
        product_id: str,
        customer_group: str = "regular",
        location_id: str = "LOC_URBAN",
        base_price: float = 100.0,
        base_demand: float = 50.0,
        unit_cost: float = 60.0,
        capacity: float | None = 100.0,
        min_price: float | None = None,
        max_price: float | None = None,
        grid_points: int | None = None,
        custom_elasticity: float | None = None,
        custom_std_error: float | None = None,
        margin_floor_pct: float = 0.15,
        max_volatility_pct: float = 0.15,
    ) -> DemandCurveResult:
        """Generate a high-resolution counterfactual demand curve across candidate prices.

        Parameters
        ----------
        product_id : str
            Identifier for the product.
        customer_group : str
            Target customer group (e.g. 'budget', 'regular', 'premium', 'business').
        location_id : str
            Store or geographic location identifier.
        base_price : float
            Current observed baseline price $P_0 > 0$.
        base_demand : float
            Baseline expected demand $Q_0 > 0$.
        unit_cost : float
            Unit cost of production / acquisition.
        capacity : float | None
            Maximum inventory or processing capacity ceiling.
        min_price : float | None
            Lower bound of price grid. Defaults to base_price * (1 + default_price_range[0]).
        max_price : float | None
            Upper bound of price grid. Defaults to base_price * (1 + default_price_range[1]).
        grid_points : int | None
            Number of discrete candidate price points to evaluate.
        custom_elasticity : float | None
            Override elasticity point estimate.
        custom_std_error : float | None
            Override elasticity standard error.
        margin_floor_pct : float
            Minimum profit margin ratio over cost (e.g. 0.15 means P >= Cost * 1.15).
        max_volatility_pct : float
            Maximum allowable percentage price deviation from baseline (e.g. 0.15 means +/- 15%).
        """
        if base_price <= 0:
            raise ValueError(f"Base price must be strictly positive, got {base_price}")
        if base_demand <= 0:
            raise ValueError(f"Base demand must be strictly positive, got {base_demand}")
        if unit_cost < 0:
            raise ValueError(f"Unit cost cannot be negative, got {unit_cost}")

        points_count = grid_points or self.default_grid_points
        if points_count < 2:
            raise ValueError(f"Grid points must be >= 2, got {points_count}")

        # 1. Resolve Elasticity and Uncertainty Bounds
        is_cold_start = False
        if custom_elasticity is not None:
            elasticity = float(custom_elasticity)
            std_error = float(custom_std_error if custom_std_error is not None else 0.15)
            ci_lower = elasticity - 1.96 * std_error
            ci_upper = elasticity + 1.96 * std_error
        elif self.causal_model is not None:
            est: ElasticityEstimate = self.causal_model.get_elasticity(product_id, customer_group)
            elasticity = est.point_estimate
            std_error = est.std_error
            ci_lower = est.ci_lower_95
            ci_upper = est.ci_upper_95
            if est.sample_size == 0:
                is_cold_start = True
                std_error *= self.cold_start_se_multiplier
                ci_lower = elasticity - 1.96 * std_error
                ci_upper = elasticity + 1.96 * std_error
        else:
            # Fallback default elasticity
            is_cold_start = True
            elasticity = -1.50
            std_error = 0.25 * self.cold_start_se_multiplier
            ci_lower = elasticity - 1.96 * std_error
            ci_upper = elasticity + 1.96 * std_error

        # 2. Determine Candidate Price Grid
        low_bound = min_price if min_price is not None else base_price * (1.0 + self.default_price_range[0])
        high_bound = max_price if max_price is not None else base_price * (1.0 + self.default_price_range[1])

        if low_bound <= 0:
            low_bound = max(0.01, base_price * 0.10)
        if high_bound <= low_bound:
            high_bound = low_bound * 1.50

        price_grid = np.linspace(low_bound, high_bound, points_count)

        # Ensure base_price is explicitly included in the grid for accurate baseline comparison
        if not np.any(np.isclose(price_grid, base_price, atol=1e-4)):
            price_grid = np.sort(np.append(price_grid, base_price))

        # Guardrail thresholds
        min_margin_price = unit_cost * (1.0 + margin_floor_pct)
        volatility_min_price = base_price * (1.0 - max_volatility_pct)
        volatility_max_price = base_price * (1.0 + max_volatility_pct)

        # 3. Evaluate Counterfactual Points
        points: list[DemandPoint] = []
        best_rev_idx = -1
        max_rev = -float("inf")
        best_prof_idx = -1
        max_prof = -float("inf")

        for idx, price in enumerate(price_grid):
            p = float(round(price, 2))
            price_ratio = p / base_price

            # Demand expectation: Q(P) = Q_0 * (P / P_0)^elasticity
            exp_q = float(base_demand * (price_ratio ** elasticity))

            # Demand bounds: Q_lower, Q_upper
            q_b1 = float(base_demand * (price_ratio ** ci_lower))
            q_b2 = float(base_demand * (price_ratio ** ci_upper))
            q_low = float(max(0.0, min(q_b1, q_b2)))
            q_high = float(max(0.0, max(q_b1, q_b2)))
            exp_q = float(max(0.0, exp_q))

            # Revenue: Rev(P) = P * Q(P)
            exp_rev = float(p * exp_q)
            rev_low = float(p * q_low)
            rev_high = float(p * q_high)

            # Profit: Profit(P) = (P - UnitCost) * Q(P)
            unit_margin = p - unit_cost
            exp_prof = float(unit_margin * exp_q)
            prof_b1 = float(unit_margin * q_low)
            prof_b2 = float(unit_margin * q_high)
            prof_low = float(min(prof_b1, prof_b2))
            prof_high = float(max(prof_b1, prof_b2))

            margin_pct = float(((p - unit_cost) / p * 100.0) if p > 0 else 0.0)
            price_chg_pct = float(((p - base_price) / base_price) * 100.0)
            demand_chg_pct = float(((exp_q - base_demand) / base_demand) * 100.0)

            # Check guardrail compliance flags
            within_margin = p >= (min_margin_price - 1e-4)
            within_volatility = (volatility_min_price - 1e-4) <= p <= (volatility_max_price + 1e-4)
            within_capacity = (exp_q <= capacity + 1e-4) if capacity is not None else True

            is_base = bool(np.isclose(p, base_price, atol=1e-3))

            pt = DemandPoint(
                price=p,
                expected_demand=float(round(exp_q, 3)),
                demand_ci_lower=float(round(q_low, 3)),
                demand_ci_upper=float(round(q_high, 3)),
                expected_revenue=float(round(exp_rev, 2)),
                revenue_ci_lower=float(round(rev_low, 2)),
                revenue_ci_upper=float(round(rev_high, 2)),
                expected_profit=float(round(exp_prof, 2)),
                profit_ci_lower=float(round(prof_low, 2)),
                profit_ci_upper=float(round(prof_high, 2)),
                margin_pct=float(round(margin_pct, 2)),
                price_change_pct=float(round(price_chg_pct, 2)),
                demand_change_pct=float(round(demand_chg_pct, 2)),
                point_elasticity=float(round(elasticity, 4)),
                is_baseline=is_base,
                within_margin_guardrail=within_margin,
                within_volatility_guardrail=within_volatility,
                within_capacity_guardrail=within_capacity,
            )
            points.append(pt)

            if exp_rev > max_rev:
                max_rev = exp_rev
                best_rev_idx = idx

            if exp_prof > max_prof:
                max_prof = exp_prof
                best_prof_idx = idx

        # Mark optimal points
        if 0 <= best_rev_idx < len(points):
            points[best_rev_idx].is_revenue_optimal = True
        if 0 <= best_prof_idx < len(points):
            points[best_prof_idx].is_profit_optimal = True

        opt_rev_price = points[best_rev_idx].price if best_rev_idx >= 0 else base_price
        opt_prof_price = points[best_prof_idx].price if best_prof_idx >= 0 else base_price

        # Baseline performance metrics
        baseline_revenue = float(base_price * base_demand)
        baseline_profit = float((base_price - unit_cost) * base_demand)

        rev_lift = float(((max_rev - baseline_revenue) / baseline_revenue * 100.0) if baseline_revenue > 0 else 0.0)
        prof_lift = float(((max_prof - baseline_profit) / baseline_profit * 100.0) if baseline_profit > 0 else 0.0)

        # Theoretical optimal price under constant elasticity: P* = UnitCost * (theta / (theta + 1)) if theta < -1
        theo_opt_profit_price = None
        if elasticity < -1.0 and unit_cost > 0:
            theo_opt_profit_price = float(round(unit_cost * (elasticity / (elasticity + 1.0)), 2))

        guardrails_applied = {
            "margin_floor_price": float(round(min_margin_price, 2)),
            "margin_floor_pct": margin_floor_pct,
            "volatility_min_price": float(round(volatility_min_price, 2)),
            "volatility_max_price": float(round(volatility_max_price, 2)),
            "max_volatility_pct": max_volatility_pct,
            "capacity_ceiling": capacity,
        }

        return DemandCurveResult(
            product_id=product_id,
            customer_group=customer_group,
            location_id=location_id,
            base_price=float(round(base_price, 2)),
            base_demand=float(round(base_demand, 2)),
            unit_cost=float(round(unit_cost, 2)),
            elasticity=float(round(elasticity, 4)),
            elasticity_std_error=float(round(std_error, 4)),
            elasticity_ci_95=(float(round(ci_lower, 4)), float(round(ci_upper, 4))),
            is_cold_start=is_cold_start,
            points=points,
            optimal_revenue_price=opt_rev_price,
            max_expected_revenue=float(round(max_rev, 2)),
            optimal_profit_price=opt_prof_price,
            max_expected_profit=float(round(max_prof, 2)),
            baseline_revenue=float(round(baseline_revenue, 2)),
            baseline_profit=float(round(baseline_profit, 2)),
            revenue_lift_pct=float(round(rev_lift, 2)),
            profit_lift_pct=float(round(prof_lift, 2)),
            theoretical_optimal_profit_price=theo_opt_profit_price,
            guardrails_applied=guardrails_applied,
        )

    def generate_multi_segment_curves(
        self,
        product_id: str,
        customer_groups: list[str] | None = None,
        location_id: str = "LOC_URBAN",
        base_price: float = 100.0,
        base_demand_by_group: dict[str, float] | None = None,
        unit_cost: float = 60.0,
        capacity: float | None = 100.0,
        grid_points: int = 50,
    ) -> MultiSegmentCurveResult:
        """Generate batch demand curves across multiple customer segments for a product."""
        groups = customer_groups or ["budget", "regular", "premium", "business"]
        base_demands = base_demand_by_group or {
            "budget": 60.0,
            "regular": 50.0,
            "premium": 35.0,
            "business": 25.0,
        }

        curves: dict[str, DemandCurveResult] = {}
        for grp in groups:
            dem = base_demands.get(grp, 50.0)
            curve = self.generate_curve(
                product_id=product_id,
                customer_group=grp,
                location_id=location_id,
                base_price=base_price,
                base_demand=dem,
                unit_cost=unit_cost,
                capacity=capacity,
                grid_points=grid_points,
            )
            curves[grp] = curve

        return MultiSegmentCurveResult(product_id=product_id, curves=curves)
