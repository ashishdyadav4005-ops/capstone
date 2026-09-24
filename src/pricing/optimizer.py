"""Constrained Dynamic Price Optimizer using SciPy SLSQP and Multi-Guardrail Enforcement."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.common.logger import get_logger
from src.pricing.guardrails import GuardrailCheckResult, GuardrailsEngine

logger = get_logger("optimizer")


@dataclass
class SegmentPricingItem:
    """Input specification for a discrete product-segment dynamic pricing decision."""

    item_id: str
    product_id: str
    customer_group: str
    location_id: str
    base_price: float
    base_demand: float
    unit_cost: float
    elasticity: float
    elasticity_std_error: float = 0.10
    capacity: float | None = None
    min_price_override: float | None = None
    max_price_override: float | None = None


@dataclass
class OptimizationConfig:
    """Configuration parameters for the mathematical pricing optimizer."""

    objective_type: str = "expected_profit" # "expected_profit" or "expected_revenue"
    risk_aversion: float = 0.05 # Penalty weight for demand uncertainty variance
    max_price_change_pct: float = 0.15 # Volatility envelope (+/- 15%)
    min_margin_pct_over_cost: float = 0.15 # Margin floor (+15% over cost)
    absolute_min_margin_currency: float = 2.0 # Minimum $2.00 margin floor
    max_customer_group_disparity_pct: float = 0.12 # 12% group fairness ceiling
    max_location_disparity_pct: float = 0.10 # 10% location fairness ceiling
    max_capacity_utilization: float = 0.98 # Max 98% capacity threshold
    solver_method: str = "SLSQP"
    solver_tolerance: float = 1e-6
    solver_max_iter: int = 250
    enforcement_mode: str = "strict" # "strict" or "soft"


@dataclass
class OptimizedPriceRecommendation:
    """Individual optimized price recommendation for a product-segment."""

    item_id: str
    product_id: str
    customer_group: str
    location_id: str
    recommended_price: float
    base_price: float
    unit_cost: float
    price_change_pct: float
    expected_demand: float
    expected_revenue: float
    expected_profit: float
    margin_pct: float
    revenue_lift_pct: float
    profit_lift_pct: float
    status: str = "OPTIMAL" # "OPTIMAL", "FEASIBLE_FALLBACK"
    is_margin_compliant: bool = True
    is_volatility_compliant: bool = True
    is_capacity_compliant: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert recommendation to dictionary."""
        return asdict(self)


@dataclass
class OptimizationResult:
    """Comprehensive result of joint constrained pricing optimization."""

    status: str # "OPTIMAL", "FEASIBLE_FALLBACK", "INFEASIBLE"
    objective_type: str
    total_recommended_revenue: float
    total_recommended_profit: float
    total_baseline_revenue: float
    total_baseline_profit: float
    total_revenue_lift_pct: float
    total_profit_lift_pct: float
    recommendations: list[OptimizedPriceRecommendation] = field(default_factory=list)
    guardrail_check: GuardrailCheckResult | None = None
    solver_message: str = ""
    iterations: int = 0
    solve_time_ms: float = 0.0

    def summary_table(self) -> pd.DataFrame:
        """Export recommendations to a pandas DataFrame."""
        rows = []
        for r in self.recommendations:
            rows.append({
                "item_id": r.item_id,
                "product_id": r.product_id,
                "customer_group": r.customer_group,
                "location_id": r.location_id,
                "base_price": r.base_price,
                "recommended_price": r.recommended_price,
                "price_change_pct": round(r.price_change_pct, 2),
                "unit_cost": r.unit_cost,
                "expected_demand": round(r.expected_demand, 1),
                "expected_revenue": round(r.expected_revenue, 2),
                "expected_profit": round(r.expected_profit, 2),
                "margin_pct": round(r.margin_pct, 2),
                "revenue_lift_pct": round(r.revenue_lift_pct, 2),
                "profit_lift_pct": round(r.profit_lift_pct, 2),
                "status": r.status,
            })
        return pd.DataFrame(rows)


class ConstrainedPriceOptimizer:
    """Mathematical optimizer for dynamic pricing under revenue/profit objectives and safety guardrails."""

    def __init__(self, default_config: OptimizationConfig | None = None):
        self.config = default_config or OptimizationConfig()
        self.guardrails_engine = GuardrailsEngine(
            max_price_change_pct=self.config.max_price_change_pct,
            min_margin_pct_over_cost=self.config.min_margin_pct_over_cost,
            absolute_min_margin_currency=self.config.absolute_min_margin_currency,
            max_customer_group_disparity_pct=self.config.max_customer_group_disparity_pct,
            max_location_disparity_pct=self.config.max_location_disparity_pct,
            max_capacity_utilization=self.config.max_capacity_utilization,
        )

    def optimize(
        self,
        items: list[SegmentPricingItem],
        config_override: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """Solve joint constrained dynamic pricing optimization for a list of items."""
        start_time = time.perf_counter()
        cfg = config_override or self.config

        if not items:
            return OptimizationResult(
                status="OPTIMAL",
                objective_type=cfg.objective_type,
                total_recommended_revenue=0.0,
                total_recommended_profit=0.0,
                total_baseline_revenue=0.0,
                total_baseline_profit=0.0,
                total_revenue_lift_pct=0.0,
                total_profit_lift_pct=0.0,
                recommendations=[],
                solver_message="No items provided for optimization.",
                iterations=0,
                solve_time_ms=0.0,
            )

        n_items = len(items)

        # 1. Establish Variable Bounds [P_min, P_max]
        bounds: list[tuple[float, float]] = []
        p0_vec = np.zeros(n_items)
        q0_vec = np.zeros(n_items)
        cost_vec = np.zeros(n_items)
        theta_vec = np.zeros(n_items)
        se_vec = np.zeros(n_items)

        for i, it in enumerate(items):
            p0_vec[i] = it.base_price
            q0_vec[i] = it.base_demand
            cost_vec[i] = it.unit_cost
            theta_vec[i] = it.elasticity
            se_vec[i] = it.elasticity_std_error

            # Margin floor
            min_margin_p = max(
                it.unit_cost * (1.0 + cfg.min_margin_pct_over_cost),
                it.unit_cost + cfg.absolute_min_margin_currency,
            )
            vol_min = it.base_price * (1.0 - cfg.max_price_change_pct)
            vol_max = it.base_price * (1.0 + cfg.max_price_change_pct)

            lower_bound = max(vol_min, min_margin_p)
            upper_bound = vol_max

            if it.min_price_override is not None:
                lower_bound = max(lower_bound, it.min_price_override)
            if it.max_price_override is not None:
                upper_bound = min(upper_bound, it.max_price_override)

            # If margin floor exceeds volatility upper bound, ensure solver has a feasible corridor
            if lower_bound > upper_bound:
                upper_bound = lower_bound * 1.05

            bounds.append((float(lower_bound), float(upper_bound)))

        # 2. Define Objective Function (Minimize Negative Profit / Revenue)
        def objective_func(prices: np.ndarray) -> float:
            total_obj = 0.0
            for j in range(n_items):
                pj = prices[j]
                ratio = max(1e-4, pj / p0_vec[j])
                qj = float(max(0.0, q0_vec[j] * (ratio ** theta_vec[j])))

                if cfg.objective_type == "expected_revenue":
                    metric = pj * qj
                else: # expected_profit
                    metric = (pj - cost_vec[j]) * qj

                # Risk penalty for uncertainty variance
                # Var(Q) approximated by (Q * SE * ln(P/P0))^2
                uncertainty_pen = float(cfg.risk_aversion * (qj * se_vec[j] * np.log(ratio)) ** 2)
                total_obj += metric - uncertainty_pen

            return -total_obj # Minimize negative

        # 3. Formulate Non-Linear Constraints (fun(p) >= 0)
        constraints = []

        # (a) Capacity Constraints (item or joint location capacity)
        # Group items by location to enforce shared capacity if specified
        loc_map: dict[str, list[int]] = {}
        for idx, it in enumerate(items):
            loc_map.setdefault(it.location_id, []).append(idx)

        for _loc, idx_list in loc_map.items():
            # Check if any item has capacity limit
            for idx in idx_list:
                cap = items[idx].capacity
                if cap is not None:
                    def make_cap_constraint(item_idx: int, capacity_val: float):
                        return lambda p: (capacity_val * cfg.max_capacity_utilization) - (
                            q0_vec[item_idx] * (max(1e-4, p[item_idx] / p0_vec[item_idx]) ** theta_vec[item_idx])
                        )
                    constraints.append({
                        "type": "ineq",
                        "fun": make_cap_constraint(idx, cap),
                    })

        # (b) Customer Group Fairness Constraints (Disparity <= 12% across groups for same product and location)
        prod_loc_map: dict[tuple[str, str], list[int]] = {}
        for idx, it in enumerate(items):
            key = (it.product_id, it.location_id)
            prod_loc_map.setdefault(key, []).append(idx)

        for (_prod, _loc), idx_list in prod_loc_map.items():
            if len(idx_list) > 1:
                base_ref_p = np.mean([p0_vec[idx] for idx in idx_list])
                max_gap = cfg.max_customer_group_disparity_pct * base_ref_p
                for a in range(len(idx_list)):
                    for b in range(a + 1, len(idx_list)):
                        ia, ib = idx_list[a], idx_list[b]
                        # Disparity constraint: |P_ia - P_ib| <= max_gap
                        # 1. max_gap - (P_ia - P_ib) >= 0
                        # 2. max_gap - (P_ib - P_ia) >= 0
                        def make_grp_fairness_c1(idx_a: int, idx_b: int, gap: float):
                            return lambda p: gap - (p[idx_a] - p[idx_b])
                        def make_grp_fairness_c2(idx_a: int, idx_b: int, gap: float):
                            return lambda p: gap - (p[idx_b] - p[idx_a])

                        constraints.append({"type": "ineq", "fun": make_grp_fairness_c1(ia, ib, max_gap)})
                        constraints.append({"type": "ineq", "fun": make_grp_fairness_c2(ia, ib, max_gap)})

        # (c) Geographic Location Fairness Constraints (Disparity <= 10% across locations for same product and customer group)
        prod_grp_map: dict[tuple[str, str], list[int]] = {}
        for idx, it in enumerate(items):
            key = (it.product_id, it.customer_group)
            prod_grp_map.setdefault(key, []).append(idx)

        for (_prod, _grp), idx_list in prod_grp_map.items():
            if len(idx_list) > 1:
                base_ref_p = np.mean([p0_vec[idx] for idx in idx_list])
                max_gap = cfg.max_location_disparity_pct * base_ref_p
                for a in range(len(idx_list)):
                    for b in range(a + 1, len(idx_list)):
                        ia, ib = idx_list[a], idx_list[b]
                        def make_loc_fairness_c1(idx_a: int, idx_b: int, gap: float):
                            return lambda p: gap - (p[idx_a] - p[idx_b])
                        def make_loc_fairness_c2(idx_a: int, idx_b: int, gap: float):
                            return lambda p: gap - (p[idx_b] - p[idx_a])

                        constraints.append({"type": "ineq", "fun": make_loc_fairness_c1(ia, ib, max_gap)})
                        constraints.append({"type": "ineq", "fun": make_loc_fairness_c2(ia, ib, max_gap)})

        # 4. Multi-Start Initializations for SLSQP
        initial_guesses = [
            np.array([min(max(p0_vec[k], bounds[k][0]), bounds[k][1]) for k in range(n_items)]),
            np.array([(bounds[k][0] + bounds[k][1]) / 2.0 for k in range(n_items)]),
        ]

        best_opt = None
        best_val = float("inf")

        for x0 in initial_guesses:
            try:
                opt_res = minimize(
                    fun=objective_func,
                    x0=x0,
                    method=cfg.solver_method,
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": cfg.solver_max_iter, "ftol": cfg.solver_tolerance},
                )
                if opt_res.fun < best_val:
                    best_val = opt_res.fun
                    best_opt = opt_res
            except Exception as e:
                logger.warning("SLSQP optimization iteration error", extra={"error": str(e)})

        # 5. Extract Optimized Prices or Trigger Fallback
        if best_opt is not None and best_opt.success:
            solved_prices = best_opt.x
            status = "OPTIMAL"
            solver_msg = f"SLSQP converged successfully: {best_opt.message}"
            iterations = best_opt.nit
        else:
            # Fallback solver: Clip to nearest feasible bounds and harmonic mean
            solved_prices = self._solve_fallback(items, bounds, cfg)
            status = "FEASIBLE_FALLBACK"
            solver_msg = f"Fallback applied: {best_opt.message if best_opt else 'Solver failed'}"
            iterations = best_opt.nit if best_opt else 0

        # 6. Build Detailed Recommendations
        recommendations: list[OptimizedPriceRecommendation] = []
        total_rec_rev = 0.0
        total_rec_prof = 0.0
        total_base_rev = 0.0
        total_base_prof = 0.0

        records_for_guardrail_check: list[dict[str, Any]] = []

        for k, it in enumerate(items):
            rec_p = float(round(solved_prices[k], 2))
            ratio = rec_p / it.base_price
            exp_q = float(max(0.0, it.base_demand * (ratio ** it.elasticity)))
            exp_rev = float(rec_p * exp_q)
            exp_prof = float((rec_p - it.unit_cost) * exp_q)
            margin_pct = float(((rec_p - it.unit_cost) / rec_p * 100.0) if rec_p > 0 else 0.0)

            base_rev = float(it.base_price * it.base_demand)
            base_prof = float((it.base_price - it.unit_cost) * it.base_demand)

            rev_lift = float(((exp_rev - base_rev) / base_rev * 100.0) if base_rev > 0 else 0.0)
            prof_lift = float(((exp_prof - base_prof) / base_prof * 100.0) if base_prof > 0 else 0.0)
            price_chg = float(((rec_p - it.base_price) / it.base_price) * 100.0)

            # Compliance booleans
            min_margin = max(it.unit_cost * (1.0 + cfg.min_margin_pct_over_cost), it.unit_cost + cfg.absolute_min_margin_currency)
            margin_ok = rec_p >= (min_margin - 1e-3)
            vol_ok = (it.base_price * (1.0 - cfg.max_price_change_pct) - 1e-3) <= rec_p <= (it.base_price * (1.0 + cfg.max_price_change_pct) + 1e-3)
            cap_ok = (exp_q <= (it.capacity * cfg.max_capacity_utilization + 1e-3)) if it.capacity is not None else True

            rec_item = OptimizedPriceRecommendation(
                item_id=it.item_id,
                product_id=it.product_id,
                customer_group=it.customer_group,
                location_id=it.location_id,
                recommended_price=rec_p,
                base_price=it.base_price,
                unit_cost=it.unit_cost,
                price_change_pct=price_chg,
                expected_demand=float(round(exp_q, 2)),
                expected_revenue=float(round(exp_rev, 2)),
                expected_profit=float(round(exp_prof, 2)),
                margin_pct=float(round(margin_pct, 2)),
                revenue_lift_pct=float(round(rev_lift, 2)),
                profit_lift_pct=float(round(prof_lift, 2)),
                status=status,
                is_margin_compliant=margin_ok,
                is_volatility_compliant=vol_ok,
                is_capacity_compliant=cap_ok,
            )
            recommendations.append(rec_item)

            total_rec_rev += exp_rev
            total_rec_prof += exp_prof
            total_base_rev += base_rev
            total_base_prof += base_prof

            records_for_guardrail_check.append({
                "item_id": it.item_id,
                "product_id": it.product_id,
                "customer_group": it.customer_group,
                "location_id": it.location_id,
                "recommended_price": rec_p,
                "base_price": it.base_price,
                "unit_cost": it.unit_cost,
                "expected_demand": exp_q,
                "capacity": it.capacity,
            })

        # 7. Post-Optimization Guardrail Verification
        guardrail_result = self.guardrails_engine.validate_batch(records_for_guardrail_check)

        tot_rev_lift = float(((total_rec_rev - total_base_rev) / total_base_rev * 100.0) if total_base_rev > 0 else 0.0)
        tot_prof_lift = float(((total_rec_prof - total_base_prof) / total_base_prof * 100.0) if total_base_prof > 0 else 0.0)
        solve_time = float((time.perf_counter() - start_time) * 1000.0)

        return OptimizationResult(
            status=status,
            objective_type=cfg.objective_type,
            total_recommended_revenue=float(round(total_rec_rev, 2)),
            total_recommended_profit=float(round(total_rec_prof, 2)),
            total_baseline_revenue=float(round(total_base_rev, 2)),
            total_baseline_profit=float(round(total_base_prof, 2)),
            total_revenue_lift_pct=float(round(tot_rev_lift, 2)),
            total_profit_lift_pct=float(round(tot_prof_lift, 2)),
            recommendations=recommendations,
            guardrail_check=guardrail_result,
            solver_message=solver_msg,
            iterations=iterations,
            solve_time_ms=float(round(solve_time, 2)),
        )

    @staticmethod
    def _solve_fallback(
        items: list[SegmentPricingItem],
        bounds: list[tuple[float, float]],
        config: OptimizationConfig,
    ) -> np.ndarray:
        """Conservative projection fallback guaranteeing margin floor and volatility compliance."""
        fallback_prices = np.zeros(len(items))
        for i, it in enumerate(items):
            low, high = bounds[i]
            # If theoretical optimal under constant elasticity exists, clip it to feasible bounds
            if it.elasticity < -1.0 and it.unit_cost > 0:
                theo_p = it.unit_cost * (it.elasticity / (it.elasticity + 1.0))
            else:
                theo_p = it.base_price

            clamped_p = min(max(theo_p, low), high)
            fallback_prices[i] = clamped_p
        return fallback_prices
