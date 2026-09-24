"""What-If Pricing Scenario Simulator and Multi-Strategy Comparison Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.models.causal_dml import DoubleMachineLearningEstimator
from src.pricing.curves import DemandCurveGenerator, DemandCurveResult


@dataclass
class PricingScenario:
    """Configurable what-if market shock scenario."""

    name: str = "custom"
    description: str = "Custom scenario"
    competitor_price_change_pct: float = 0.0 # e.g. -0.10 means competitor drops price 10%
    cross_price_elasticity: float = 0.35 # Demand shift elasticity w.r.t competitor price
    unit_cost_change_pct: float = 0.0 # e.g. +0.20 means 20% supply cost surge
    demand_surge_multiplier: float = 1.0 # e.g. 1.50 means 50% seasonal/holiday demand surge
    elasticity_shock: float = 0.0 # e.g. -0.25 means customers become more price-sensitive
    capacity_multiplier: float = 1.0 # e.g. 0.80 means 20% supply chain disruption


@dataclass
class StrategyEvaluation:
    """Performance and compliance metrics for a single pricing strategy."""

    strategy_name: str
    recommended_price: float
    expected_demand: float
    expected_revenue: float
    expected_profit: float
    margin_pct: float
    revenue_lift_pct: float
    profit_lift_pct: float
    price_change_pct: float
    volatility_violation: bool
    margin_violation: bool
    capacity_violation: bool
    is_guardrail_compliant: bool


@dataclass
class StrategyComparisonResult:
    """Comparison result of multiple pricing strategies under a specific scenario."""

    product_id: str
    customer_group: str
    location_id: str
    base_price: float
    base_demand: float
    unit_cost: float
    scenario_applied: str
    evaluations: list[StrategyEvaluation] = field(default_factory=list)
    best_revenue_strategy: str = ""
    best_profit_strategy: str = ""
    best_compliant_strategy: str = ""

    def summary_table(self) -> pd.DataFrame:
        """Convert strategy evaluations to a pandas DataFrame."""
        rows = []
        for ev in self.evaluations:
            rows.append({
                "strategy": ev.strategy_name,
                "price": ev.recommended_price,
                "price_change_pct": round(ev.price_change_pct, 2),
                "expected_demand": round(ev.expected_demand, 1),
                "expected_revenue": round(ev.expected_revenue, 2),
                "expected_profit": round(ev.expected_profit, 2),
                "margin_pct": round(ev.margin_pct, 2),
                "revenue_lift_pct": round(ev.revenue_lift_pct, 2),
                "profit_lift_pct": round(ev.profit_lift_pct, 2),
                "guardrail_compliant": ev.is_guardrail_compliant,
            })
        return pd.DataFrame(rows)


@dataclass
class CustomerGroupImpact:
    """Fairness and disparity evaluation across customer segments for a product."""

    product_id: str
    location_id: str
    group_prices: dict[str, float]
    max_price_disparity_pct: float
    disparity_threshold_pct: float
    is_fairness_compliant: bool
    group_metrics: list[dict[str, Any]]

    def to_dataframe(self) -> pd.DataFrame:
        """Return group metrics as a DataFrame."""
        return pd.DataFrame(self.group_metrics)


class PricingScenarioSimulator:
    """Simulates market shocks and evaluates dynamic pricing strategies under guardrails."""

    def __init__(
        self,
        curve_generator: DemandCurveGenerator | None = None,
        causal_model: DoubleMachineLearningEstimator | None = None,
    ):
        self.curve_generator = curve_generator or DemandCurveGenerator(causal_model=causal_model)

    def simulate_scenario(
        self,
        product_id: str,
        customer_group: str = "regular",
        location_id: str = "LOC_URBAN",
        base_price: float = 100.0,
        base_demand: float = 50.0,
        unit_cost: float = 60.0,
        capacity: float | None = 100.0,
        scenario: PricingScenario | None = None,
        margin_floor_pct: float = 0.15,
        max_volatility_pct: float = 0.15,
    ) -> tuple[DemandCurveResult, dict[str, Any]]:
        """Simulate demand and profit curves under a specific market scenario shock.

        Returns
        -------
        tuple[DemandCurveResult, dict[str, Any]]
            Shocked demand curve result and summary of adjusted shock parameters.
        """
        sc = scenario or PricingScenario(name="baseline", description="No shock applied")

        # 1. Adjust unit cost
        adj_unit_cost = float(max(0.01, unit_cost * (1.0 + sc.unit_cost_change_pct)))

        # 2. Adjust baseline demand from competitor price cross-elasticity and surge
        competitor_demand_shift = 1.0 + (sc.cross_price_elasticity * sc.competitor_price_change_pct)
        competitor_demand_shift = max(0.10, competitor_demand_shift)
        adj_base_demand = float(max(1.0, base_demand * sc.demand_surge_multiplier * competitor_demand_shift))

        # 3. Adjust capacity
        adj_capacity = float(capacity * sc.capacity_multiplier) if capacity is not None else None

        # 4. Resolve base elasticity and apply elasticity shock
        base_elasticity = -1.50
        base_se = 0.15
        if self.curve_generator.causal_model is not None:
            est = self.curve_generator.causal_model.get_elasticity(product_id, customer_group)
            base_elasticity = est.point_estimate
            base_se = est.std_error

        adj_elasticity = float(base_elasticity + sc.elasticity_shock)

        # 5. Generate shocked curve
        shocked_curve = self.curve_generator.generate_curve(
            product_id=product_id,
            customer_group=customer_group,
            location_id=location_id,
            base_price=base_price,
            base_demand=adj_base_demand,
            unit_cost=adj_unit_cost,
            capacity=adj_capacity,
            custom_elasticity=adj_elasticity,
            custom_std_error=base_se,
            margin_floor_pct=margin_floor_pct,
            max_volatility_pct=max_volatility_pct,
        )

        shock_summary = {
            "scenario_name": sc.name,
            "scenario_description": sc.description,
            "original_unit_cost": unit_cost,
            "adjusted_unit_cost": adj_unit_cost,
            "original_base_demand": base_demand,
            "adjusted_base_demand": adj_base_demand,
            "original_elasticity": base_elasticity,
            "adjusted_elasticity": adj_elasticity,
            "original_capacity": capacity,
            "adjusted_capacity": adj_capacity,
        }

        return shocked_curve, shock_summary

    def compare_strategies(
        self,
        product_id: str,
        customer_group: str = "regular",
        location_id: str = "LOC_URBAN",
        base_price: float = 100.0,
        base_demand: float = 50.0,
        unit_cost: float = 60.0,
        capacity: float | None = 100.0,
        scenario: PricingScenario | None = None,
        naive_model_price: float | None = None,
        cost_plus_markup: float = 0.30,
        margin_floor_pct: float = 0.15,
        max_volatility_pct: float = 0.15,
    ) -> StrategyComparisonResult:
        """Compare multiple dynamic pricing strategies side-by-side."""
        curve, shock_summary = self.simulate_scenario(
            product_id=product_id,
            customer_group=customer_group,
            location_id=location_id,
            base_price=base_price,
            base_demand=base_demand,
            unit_cost=unit_cost,
            capacity=capacity,
            scenario=scenario,
            margin_floor_pct=margin_floor_pct,
            max_volatility_pct=max_volatility_pct,
        )

        # Bounds for guardrails
        curr_cost = shock_summary["adjusted_unit_cost"]
        min_margin_price = curr_cost * (1.0 + margin_floor_pct)
        vol_min = base_price * (1.0 - max_volatility_pct)
        vol_max = base_price * (1.0 + max_volatility_pct)
        eff_capacity = shock_summary["adjusted_capacity"]
        elasticity = curve.elasticity

        # Define candidate strategies
        strategies: dict[str, float] = {}

        # 1. Baseline Current Price
        strategies["Baseline (Current)"] = base_price

        # 2. Cost-Plus Markup
        strategies["Cost-Plus Markup (30%)"] = float(round(curr_cost * (1.0 + cost_plus_markup), 2))

        # 3. Naive Observational Model
        if naive_model_price is not None:
            strategies["Naive Observational"] = float(round(naive_model_price, 2))
        else:
            # Naive model usually recommends a modest heuristic price
            strategies["Naive Observational"] = float(round(base_price * 1.04, 2))

        # 4. Causal Unconstrained Revenue Maxima
        strategies["Causal Unconstrained (Revenue Max)"] = curve.optimal_revenue_price

        # 5. Causal Unconstrained Profit Maxima
        strategies["Causal Unconstrained (Profit Max)"] = curve.optimal_profit_price

        # 6. Causal Guardrailed Feasible Target
        # Clip profit-optimal price into feasible volatility and margin bounds
        guardrailed_p = curve.optimal_profit_price
        guardrailed_p = max(min_margin_price, guardrailed_p) # satisfy margin floor
        guardrailed_p = max(vol_min, min(vol_max, guardrailed_p)) # satisfy volatility bound
        strategies["Causal Guardrailed (Feasible)"] = float(round(guardrailed_p, 2))

        # Evaluate performance of each strategy
        evaluations: list[StrategyEvaluation] = []
        best_rev_strat = ""
        max_rev = -float("inf")
        best_prof_strat = ""
        max_prof = -float("inf")
        best_comp_strat = ""
        max_comp_prof = -float("inf")

        base_rev = float(base_price * curve.base_demand)
        base_prof = float((base_price - curr_cost) * curve.base_demand)

        for name, price in strategies.items():
            p = float(round(price, 2))
            price_ratio = p / base_price
            exp_q = float(max(0.0, curve.base_demand * (price_ratio ** elasticity)))
            exp_rev = float(p * exp_q)
            exp_prof = float((p - curr_cost) * exp_q)
            margin_pct = float(((p - curr_cost) / p * 100.0) if p > 0 else 0.0)

            rev_lift = float(((exp_rev - base_rev) / base_rev * 100.0) if base_rev > 0 else 0.0)
            prof_lift = float(((exp_prof - base_prof) / base_prof * 100.0) if base_prof > 0 else 0.0)
            price_chg = float(((p - base_price) / base_price) * 100.0)

            # Guardrail compliance check
            vol_viol = bool(p < (vol_min - 1e-4) or p > (vol_max + 1e-4))
            margin_viol = bool(p < (min_margin_price - 1e-4))
            cap_viol = bool((exp_q > (eff_capacity + 1e-4)) if eff_capacity is not None else False)
            compliant = not (vol_viol or margin_viol or cap_viol)

            ev = StrategyEvaluation(
                strategy_name=name,
                recommended_price=p,
                expected_demand=float(round(exp_q, 2)),
                expected_revenue=float(round(exp_rev, 2)),
                expected_profit=float(round(exp_prof, 2)),
                margin_pct=float(round(margin_pct, 2)),
                revenue_lift_pct=float(round(rev_lift, 2)),
                profit_lift_pct=float(round(prof_lift, 2)),
                price_change_pct=float(round(price_chg, 2)),
                volatility_violation=vol_viol,
                margin_violation=margin_viol,
                capacity_violation=cap_viol,
                is_guardrail_compliant=compliant,
            )
            evaluations.append(ev)

            if exp_rev > max_rev:
                max_rev = exp_rev
                best_rev_strat = name

            if exp_prof > max_prof:
                max_prof = exp_prof
                best_prof_strat = name

            if compliant and exp_prof > max_comp_prof:
                max_comp_prof = exp_prof
                best_comp_strat = name

        return StrategyComparisonResult(
            product_id=product_id,
            customer_group=customer_group,
            location_id=location_id,
            base_price=base_price,
            base_demand=curve.base_demand,
            unit_cost=curr_cost,
            scenario_applied=scenario.name if scenario else "baseline",
            evaluations=evaluations,
            best_revenue_strategy=best_rev_strat,
            best_profit_strategy=best_prof_strat,
            best_compliant_strategy=best_comp_strat or "Baseline (Current)",
        )

    def evaluate_customer_group_impact(
        self,
        product_id: str,
        customer_groups: list[str] | None = None,
        location_id: str = "LOC_URBAN",
        base_price: float = 100.0,
        unit_cost: float = 60.0,
        max_disparity_pct: float = 0.12, # 12% max allowable gap across groups
    ) -> CustomerGroupImpact:
        """Evaluate fairness and price disparity across customer segments for a product."""
        groups = customer_groups or ["budget", "regular", "premium", "business"]
        multi_curves = self.curve_generator.generate_multi_segment_curves(
            product_id=product_id,
            customer_groups=groups,
            location_id=location_id,
            base_price=base_price,
            unit_cost=unit_cost,
        )

        group_prices: dict[str, float] = {}
        group_metrics: list[dict[str, Any]] = []

        for grp, curve in multi_curves.curves.items():
            opt_p = curve.optimal_profit_price
            group_prices[grp] = opt_p

            # Consumer surplus delta proxy: -(P_opt - P_0) * Q_opt
            p_diff = opt_p - base_price
            cs_proxy = float(-p_diff * curve.points[0].expected_demand)

            group_metrics.append({
                "customer_group": grp,
                "elasticity": curve.elasticity,
                "base_price": curve.base_price,
                "optimal_price": opt_p,
                "price_diff_pct": round(((opt_p - base_price) / base_price * 100.0), 2),
                "expected_demand": round(curve.points[0].expected_demand, 1),
                "expected_profit": round(curve.max_expected_profit, 2),
                "consumer_surplus_delta_proxy": round(cs_proxy, 2),
            })

        prices_list = list(group_prices.values())
        min_p = min(prices_list)
        max_p = max(prices_list)
        disparity_pct = float(((max_p - min_p) / min_p) if min_p > 0 else 0.0)

        is_compliant = disparity_pct <= (max_disparity_pct + 1e-4)

        return CustomerGroupImpact(
            product_id=product_id,
            location_id=location_id,
            group_prices=group_prices,
            max_price_disparity_pct=float(round(disparity_pct * 100.0, 2)),
            disparity_threshold_pct=float(round(max_disparity_pct * 100.0, 2)),
            is_fairness_compliant=is_compliant,
            group_metrics=group_metrics,
        )
