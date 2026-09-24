"""Pricing API Router: Counterfactual curves, what-if simulations, optimization, and catalog."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import (
    get_audit_logger,
    get_causal_model,
    get_curve_generator,
    get_price_optimizer,
    get_recommendation_repository,
    get_scenario_simulator,
)
from src.api.repository import RecommendationRepository
from src.api.schemas import (
    CatalogProduct,
    CatalogResponse,
    CurvePointSchema,
    DemandCurveResultSchema,
    GuardrailViolationSchema,
    MultiSegmentCurveRequest,
    MultiSegmentCurveResponse,
    OptimizationRequest,
    OptimizationResponse,
    PriceRecommendationSchema,
    RecommendationRecord,
    RecommendationState,
    SimulationRequest,
    SimulationResponse,
    SingleCurveRequest,
    StrategyEvaluationSchema,
)
from src.audit.logger import AuditLogger
from src.models.causal_dml import DoubleMachineLearningEstimator
from src.pricing.curves import DemandCurveGenerator
from src.pricing.optimizer import (
    ConstrainedPriceOptimizer,
    OptimizationConfig,
    SegmentPricingItem,
)
from src.pricing.simulator import PricingScenario, PricingScenarioSimulator

router = APIRouter(prefix="/api/v1/pricing", tags=["Pricing Engine"])


@router.get("/catalog", response_model=CatalogResponse)
def get_catalog() -> CatalogResponse:
    """Return active catalog metadata, products, baseline pricing, costs, and customer groups."""
    products = [
        CatalogProduct(product_id="PROD_001", product_name="Enterprise Cloud Storage (TB/Mo)", base_price=100.0, unit_cost=60.0, capacity=100.0),
        CatalogProduct(product_id="PROD_002", product_name="Compute Node Standard (vCPU-Hr)", base_price=50.0, unit_cost=30.0, capacity=150.0),
        CatalogProduct(product_id="PROD_003", product_name="Dedicated AI Inference Engine", base_price=200.0, unit_cost=120.0, capacity=60.0),
        CatalogProduct(product_id="PROD_004", product_name="API Gateway Routing Package", base_price=25.0, unit_cost=15.0, capacity=300.0),
        CatalogProduct(product_id="PROD_005", product_name="High-Security HSM Enclave", base_price=140.0, unit_cost=85.0, capacity=80.0),
    ]
    return CatalogResponse(products=products)


@router.post("/curves", response_model=DemandCurveResultSchema)
def generate_curve(
    req: SingleCurveRequest,
    generator: DemandCurveGenerator = Depends(get_curve_generator),
) -> DemandCurveResultSchema:
    """Generate a high-resolution counterfactual demand curve across candidate prices with confidence bounds."""
    try:
        curve = generator.generate_curve(
            product_id=req.product_id,
            customer_group=req.customer_group,
            location_id=req.location_id,
            base_price=req.base_price,
            base_demand=req.base_demand,
            unit_cost=req.unit_cost,
            capacity=req.capacity,
            grid_points=req.grid_points,
            margin_floor_pct=req.margin_floor_pct,
            max_volatility_pct=req.max_volatility_pct,
        )

        pts_schemas = [
            CurvePointSchema(
                price=p.price,
                expected_demand=p.expected_demand,
                demand_ci_lower=p.demand_ci_lower,
                demand_ci_upper=p.demand_ci_upper,
                expected_revenue=p.expected_revenue,
                revenue_ci_lower=p.revenue_ci_lower,
                revenue_ci_upper=p.revenue_ci_upper,
                expected_profit=p.expected_profit,
                profit_ci_lower=p.profit_ci_lower,
                profit_ci_upper=p.profit_ci_upper,
                margin_pct=p.margin_pct,
                price_change_pct=p.price_change_pct,
                demand_change_pct=p.demand_change_pct,
                point_elasticity=p.point_elasticity,
                is_revenue_optimal=p.is_revenue_optimal,
                is_profit_optimal=p.is_profit_optimal,
                is_baseline=p.is_baseline,
                within_margin_guardrail=p.within_margin_guardrail,
                within_volatility_guardrail=p.within_volatility_guardrail,
                within_capacity_guardrail=p.within_capacity_guardrail,
            )
            for p in curve.points
        ]

        return DemandCurveResultSchema(
            product_id=curve.product_id,
            customer_group=curve.customer_group,
            location_id=curve.location_id,
            base_price=curve.base_price,
            base_demand=curve.base_demand,
            unit_cost=curve.unit_cost,
            elasticity=curve.elasticity,
            elasticity_std_error=curve.elasticity_std_error,
            elasticity_ci_95=list(curve.elasticity_ci_95),
            is_cold_start=curve.is_cold_start,
            optimal_revenue_price=curve.optimal_revenue_price,
            max_expected_revenue=curve.max_expected_revenue,
            optimal_profit_price=curve.optimal_profit_price,
            max_expected_profit=curve.max_expected_profit,
            baseline_revenue=curve.baseline_revenue,
            baseline_profit=curve.baseline_profit,
            revenue_lift_pct=curve.revenue_lift_pct,
            profit_lift_pct=curve.profit_lift_pct,
            theoretical_optimal_profit_price=curve.theoretical_optimal_profit_price,
            points=pts_schemas,
            guardrails_applied=curve.guardrails_applied,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/curves/multi-segment", response_model=MultiSegmentCurveResponse)
def generate_multi_segment_curves(
    req: MultiSegmentCurveRequest,
    generator: DemandCurveGenerator = Depends(get_curve_generator),
) -> MultiSegmentCurveResponse:
    """Generate batch counterfactual demand curves across all customer segments."""
    try:
        multi = generator.generate_multi_segment_curves(
            product_id=req.product_id,
            customer_groups=req.customer_groups,
            location_id=req.location_id,
            base_price=req.base_price,
            unit_cost=req.unit_cost,
            capacity=req.capacity,
            grid_points=req.grid_points,
        )

        curves_dict = {}
        for grp, curve in multi.curves.items():
            pts_schemas = [
                CurvePointSchema(
                    price=p.price,
                    expected_demand=p.expected_demand,
                    demand_ci_lower=p.demand_ci_lower,
                    demand_ci_upper=p.demand_ci_upper,
                    expected_revenue=p.expected_revenue,
                    revenue_ci_lower=p.revenue_ci_lower,
                    revenue_ci_upper=p.revenue_ci_upper,
                    expected_profit=p.expected_profit,
                    profit_ci_lower=p.profit_ci_lower,
                    profit_ci_upper=p.profit_ci_upper,
                    margin_pct=p.margin_pct,
                    price_change_pct=p.price_change_pct,
                    demand_change_pct=p.demand_change_pct,
                    point_elasticity=p.point_elasticity,
                    is_revenue_optimal=p.is_revenue_optimal,
                    is_profit_optimal=p.is_profit_optimal,
                    is_baseline=p.is_baseline,
                    within_margin_guardrail=p.within_margin_guardrail,
                    within_volatility_guardrail=p.within_volatility_guardrail,
                    within_capacity_guardrail=p.within_capacity_guardrail,
                )
                for p in curve.points
            ]

            curves_dict[grp] = DemandCurveResultSchema(
                product_id=curve.product_id,
                customer_group=curve.customer_group,
                location_id=curve.location_id,
                base_price=curve.base_price,
                base_demand=curve.base_demand,
                unit_cost=curve.unit_cost,
                elasticity=curve.elasticity,
                elasticity_std_error=curve.elasticity_std_error,
                elasticity_ci_95=list(curve.elasticity_ci_95),
                is_cold_start=curve.is_cold_start,
                optimal_revenue_price=curve.optimal_revenue_price,
                max_expected_revenue=curve.max_expected_revenue,
                optimal_profit_price=curve.optimal_profit_price,
                max_expected_profit=curve.max_expected_profit,
                baseline_revenue=curve.baseline_revenue,
                baseline_profit=curve.baseline_profit,
                revenue_lift_pct=curve.revenue_lift_pct,
                profit_lift_pct=curve.profit_lift_pct,
                theoretical_optimal_profit_price=curve.theoretical_optimal_profit_price,
                points=pts_schemas,
                guardrails_applied=curve.guardrails_applied,
            )

        return MultiSegmentCurveResponse(product_id=req.product_id, curves=curves_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/simulate", response_model=SimulationResponse)
def simulate_scenario(
    req: SimulationRequest,
    simulator: PricingScenarioSimulator = Depends(get_scenario_simulator),
) -> SimulationResponse:
    """Simulate market shocks (competitor price drops, cost surge, holiday peak) and evaluate strategies."""
    try:
        sc = PricingScenario(
            name=req.scenario.name,
            description=req.scenario.description,
            competitor_price_change_pct=req.scenario.competitor_price_change_pct,
            cross_price_elasticity=req.scenario.cross_price_elasticity,
            unit_cost_change_pct=req.scenario.unit_cost_change_pct,
            demand_surge_multiplier=req.scenario.demand_surge_multiplier,
            elasticity_shock=req.scenario.elasticity_shock,
            capacity_multiplier=req.scenario.capacity_multiplier,
        )

        res = simulator.compare_strategies(
            product_id=req.product_id,
            customer_group=req.customer_group,
            location_id=req.location_id,
            base_price=req.base_price,
            base_demand=req.base_demand,
            unit_cost=req.unit_cost,
            capacity=req.capacity,
            scenario=sc,
            margin_floor_pct=req.margin_floor_pct,
            max_volatility_pct=req.max_volatility_pct,
        )

        eval_schemas = [
            StrategyEvaluationSchema(
                strategy_name=e.strategy_name,
                recommended_price=e.recommended_price,
                expected_demand=e.expected_demand,
                expected_revenue=e.expected_revenue,
                expected_profit=e.expected_profit,
                margin_pct=e.margin_pct,
                revenue_lift_pct=e.revenue_lift_pct,
                profit_lift_pct=e.profit_lift_pct,
                price_change_pct=e.price_change_pct,
                volatility_violation=e.volatility_violation,
                margin_violation=e.margin_violation,
                capacity_violation=e.capacity_violation,
                is_guardrail_compliant=e.is_guardrail_compliant,
            )
            for e in res.evaluations
        ]

        return SimulationResponse(
            product_id=res.product_id,
            customer_group=res.customer_group,
            location_id=res.location_id,
            scenario_applied=res.scenario_applied,
            evaluations=eval_schemas,
            best_revenue_strategy=res.best_revenue_strategy,
            best_profit_strategy=res.best_profit_strategy,
            best_compliant_strategy=res.best_compliant_strategy,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/optimize", response_model=OptimizationResponse)
def optimize_prices(
    req: OptimizationRequest,
    optimizer: ConstrainedPriceOptimizer = Depends(get_price_optimizer),
    model: DoubleMachineLearningEstimator | None = Depends(get_causal_model),
    repo: RecommendationRepository = Depends(get_recommendation_repository),
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> OptimizationResponse:
    """Solve joint constrained dynamic pricing optimization and persist generated recommendations."""
    try:
        items = []
        for it in req.items:
            # Resolve elasticity if not provided in input
            elasticity = it.elasticity
            se = it.elasticity_std_error
            if elasticity is None:
                if model is not None:
                    est = model.get_elasticity(it.product_id, it.customer_group)
                    elasticity = est.point_estimate
                    se = est.std_error
                else:
                    elasticity = -1.50
                    se = 0.10

            items.append(
                SegmentPricingItem(
                    item_id=it.item_id,
                    product_id=it.product_id,
                    customer_group=it.customer_group,
                    location_id=it.location_id,
                    base_price=it.base_price,
                    base_demand=it.base_demand,
                    unit_cost=it.unit_cost,
                    elasticity=elasticity,
                    elasticity_std_error=se,
                    capacity=it.capacity,
                )
            )

        cfg = OptimizationConfig(
            objective_type=req.objective_type,
            risk_aversion=req.risk_aversion,
            max_price_change_pct=req.max_price_change_pct,
            min_margin_pct_over_cost=req.min_margin_pct_over_cost,
            max_customer_group_disparity_pct=req.max_customer_group_disparity_pct,
            max_location_disparity_pct=req.max_location_disparity_pct,
        )

        res = optimizer.optimize(items, config_override=cfg)

        rec_schemas = []
        records_to_save = []
        now = RecommendationRepository.now_iso()

        for r in res.recommendations:
            rec_id = f"REC_{uuid.uuid4().hex[:8].upper()}"
            rec_schemas.append(
                PriceRecommendationSchema(
                    item_id=r.item_id,
                    product_id=r.product_id,
                    customer_group=r.customer_group,
                    location_id=r.location_id,
                    recommended_price=r.recommended_price,
                    base_price=r.base_price,
                    unit_cost=r.unit_cost,
                    price_change_pct=r.price_change_pct,
                    expected_demand=r.expected_demand,
                    expected_revenue=r.expected_revenue,
                    expected_profit=r.expected_profit,
                    margin_pct=r.margin_pct,
                    revenue_lift_pct=r.revenue_lift_pct,
                    profit_lift_pct=r.profit_lift_pct,
                    status=r.status,
                    is_margin_compliant=r.is_margin_compliant,
                    is_volatility_compliant=r.is_volatility_compliant,
                    is_capacity_compliant=r.is_capacity_compliant,
                )
            )

            records_to_save.append(
                RecommendationRecord(
                    recommendation_id=rec_id,
                    item_id=r.item_id,
                    product_id=r.product_id,
                    customer_group=r.customer_group,
                    location_id=r.location_id,
                    base_price=r.base_price,
                    recommended_price=r.recommended_price,
                    final_price=r.recommended_price,
                    unit_cost=r.unit_cost,
                    expected_demand=r.expected_demand,
                    expected_revenue=r.expected_revenue,
                    expected_profit=r.expected_profit,
                    margin_pct=r.margin_pct,
                    state=RecommendationState.GENERATED,
                    created_by="OPTIMIZER",
                    created_at=now,
                    updated_at=now,
                )
            )

        # Bulk save generated recommendations to SQLite
        if records_to_save:
            repo.bulk_save(records_to_save)
            try:
                audit_logger.log_recommendations_generated(
                    count=len(records_to_save),
                    actor_username="OPTIMIZER",
                    actor_role="system",
                    details={
                        "objective_type": res.objective_type,
                        "total_profit_lift_pct": res.total_profit_lift_pct,
                        "solve_time_ms": res.solve_time_ms,
                    },
                )
            except Exception:
                pass

        violations_schemas = []
        is_compliant = True
        if res.guardrail_check:
            is_compliant = res.guardrail_check.is_compliant
            violations_schemas = [
                GuardrailViolationSchema(
                    rule_name=v.rule_name,
                    item_identifier=v.item_identifier,
                    violation_message=v.violation_message,
                    current_value=v.current_value,
                    allowed_threshold=v.allowed_threshold,
                    severity=v.severity,
                )
                for v in res.guardrail_check.violations
            ]

        return OptimizationResponse(
            status=res.status,
            objective_type=res.objective_type,
            total_recommended_revenue=res.total_recommended_revenue,
            total_recommended_profit=res.total_recommended_profit,
            total_baseline_revenue=res.total_baseline_revenue,
            total_baseline_profit=res.total_baseline_profit,
            total_revenue_lift_pct=res.total_revenue_lift_pct,
            total_profit_lift_pct=res.total_profit_lift_pct,
            solve_time_ms=res.solve_time_ms,
            recommendations=rec_schemas,
            guardrail_compliant=is_compliant,
            violations=violations_schemas,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
