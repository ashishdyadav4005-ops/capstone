"""Pydantic request and response schemas for BDS-39 FastAPI Service."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecommendationState(StrEnum):
    """Lifecycle state machine stages for a dynamic price recommendation."""

    GENERATED = "GENERATED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PUBLISHED = "PUBLISHED"
    ROLLED_BACK = "ROLLED_BACK"


# ---------------------------------------------------------------------------
# Health & Catalog Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Service health status response."""

    status: str = "healthy"
    version: str
    database: str = "connected"
    model_loaded: bool = True


class CatalogProduct(BaseModel):
    """Product catalog metadata with baseline pricing and costs."""

    product_id: str
    product_name: str
    base_price: float
    unit_cost: float
    capacity: float = 100.0


class CatalogResponse(BaseModel):
    """Active catalog discovery response."""

    products: list[CatalogProduct]
    customer_groups: list[str] = ["budget", "regular", "premium", "business"]
    locations: list[str] = ["LOC_URBAN", "LOC_SUBURBAN", "LOC_RURAL", "LOC_ONLINE"]


# ---------------------------------------------------------------------------
# Demand Curves Schemas
# ---------------------------------------------------------------------------


class CurvePointSchema(BaseModel):
    """Single price evaluation point along a counterfactual demand curve."""

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


class SingleCurveRequest(BaseModel):
    """Request to generate a counterfactual demand curve for a product-segment."""

    product_id: str = "PROD_001"
    customer_group: str = "regular"
    location_id: str = "LOC_URBAN"
    base_price: float = Field(default=100.0, gt=0)
    base_demand: float = Field(default=50.0, gt=0)
    unit_cost: float = Field(default=60.0, ge=0)
    capacity: float | None = 100.0
    grid_points: int = Field(default=50, ge=5, le=200)
    margin_floor_pct: float = 0.15
    max_volatility_pct: float = 0.15


class DemandCurveResultSchema(BaseModel):
    """Output for a single counterfactual demand curve."""

    product_id: str
    customer_group: str
    location_id: str
    base_price: float
    base_demand: float
    unit_cost: float
    elasticity: float
    elasticity_std_error: float
    elasticity_ci_95: list[float]
    is_cold_start: bool
    optimal_revenue_price: float
    max_expected_revenue: float
    optimal_profit_price: float
    max_expected_profit: float
    baseline_revenue: float
    baseline_profit: float
    revenue_lift_pct: float
    profit_lift_pct: float
    theoretical_optimal_profit_price: float | None
    points: list[CurvePointSchema]
    guardrails_applied: dict[str, Any]


class MultiSegmentCurveRequest(BaseModel):
    """Request to generate batch curves across customer segments."""

    product_id: str = "PROD_001"
    customer_groups: list[str] = ["budget", "regular", "premium", "business"]
    location_id: str = "LOC_URBAN"
    base_price: float = Field(default=100.0, gt=0)
    unit_cost: float = Field(default=60.0, ge=0)
    capacity: float | None = 100.0
    grid_points: int = Field(default=50, ge=5, le=200)


class MultiSegmentCurveResponse(BaseModel):
    """Multi-segment batch demand curves."""

    product_id: str
    curves: dict[str, DemandCurveResultSchema]


# ---------------------------------------------------------------------------
# What-If Simulation Schemas
# ---------------------------------------------------------------------------


class ScenarioShockSchema(BaseModel):
    """Market shock parameters for what-if simulations."""

    name: str = "Custom Shock"
    description: str = "Simulated market scenario"
    competitor_price_change_pct: float = Field(default=0.0, ge=-0.80, le=2.0)
    cross_price_elasticity: float = 0.35
    unit_cost_change_pct: float = Field(default=0.0, ge=-0.50, le=2.0)
    demand_surge_multiplier: float = Field(default=1.0, gt=0)
    elasticity_shock: float = 0.0
    capacity_multiplier: float = Field(default=1.0, gt=0)


class SimulationRequest(BaseModel):
    """Request payload for pricing scenario simulation."""

    product_id: str = "PROD_001"
    customer_group: str = "regular"
    location_id: str = "LOC_URBAN"
    base_price: float = Field(default=100.0, gt=0)
    base_demand: float = Field(default=50.0, gt=0)
    unit_cost: float = Field(default=60.0, ge=0)
    capacity: float | None = 100.0
    scenario: ScenarioShockSchema = Field(default_factory=ScenarioShockSchema)
    margin_floor_pct: float = 0.15
    max_volatility_pct: float = 0.15


class StrategyEvaluationSchema(BaseModel):
    """Evaluated metric row for a pricing strategy under a scenario."""

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


class SimulationResponse(BaseModel):
    """What-if simulation outcome and strategy benchmark."""

    product_id: str
    customer_group: str
    location_id: str
    scenario_applied: str
    evaluations: list[StrategyEvaluationSchema]
    best_revenue_strategy: str
    best_profit_strategy: str
    best_compliant_strategy: str


# ---------------------------------------------------------------------------
# Optimization Schemas
# ---------------------------------------------------------------------------


class OptimizationItemInput(BaseModel):
    """Single item input for portfolio optimization."""

    item_id: str
    product_id: str
    customer_group: str
    location_id: str
    base_price: float = Field(gt=0)
    base_demand: float = Field(gt=0)
    unit_cost: float = Field(ge=0)
    elasticity: float | None = None
    elasticity_std_error: float = 0.10
    capacity: float | None = 100.0


class OptimizationRequest(BaseModel):
    """Request payload for constrained dynamic pricing optimization."""

    items: list[OptimizationItemInput]
    objective_type: str = "expected_profit" # "expected_profit" or "expected_revenue"
    risk_aversion: float = 0.05
    max_price_change_pct: float = 0.15
    min_margin_pct_over_cost: float = 0.15
    max_customer_group_disparity_pct: float = 0.12
    max_location_disparity_pct: float = 0.10


class PriceRecommendationSchema(BaseModel):
    """Individual optimized price output."""

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
    status: str
    is_margin_compliant: bool
    is_volatility_compliant: bool
    is_capacity_compliant: bool


class GuardrailViolationSchema(BaseModel):
    """Guardrail violation record."""

    rule_name: str
    item_identifier: str
    violation_message: str
    current_value: float
    allowed_threshold: float
    severity: str


class OptimizationResponse(BaseModel):
    """Constrained optimization result."""

    status: str
    objective_type: str
    total_recommended_revenue: float
    total_recommended_profit: float
    total_baseline_revenue: float
    total_baseline_profit: float
    total_revenue_lift_pct: float
    total_profit_lift_pct: float
    solve_time_ms: float
    recommendations: list[PriceRecommendationSchema]
    guardrail_compliant: bool
    violations: list[GuardrailViolationSchema] = []


# ---------------------------------------------------------------------------
# Recommendation Lifecycle Schemas
# ---------------------------------------------------------------------------


class RecommendationRecord(BaseModel):
    """Stored recommendation entity with lifecycle metadata."""

    model_config = ConfigDict(from_attributes=True)

    recommendation_id: str
    item_id: str
    product_id: str
    customer_group: str
    location_id: str
    base_price: float
    recommended_price: float
    final_price: float
    unit_cost: float
    expected_demand: float
    expected_revenue: float
    expected_profit: float
    margin_pct: float
    state: RecommendationState
    created_by: str = "SYSTEM"
    reviewed_by: str | None = None
    override_price: float | None = None
    override_reason: str | None = None
    audit_notes: str | None = None
    created_at: str
    updated_at: str


class OverrideRequest(BaseModel):
    """Request payload for manager price override."""

    override_price: float = Field(gt=0)
    reason: str = Field(min_length=5, description="Mandatory reason for overriding algorithmic price")
    reviewed_by: str = "manager"


class ActionRequest(BaseModel):
    """Action payload for reviewing, approving, rejecting, publishing, or rolling back."""

    actor: str = "manager"
    notes: str | None = None
    reason: str | None = None
