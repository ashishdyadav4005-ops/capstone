"""Pricing package: Counterfactual demand curves, market simulation, and constrained optimization."""

from src.pricing.curves import (
    DemandCurveGenerator,
    DemandCurveResult,
    DemandPoint,
    MultiSegmentCurveResult,
)
from src.pricing.guardrails import (
    GuardrailCheckResult,
    GuardrailsEngine,
    GuardrailViolation,
)
from src.pricing.optimizer import (
    ConstrainedPriceOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptimizedPriceRecommendation,
    SegmentPricingItem,
)
from src.pricing.simulator import (
    CustomerGroupImpact,
    PricingScenario,
    PricingScenarioSimulator,
    StrategyComparisonResult,
    StrategyEvaluation,
)

__all__ = [
    "DemandPoint",
    "DemandCurveResult",
    "MultiSegmentCurveResult",
    "DemandCurveGenerator",
    "PricingScenario",
    "StrategyEvaluation",
    "StrategyComparisonResult",
    "CustomerGroupImpact",
    "PricingScenarioSimulator",
    "GuardrailViolation",
    "GuardrailCheckResult",
    "GuardrailsEngine",
    "SegmentPricingItem",
    "OptimizationConfig",
    "OptimizedPriceRecommendation",
    "OptimizationResult",
    "ConstrainedPriceOptimizer",
]
