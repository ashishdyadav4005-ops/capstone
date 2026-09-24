"""FastAPI Dependency Injection providers for pricing services and repositories."""

from __future__ import annotations

from functools import lru_cache

import joblib

from src.api.lifecycle import RecommendationLifecycleManager
from src.api.repository import RecommendationRepository
from src.audit.logger import AuditLogger
from src.audit.repository import AuditRepository
from src.common.config import find_project_root, get_config
from src.common.logger import get_logger
from src.models.causal_dml import DoubleMachineLearningEstimator
from src.monitoring.drift import DriftDetector
from src.pricing.curves import DemandCurveGenerator
from src.pricing.guardrails import GuardrailsEngine
from src.pricing.optimizer import ConstrainedPriceOptimizer
from src.pricing.simulator import PricingScenarioSimulator

logger = get_logger("api_dependencies")


@lru_cache
def get_audit_repository() -> AuditRepository:
    """Provide singleton AuditRepository."""
    return AuditRepository()


@lru_cache
def get_audit_logger() -> AuditLogger:
    """Provide singleton AuditLogger."""
    repo = get_audit_repository()
    return AuditLogger(repository=repo)


@lru_cache
def get_causal_model() -> DoubleMachineLearningEstimator | None:
    """Load or cache trained Causal DML model artifact."""
    root = find_project_root()
    model_path = root / "data" / "models" / "causal_dml_model.pkl"
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            logger.info("Loaded pre-trained Causal DML model into API dependency cache.", extra={"path": str(model_path)})
            return model
        except Exception as e:
            logger.warning("Failed to load causal model file, fallback used.", extra={"error": str(e)})
    return None


@lru_cache
def get_guardrails_engine() -> GuardrailsEngine:
    """Provide singleton GuardrailsEngine."""
    config = get_config()
    raw_guardrails = config.raw_configs.get("guardrails", {})
    vol = raw_guardrails.get("volatility", {})
    margin = raw_guardrails.get("margin", {})
    fairness = raw_guardrails.get("fairness", {})
    cap = raw_guardrails.get("capacity", {})

    return GuardrailsEngine(
        max_price_change_pct=vol.get("max_price_change_pct", 0.15),
        min_margin_pct_over_cost=margin.get("min_margin_pct_over_cost", 0.15),
        absolute_min_margin_currency=margin.get("absolute_min_margin_currency", 2.0),
        max_customer_group_disparity_pct=fairness.get("max_customer_group_disparity_pct", 0.12),
        max_location_disparity_pct=fairness.get("max_location_disparity_pct", 0.10),
        max_capacity_utilization=cap.get("max_utilization_threshold", 0.98),
    )


@lru_cache
def get_curve_generator() -> DemandCurveGenerator:
    """Provide singleton DemandCurveGenerator."""
    model = get_causal_model()
    return DemandCurveGenerator(causal_model=model)


@lru_cache
def get_scenario_simulator() -> PricingScenarioSimulator:
    """Provide singleton PricingScenarioSimulator."""
    curve_gen = get_curve_generator()
    return PricingScenarioSimulator(curve_generator=curve_gen)


@lru_cache
def get_price_optimizer() -> ConstrainedPriceOptimizer:
    """Provide singleton ConstrainedPriceOptimizer."""
    return ConstrainedPriceOptimizer()


@lru_cache
def get_recommendation_repository() -> RecommendationRepository:
    """Provide singleton RecommendationRepository."""
    return RecommendationRepository()


@lru_cache
def get_lifecycle_manager() -> RecommendationLifecycleManager:
    """Provide singleton RecommendationLifecycleManager."""
    repo = get_recommendation_repository()
    guardrails = get_guardrails_engine()
    audit_logger = get_audit_logger()
    return RecommendationLifecycleManager(
        repository=repo,
        guardrails_engine=guardrails,
        audit_logger=audit_logger,
    )


@lru_cache
def get_drift_detector() -> DriftDetector:
    """Provide singleton DriftDetector initialized with baseline reference sales data."""
    import pandas as pd

    from src.data.generator import SyntheticDataGenerator

    root = find_project_root()
    ref_path = root / "data" / "processed" / "train.parquet"
    if ref_path.exists():
        try:
            df_ref = pd.read_parquet(ref_path)
            return DriftDetector(reference_data=df_ref)
        except Exception:
            pass

    generator = SyntheticDataGenerator(random_seed=42, n_days=180)
    df_ref = generator.generate()
    return DriftDetector(reference_data=df_ref)

