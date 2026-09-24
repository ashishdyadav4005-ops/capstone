"""Monitoring API Router: Prometheus metrics exposition and on-demand drift detection."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from src.api.dependencies import get_drift_detector
from src.auth.dependencies import require_roles
from src.auth.roles import UserRole
from src.data.generator import SyntheticDataGenerator
from src.monitoring.drift import DriftDetector, DriftReport
from src.monitoring.metrics import (
    PROMETHEUS_CONTENT_TYPE,
    get_metrics_registry,
)

router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring & Observability"])

_latest_drift_report: dict[str, Any] | None = None


class DriftCheckRequest(BaseModel):
    """Payload for on-demand dataset drift analysis."""

    records: list[dict[str, Any]] | None = Field(default=None, description="Current scoring or production records batch")
    features: list[str] | None = Field(default=None, description="Subset of features to analyze")
    psi_warning_threshold: float = Field(default=0.10, ge=0.01, le=1.0)
    psi_critical_threshold: float = Field(default=0.25, ge=0.05, le=1.0)


class FeatureDriftSchema(BaseModel):
    """Statistical drift metrics for an individual feature."""

    feature_name: str
    psi_score: float
    wasserstein_distance: float
    ks_statistic: float
    ks_p_value: float
    drift_status: str
    is_drift_detected: bool
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    message: str


class DriftReportResponse(BaseModel):
    """Outcome of dataset and concept drift audit."""

    evaluated_at: str
    reference_sample_size: int
    current_sample_size: int
    overall_status: str
    retraining_recommended: bool
    critical_features: list[str]
    warning_features: list[str]
    features: list[FeatureDriftSchema]
    concept_drift: dict[str, Any] | None = None


class MonitoringHealthResponse(BaseModel):
    """Operational telemetry and metric collector status."""

    status: str = "healthy"
    metrics_exporter: str = "active"
    drift_engine: str = "active"
    audit_chain_valid: bool = True
    active_locked_accounts: int = 0
    total_pricing_requests: float = 0.0


@router.get("/metrics")
def get_metrics() -> Response:
    """Export all system and pricing observability metrics in standard Prometheus text format."""
    registry = get_metrics_registry()
    content = registry.generate_prometheus_text()
    return Response(content=content, media_type=PROMETHEUS_CONTENT_TYPE)


@router.post(
    "/drift/check",
    response_model=DriftReportResponse,
    dependencies=[Depends(require_roles([UserRole.ADMIN, UserRole.ANALYST, UserRole.MANAGER, UserRole.GOVERNANCE]))],
)
def run_drift_check(
    request: DriftCheckRequest | None = None,
    detector: DriftDetector = Depends(get_drift_detector),
) -> DriftReportResponse:
    """Run on-demand statistical drift evaluation against baseline reference data."""
    global _latest_drift_report

    req = request or DriftCheckRequest()

    if req.records and len(req.records) > 0:
        df_current = pd.DataFrame(req.records)
    else:
        # Generate a current batch if not explicitly supplied
        generator = SyntheticDataGenerator(random_seed=99, n_days=30)
        df_current = generator.generate()

    # Configure custom thresholds if specified
    detector.psi_warning_thresh = req.psi_warning_threshold
    detector.psi_critical_thresh = req.psi_critical_threshold

    report: DriftReport = detector.evaluate_dataset_drift(
        current_data=df_current,
        features_to_check=req.features,
    )

    # Update Prometheus gauges
    registry = get_metrics_registry()
    for feat_res in report.features:
        registry.feature_drift_psi.set(feat_res.psi_score, feature_name=feat_res.feature_name)

    dict_report = report.to_dict()
    _latest_drift_report = dict_report

    return DriftReportResponse(**dict_report)


@router.get(
    "/drift/latest",
    response_model=DriftReportResponse,
    dependencies=[Depends(require_roles([UserRole.ADMIN, UserRole.ANALYST, UserRole.MANAGER, UserRole.GOVERNANCE]))],
)
def get_latest_drift_report(
    detector: DriftDetector = Depends(get_drift_detector),
) -> DriftReportResponse:
    """Retrieve the most recent statistical drift audit report."""
    global _latest_drift_report

    if _latest_drift_report is None:
        generator = SyntheticDataGenerator(random_seed=99, n_days=30)
        df_current = generator.generate()
        report: DriftReport = detector.evaluate_dataset_drift(current_data=df_current)
        _latest_drift_report = report.to_dict()

    return DriftReportResponse(**_latest_drift_report)


@router.get("/health", response_model=MonitoringHealthResponse)
def get_monitoring_health() -> MonitoringHealthResponse:
    """Return runtime monitoring telemetry and metric health summary."""
    registry = get_metrics_registry()
    return MonitoringHealthResponse(
        status="healthy",
        metrics_exporter="active",
        drift_engine="active",
        audit_chain_valid=bool(registry.audit_chain_valid.get() == 1.0),
        active_locked_accounts=int(registry.active_locked_accounts.get()),
    )
