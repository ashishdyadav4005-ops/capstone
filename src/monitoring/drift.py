"""Mathematical data and concept drift detection engine.

Implements Population Stability Index (PSI), Wasserstein Distance (Earth Mover's Distance),
Kolmogorov-Smirnov (KS) two-sample statistical tests, and residual-based concept drift analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from src.common.logger import get_logger

logger = get_logger("drift_monitoring")


class DriftStatus(StrEnum):
    """Categorical alert level for distribution drift."""

    STABLE = "STABLE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


PSI_WARNING_THRESHOLD: float = 0.10
PSI_CRITICAL_THRESHOLD: float = 0.25
KS_ALPHA_THRESHOLD: float = 0.05


def calculate_psi(
    reference: np.ndarray | pd.Series | list[float],
    current: np.ndarray | pd.Series | list[float],
    num_bins: int = 10,
    is_categorical: bool = False,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current samples.

    Parameters
    ----------
    reference : array-like
        Baseline or training distribution sample.
    current : array-like
        Current production or scoring distribution sample.
    num_bins : int, default=10
        Number of quantile bins for continuous features.
    is_categorical : bool, default=False
        Whether the inputs are discrete/categorical values.
    epsilon : float, default=1e-4
        Smoothing constant to prevent zero divisions and log(0).

    Returns
    -------
    float
        Population Stability Index score (>= 0.0).
    """
    ref_arr = np.asarray(reference).ravel()
    cur_arr = np.asarray(current).ravel()

    # Drop NaNs
    ref_arr = ref_arr[~pd.isna(ref_arr)]
    cur_arr = cur_arr[~pd.isna(cur_arr)]

    if len(ref_arr) == 0 or len(cur_arr) == 0:
        return 0.0

    if is_categorical:
        categories = np.union1d(np.unique(ref_arr), np.unique(cur_arr))
        ref_counts = np.array([np.sum(ref_arr == c) for c in categories], dtype=float)
        cur_counts = np.array([np.sum(cur_arr == c) for c in categories], dtype=float)

        ref_pct = ref_counts / len(ref_arr)
        cur_pct = cur_counts / len(cur_arr)
    else:
        # Check if reference is constant
        if np.all(ref_arr == ref_arr[0]):
            if np.all(cur_arr == ref_arr[0]):
                return 0.0
            return float(PSI_CRITICAL_THRESHOLD * 2.0)

        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(ref_arr, quantiles)
        bins = np.unique(bins)

        if len(bins) < 2:
            bins = np.array([bins[0] - 1e-3, bins[0] + 1e-3])

        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(ref_arr, bins=bins)
        cur_counts, _ = np.histogram(cur_arr, bins=bins)

        ref_pct = ref_counts.astype(float) / len(ref_arr)
        cur_pct = cur_counts.astype(float) / len(cur_arr)

    # Apply epsilon smoothing and renormalize
    ref_pct = np.clip(ref_pct, epsilon, 1.0)
    cur_pct = np.clip(cur_pct, epsilon, 1.0)

    ref_pct = ref_pct / np.sum(ref_pct)
    cur_pct = cur_pct / np.sum(cur_pct)

    # PSI = sum((Actual - Expected) * ln(Actual / Expected))
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(max(0.0, round(float(psi), 4)))


def calculate_wasserstein(
    reference: np.ndarray | pd.Series | list[float],
    current: np.ndarray | pd.Series | list[float],
) -> float:
    """Calculate Earth Mover's Distance (1-Wasserstein Distance) between samples."""
    ref_arr = np.asarray(reference, dtype=float).ravel()
    cur_arr = np.asarray(current, dtype=float).ravel()

    ref_arr = ref_arr[~np.isnan(ref_arr)]
    cur_arr = cur_arr[~np.isnan(cur_arr)]

    if len(ref_arr) == 0 or len(cur_arr) == 0:
        return 0.0

    return float(round(wasserstein_distance(ref_arr, cur_arr), 4))


def calculate_ks_test(
    reference: np.ndarray | pd.Series | list[float],
    current: np.ndarray | pd.Series | list[float],
) -> tuple[float, float]:
    """Run two-sample Kolmogorov-Smirnov test for continuous distributions."""
    ref_arr = np.asarray(reference, dtype=float).ravel()
    cur_arr = np.asarray(current, dtype=float).ravel()

    ref_arr = ref_arr[~np.isnan(ref_arr)]
    cur_arr = cur_arr[~np.isnan(cur_arr)]

    if len(ref_arr) == 0 or len(cur_arr) == 0:
        return 0.0, 1.0

    res = ks_2samp(ref_arr, cur_arr)
    return float(round(res.statistic, 4)), float(round(res.pvalue, 6))


@dataclass
class FeatureDriftResult:
    """Individual feature statistical drift metrics."""

    feature_name: str
    psi_score: float
    wasserstein_distance: float
    ks_statistic: float
    ks_p_value: float
    drift_status: DriftStatus
    is_drift_detected: bool
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        d = asdict(self)
        d["drift_status"] = self.drift_status.value
        return d


@dataclass
class ConceptDriftResult:
    """Outcome of model residual and prediction calibration drift evaluation."""

    baseline_mae: float
    current_mae: float
    residual_shift: float
    ks_statistic: float
    ks_p_value: float
    drift_status: DriftStatus
    is_drift_detected: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        d = asdict(self)
        d["drift_status"] = self.drift_status.value
        return d


@dataclass
class DriftReport:
    """Consolidated multi-feature dataset and model health drift report."""

    evaluated_at: str
    reference_sample_size: int
    current_sample_size: int
    overall_status: DriftStatus
    retraining_recommended: bool
    features: list[FeatureDriftResult] = field(default_factory=list)
    concept_drift: ConceptDriftResult | None = None
    critical_features: list[str] = field(default_factory=list)
    warning_features: list[str] = field(default_factory=list)

    def summary_table(self) -> pd.DataFrame:
        """Convert feature drift metrics to pandas DataFrame."""
        rows = []
        for f in self.features:
            rows.append({
                "feature": f.feature_name,
                "psi_score": f.psi_score,
                "drift_status": f.drift_status.value,
                "wasserstein_dist": f.wasserstein_distance,
                "ks_stat": f.ks_statistic,
                "ks_p_val": f.ks_p_value,
                "ref_mean": round(f.reference_mean, 2),
                "cur_mean": round(f.current_mean, 2),
            })
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Convert full report to serializable dictionary."""
        return {
            "evaluated_at": self.evaluated_at,
            "reference_sample_size": self.reference_sample_size,
            "current_sample_size": self.current_sample_size,
            "overall_status": self.overall_status.value,
            "retraining_recommended": self.retraining_recommended,
            "critical_features": self.critical_features,
            "warning_features": self.warning_features,
            "features": [f.to_dict() for f in self.features],
            "concept_drift": self.concept_drift.to_dict() if self.concept_drift else None,
        }


class DriftDetector:
    """Production data and concept drift detection engine."""

    DEFAULT_FEATURES = [
        "price",
        "quantity",
        "competitor_price",
        "cost",
        "unit_cost",
        "weather_temp",
        "customer_group",
    ]

    def __init__(
        self,
        reference_data: pd.DataFrame | None = None,
        key_features: list[str] | None = None,
        psi_warning_thresh: float = PSI_WARNING_THRESHOLD,
        psi_critical_thresh: float = PSI_CRITICAL_THRESHOLD,
    ):
        self.reference_data = reference_data.copy() if reference_data is not None else None
        self.key_features = key_features or self.DEFAULT_FEATURES
        self.psi_warning_thresh = psi_warning_thresh
        self.psi_critical_thresh = psi_critical_thresh

    def set_reference(self, reference_data: pd.DataFrame) -> None:
        """Set or update the baseline reference dataset."""
        self.reference_data = reference_data.copy()

    def compute_feature_drift(
        self,
        current_data: pd.DataFrame,
        feature_name: str,
        num_bins: int = 10,
    ) -> FeatureDriftResult:
        """Compute drift metrics for a single feature between reference and current data."""
        if self.reference_data is None:
            raise ValueError("Reference dataset has not been set in DriftDetector.")

        if feature_name not in self.reference_data.columns:
            raise KeyError(f"Feature '{feature_name}' not found in reference data.")
        if feature_name not in current_data.columns:
            raise KeyError(f"Feature '{feature_name}' not found in current data.")

        ref_series = self.reference_data[feature_name]
        cur_series = current_data[feature_name]

        is_cat = not np.issubdtype(ref_series.dtype, np.number) or len(np.unique(ref_series)) <= 5

        psi = calculate_psi(
            reference=ref_series,
            current=cur_series,
            num_bins=num_bins,
            is_categorical=is_cat,
        )

        if is_cat:
            w_dist = 0.0
            ks_stat = 0.0
            ks_pval = 1.0
            ref_m = 0.0
            cur_m = 0.0
            ref_s = 0.0
            cur_s = 0.0
        else:
            w_dist = calculate_wasserstein(ref_series, cur_series)
            ks_stat, ks_pval = calculate_ks_test(ref_series, cur_series)
            ref_m = float(np.nanmean(ref_series))
            cur_m = float(np.nanmean(cur_series))
            ref_s = float(np.nanstd(ref_series))
            cur_s = float(np.nanstd(cur_series))

        # Status classification
        if psi >= self.psi_critical_thresh:
            status = DriftStatus.CRITICAL
            msg = f"Critical drift detected (PSI={psi:.4f} >= {self.psi_critical_thresh}). Immediate recalibration needed."
            drift_detected = True
        elif psi >= self.psi_warning_thresh or (not is_cat and ks_pval < KS_ALPHA_THRESHOLD and w_dist > (ref_s * 0.20 if ref_s > 0 else 1.0)):
            status = DriftStatus.WARNING
            msg = f"Moderate drift detected (PSI={psi:.4f}). Monitoring recommended."
            drift_detected = True
        else:
            status = DriftStatus.STABLE
            msg = f"Distribution stable (PSI={psi:.4f})."
            drift_detected = False

        return FeatureDriftResult(
            feature_name=feature_name,
            psi_score=psi,
            wasserstein_distance=w_dist,
            ks_statistic=ks_stat,
            ks_p_value=ks_pval,
            drift_status=status,
            is_drift_detected=drift_detected,
            reference_mean=round(ref_m, 4),
            current_mean=round(cur_m, 4),
            reference_std=round(ref_s, 4),
            current_std=round(cur_s, 4),
            message=msg,
        )

    def evaluate_concept_drift(
        self,
        y_true_ref: np.ndarray | pd.Series,
        y_pred_ref: np.ndarray | pd.Series,
        y_true_cur: np.ndarray | pd.Series,
        y_pred_cur: np.ndarray | pd.Series,
    ) -> ConceptDriftResult:
        """Analyze model prediction residual distribution shift (concept drift)."""
        ref_resid = np.asarray(y_true_ref, dtype=float) - np.asarray(y_pred_ref, dtype=float)
        cur_resid = np.asarray(y_true_cur, dtype=float) - np.asarray(y_pred_cur, dtype=float)

        ref_mae = float(np.mean(np.abs(ref_resid)))
        cur_mae = float(np.mean(np.abs(cur_resid)))
        shift = float(cur_mae - ref_mae)

        ks_stat, ks_pval = calculate_ks_test(ref_resid, cur_resid)

        if ks_pval < KS_ALPHA_THRESHOLD and (cur_mae > ref_mae * 1.30):
            status = DriftStatus.CRITICAL
            msg = f"Significant concept drift: Model MAE increased by {((cur_mae-ref_mae)/ref_mae*100):.1f}% (KS p-val={ks_pval:.4f})."
            drift_det = True
        elif ks_pval < KS_ALPHA_THRESHOLD or (cur_mae > ref_mae * 1.15):
            status = DriftStatus.WARNING
            msg = f"Mild concept drift: Error distribution shifted (KS stat={ks_stat:.4f}, p-val={ks_pval:.4f})."
            drift_det = True
        else:
            status = DriftStatus.STABLE
            msg = f"Model residual distribution stable (Ref MAE={ref_mae:.2f}, Cur MAE={cur_mae:.2f})."
            drift_det = False

        return ConceptDriftResult(
            baseline_mae=round(ref_mae, 4),
            current_mae=round(cur_mae, 4),
            residual_shift=round(shift, 4),
            ks_statistic=ks_stat,
            ks_p_value=ks_pval,
            drift_status=status,
            is_drift_detected=drift_det,
            message=msg,
        )

    def evaluate_dataset_drift(
        self,
        current_data: pd.DataFrame,
        features_to_check: list[str] | None = None,
        concept_inputs: dict[str, Any] | None = None,
    ) -> DriftReport:
        """Run comprehensive statistical drift audit across all target features."""
        if self.reference_data is None:
            raise ValueError("Reference dataset has not been configured.")

        features = features_to_check or [f for f in self.key_features if f in self.reference_data.columns and f in current_data.columns]
        if not features:
            features = [c for c in self.reference_data.columns if c in current_data.columns and c not in ["date", "item_id", "timestamp"]]

        results: list[FeatureDriftResult] = []
        criticals: list[str] = []
        warnings: list[str] = []

        for feat in features:
            try:
                res = self.compute_feature_drift(current_data, feat)
                results.append(res)
                if res.drift_status == DriftStatus.CRITICAL:
                    criticals.append(feat)
                elif res.drift_status == DriftStatus.WARNING:
                    warnings.append(feat)
            except Exception as e:
                logger.warning(f"Failed to compute drift for feature {feat}: {e}")

        # Concept drift if provided
        concept_res = None
        if concept_inputs is not None:
            concept_res = self.evaluate_concept_drift(
                y_true_ref=concept_inputs["y_true_ref"],
                y_pred_ref=concept_inputs["y_pred_ref"],
                y_true_cur=concept_inputs["y_true_cur"],
                y_pred_cur=concept_inputs["y_pred_cur"],
            )
            if concept_res.drift_status == DriftStatus.CRITICAL:
                criticals.append("model_residuals")
            elif concept_res.drift_status == DriftStatus.WARNING:
                warnings.append("model_residuals")

        # Overall status
        if len(criticals) > 0:
            overall = DriftStatus.CRITICAL
            retrain = True
        elif len(warnings) > 0:
            overall = DriftStatus.WARNING
            retrain = False
        else:
            overall = DriftStatus.STABLE
            retrain = False

        return DriftReport(
            evaluated_at=datetime.now(UTC).isoformat(),
            reference_sample_size=len(self.reference_data),
            current_sample_size=len(current_data),
            overall_status=overall,
            retraining_recommended=retrain,
            features=results,
            concept_drift=concept_res,
            critical_features=criticals,
            warning_features=warnings,
        )
