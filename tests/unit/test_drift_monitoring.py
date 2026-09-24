"""Unit tests for mathematical drift detection algorithms and Prometheus metrics collector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    DriftDetector,
    DriftStatus,
    calculate_ks_test,
    calculate_psi,
    calculate_wasserstein,
)
from src.monitoring.metrics import (
    Counter,
    Gauge,
    Histogram,
    TimingContext,
    get_metrics_registry,
)


class TestDriftAlgorithms:
    """Test suite for mathematical statistical drift calculations."""

    def test_psi_identical_distributions_returns_near_zero(self):
        """Verify PSI is near zero for identical distributions."""
        np.random.seed(42)
        ref = np.random.normal(loc=100.0, scale=15.0, size=2000)
        cur = np.random.normal(loc=100.0, scale=15.0, size=2000)

        psi = calculate_psi(ref, cur, num_bins=10)
        assert psi < 0.05
        assert psi >= 0.0

    def test_psi_shifted_distribution_returns_high_score(self):
        """Verify PSI correctly identifies large distribution shift (> 0.25)."""
        np.random.seed(42)
        ref = np.random.normal(loc=100.0, scale=10.0, size=2000)
        cur = np.random.normal(loc=140.0, scale=10.0, size=2000) # +40 mean shift

        psi = calculate_psi(ref, cur, num_bins=10)
        assert psi > 0.25

    def test_psi_categorical_distribution(self):
        """Verify PSI calculation on discrete categorical arrays."""
        ref_cat = np.array(["budget"] * 500 + ["regular"] * 300 + ["premium"] * 200)
        cur_stable = np.array(["budget"] * 490 + ["regular"] * 310 + ["premium"] * 200)
        cur_shifted = np.array(["budget"] * 100 + ["regular"] * 100 + ["premium"] * 800)

        psi_stable = calculate_psi(ref_cat, cur_stable, is_categorical=True)
        assert psi_stable < 0.05

        psi_shifted = calculate_psi(ref_cat, cur_shifted, is_categorical=True)
        assert psi_shifted > 0.25

    def test_wasserstein_and_ks_test(self):
        """Verify Earth Mover's Distance and Kolmogorov-Smirnov test."""
        np.random.seed(42)
        ref = np.random.normal(50.0, 5.0, 1000)
        cur_same = np.random.normal(50.0, 5.0, 1000)
        cur_diff = np.random.normal(70.0, 5.0, 1000)

        w_same = calculate_wasserstein(ref, cur_same)
        w_diff = calculate_wasserstein(ref, cur_diff)
        assert w_same < 2.0
        assert w_diff > 15.0

        ks_stat_same, ks_p_same = calculate_ks_test(ref, cur_same)
        ks_stat_diff, ks_p_diff = calculate_ks_test(ref, cur_diff)

        assert ks_p_same > 0.01
        assert ks_p_diff < 1e-4
        assert ks_stat_diff > 0.5


class TestDriftDetectorEngine:
    """Test suite for high-level DriftDetector orchestrator."""

    @pytest.fixture
    def reference_dataset(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            "price": np.random.normal(100.0, 15.0, n),
            "quantity": np.random.poisson(40, n),
            "competitor_price": np.random.normal(95.0, 12.0, n),
            "unit_cost": np.random.normal(60.0, 5.0, n),
            "customer_group": np.random.choice(["budget", "regular", "premium"], size=n),
        })

    def test_drift_detector_stable_dataset(self, reference_dataset: pd.DataFrame):
        """Verify stable production dataset returns STABLE overall status."""
        detector = DriftDetector(reference_data=reference_dataset)

        np.random.seed(43)
        n = 500
        cur_df = pd.DataFrame({
            "price": np.random.normal(100.0, 15.0, n),
            "quantity": np.random.poisson(40, n),
            "competitor_price": np.random.normal(95.0, 12.0, n),
            "unit_cost": np.random.normal(60.0, 5.0, n),
            "customer_group": np.random.choice(["budget", "regular", "premium"], size=n),
        })

        report = detector.evaluate_dataset_drift(cur_df)
        assert report.overall_status == DriftStatus.STABLE
        assert report.retraining_recommended is False
        assert len(report.critical_features) == 0

        summary = report.summary_table()
        assert len(summary) >= 4
        assert "psi_score" in summary.columns

    def test_drift_detector_critical_drift(self, reference_dataset: pd.DataFrame):
        """Verify shifted price distribution triggers CRITICAL status and retraining recommendation."""
        detector = DriftDetector(reference_data=reference_dataset)

        np.random.seed(42)
        n = 500
        cur_df = pd.DataFrame({
            "price": np.random.normal(180.0, 15.0, n), # Extreme price inflation (+80%)
            "quantity": np.random.poisson(15, n), # Demand collapse
            "competitor_price": np.random.normal(95.0, 12.0, n),
            "unit_cost": np.random.normal(60.0, 5.0, n),
            "customer_group": np.random.choice(["budget", "regular", "premium"], size=n),
        })

        report = detector.evaluate_dataset_drift(cur_df)
        assert report.overall_status == DriftStatus.CRITICAL
        assert report.retraining_recommended is True
        assert "price" in report.critical_features or "quantity" in report.critical_features

    def test_concept_drift_evaluation(self):
        """Verify model residual concept drift evaluation."""
        detector = DriftDetector()

        # Baseline predictions: small residuals
        y_true_ref = np.array([50.0, 60.0, 70.0, 80.0, 90.0] * 50)
        y_pred_ref = y_true_ref + np.random.normal(0, 2.0, len(y_true_ref))

        # Current predictions: model degraded significantly
        y_true_cur = np.array([50.0, 60.0, 70.0, 80.0, 90.0] * 50)
        y_pred_cur = y_true_cur + np.random.normal(15.0, 8.0, len(y_true_cur))

        concept = detector.evaluate_concept_drift(
            y_true_ref=y_true_ref,
            y_pred_ref=y_pred_ref,
            y_true_cur=y_true_cur,
            y_pred_cur=y_pred_cur,
        )

        assert concept.is_drift_detected is True
        assert concept.drift_status in [DriftStatus.WARNING, DriftStatus.CRITICAL]
        assert concept.current_mae > concept.baseline_mae


class TestPrometheusMetricsCollector:
    """Test suite for Prometheus Counter, Gauge, Histogram, and text exposition."""

    def test_counter_operations(self):
        """Verify Counter incrementing and text formatting."""
        c = Counter(name="test_requests_total", description="Test counter", label_names=["method", "status"])
        c.inc(method="GET", status="200")
        c.inc(amount=4.0, method="GET", status="200")
        c.inc(method="POST", status="500")

        assert c.get(method="GET", status="200") == 5.0
        assert c.get(method="POST", status="500") == 1.0

        txt = c.format_prometheus()
        assert '# TYPE test_requests_total counter' in txt
        assert 'test_requests_total{method="GET",status="200"} 5.0' in txt
        assert 'test_requests_total{method="POST",status="500"} 1.0' in txt

    def test_gauge_operations(self):
        """Verify Gauge set, inc, dec, and text formatting."""
        g = Gauge(name="test_disparity_gauge", description="Disparity gauge", label_names=["segment"])
        g.set(12.5, segment="budget")
        g.inc(amount=1.5, segment="budget")
        g.dec(amount=2.0, segment="budget")

        assert g.get(segment="budget") == 12.0

        txt = g.format_prometheus()
        assert '# TYPE test_disparity_gauge gauge' in txt
        assert 'test_disparity_gauge{segment="budget"} 12.0' in txt

    def test_histogram_operations_and_timing_context(self):
        """Verify Histogram bucket observations, sum, count, and TimingContext."""
        h = Histogram(
            name="test_latency_seconds",
            description="Latency histogram",
            label_names=["action"],
            buckets=(0.01, 0.05, 0.1, 0.5),
        )

        h.observe(0.005, action="optimize")
        h.observe(0.04, action="optimize")
        h.observe(0.2, action="optimize")

        txt = h.format_prometheus()
        assert '# TYPE test_latency_seconds histogram' in txt
        assert 'test_latency_seconds_count{action="optimize"} 3' in txt
        assert 'test_latency_seconds_bucket{action="optimize",le="0.01"} 1' in txt
        assert 'test_latency_seconds_bucket{action="optimize",le="0.05"} 2' in txt

        # TimingContext
        with TimingContext(h, action="optimize"):
            _ = sum(i * i for i in range(1000))

        txt_after = h.format_prometheus()
        assert 'test_latency_seconds_count{action="optimize"} 4' in txt_after

    def test_metrics_registry_singleton_and_export(self):
        """Verify global MetricsRegistry exports all pricing and drift metrics."""
        reg = get_metrics_registry()
        reg.pricing_requests_total.inc(endpoint="/api/v1/pricing/optimize", status="200", role="analyst")
        reg.feature_drift_psi.set(0.042, feature_name="price")
        reg.audit_chain_valid.set(1.0)

        output = reg.generate_prometheus_text()
        assert "pricing_requests_total" in output
        assert "feature_drift_psi" in output
        assert "audit_chain_valid" in output
        assert 'feature_name="price"' in output
