"""Prometheus metrics collector and exposition exporter for BDS-39 Dynamic Pricing Engine."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from typing import Any

from src.common.logger import get_logger

logger = get_logger("metrics")

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class Counter:
    """Thread-safe Prometheus Counter metric."""

    def __init__(self, name: str, description: str, label_names: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = tuple(label_names or [])
        self._values: dict[tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        """Increment the counter by amount."""
        if amount < 0:
            raise ValueError("Counters can only be incremented by non-negative amounts.")
        key = tuple(str(labels.get(lbl, "")) for lbl in self.label_names)
        with self._lock:
            self._values[key] += amount

    def get(self, **labels: Any) -> float:
        """Get the current counter value for a given label set."""
        key = tuple(str(labels.get(lbl, "")) for lbl in self.label_names)
        with self._lock:
            return self._values[key]

    def format_prometheus(self) -> str:
        """Format metric into standard Prometheus text representation."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            if not self._values:
                # Default zero value if no labels
                if not self.label_names:
                    lines.append(f"{self.name} 0.0")
                return "\n".join(lines)

            for key, val in sorted(self._values.items()):
                if self.label_names:
                    label_str = ",".join(f'{k}="{v}"' for k, v in zip(self.label_names, key, strict=False))
                    lines.append(f"{self.name}{{{label_str}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return "\n".join(lines)


class Gauge:
    """Thread-safe Prometheus Gauge metric."""

    def __init__(self, name: str, description: str, label_names: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = tuple(label_names or [])
        self._values: dict[tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, **labels: Any) -> None:
        """Set gauge to exact value."""
        key = tuple(str(labels.get(lbl, "")) for lbl in self.label_names)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        """Increment gauge value."""
        key = tuple(str(labels.get(lbl, "")) for lbl in self.label_names)
        with self._lock:
            self._values[key] += amount

    def dec(self, amount: float = 1.0, **labels: Any) -> None:
        """Decrement gauge value."""
        key = tuple(str(labels.get(lbl, "")) for lbl in self.label_names)
        with self._lock:
            self._values[key] -= amount

    def get(self, **labels: Any) -> float:
        """Get current gauge value."""
        key = tuple(str(labels.get(lbl, "")) for lbl in self.label_names)
        with self._lock:
            return self._values[key]

    def format_prometheus(self) -> str:
        """Format gauge into standard Prometheus text representation."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            if not self._values:
                if not self.label_names:
                    lines.append(f"{self.name} 0.0")
                return "\n".join(lines)

            for key, val in sorted(self._values.items()):
                if self.label_names:
                    label_str = ",".join(f'{k}="{v}"' for k, v in zip(self.label_names, key, strict=False))
                    lines.append(f"{self.name}{{{label_str}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return "\n".join(lines)


class Histogram:
    """Thread-safe Prometheus Histogram metric with cumulative bucket distributions."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ):
        self.name = name
        self.description = description
        self.label_names = tuple(label_names or [])
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[tuple[str, ...], dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._sums: dict[tuple[str, ...], float] = defaultdict(float)
        self._total_counts: dict[tuple[str, ...], int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: Any) -> None:
        """Record an observed value in the histogram."""
        v = float(value)
        key = tuple(str(labels.get(lbl, "")) for lbl in self.label_names)
        with self._lock:
            self._sums[key] += v
            self._total_counts[key] += 1
            for b in self.buckets:
                if v <= b:
                    self._counts[key][b] += 1

    def format_prometheus(self) -> str:
        """Format histogram into standard Prometheus text representation."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            if not self._total_counts:
                return "\n".join(lines)

            for key in sorted(self._total_counts.keys()):
                base_labels = [f'{k}="{v}"' for k, v in zip(self.label_names, key, strict=False)]
                cum_count = 0
                for b in self.buckets:
                    b_count = self._counts[key][b]
                    cum_count = b_count
                    b_str = "+Inf" if math.isinf(b) else str(b)
                    lbl_items = base_labels + [f'le="{b_str}"']
                    lbl_str = "{" + ",".join(lbl_items) + "}"
                    lines.append(f"{self.name}_bucket{lbl_str} {cum_count}")

                # +Inf bucket
                lbl_inf = "{" + ",".join(base_labels + ['le="+Inf"']) + "}"
                lines.append(f"{self.name}_bucket{lbl_inf} {self._total_counts[key]}")

                # Sum and count
                lbl_base = ("{" + ",".join(base_labels) + "}") if base_labels else ""
                lines.append(f"{self.name}_sum{lbl_base} {self._sums[key]}")
                lines.append(f"{self.name}_count{lbl_base} {self._total_counts[key]}")

        return "\n".join(lines)


class MetricsRegistry:
    """Singleton container managing all system observability metrics."""

    def __init__(self):
        # 1. Counters
        self.pricing_requests_total = Counter(
            name="pricing_requests_total",
            description="Total HTTP requests processed by the pricing engine.",
            label_names=["endpoint", "status", "role"],
        )
        self.pricing_recommendations_total = Counter(
            name="pricing_recommendations_total",
            description="Total dynamic pricing recommendations generated.",
            label_names=["product_id", "customer_group", "status"],
        )
        self.pricing_overrides_total = Counter(
            name="pricing_overrides_total",
            description="Total human-in-the-loop price overrides submitted.",
            label_names=["actor", "product_id"],
        )
        self.pricing_approvals_total = Counter(
            name="pricing_approvals_total",
            description="Total price recommendations approved.",
            label_names=["actor"],
        )
        self.pricing_rollbacks_total = Counter(
            name="pricing_rollbacks_total",
            description="Total emergency price rollbacks executed.",
            label_names=["product_id"],
        )
        self.guardrail_violations_total = Counter(
            name="guardrail_violations_total",
            description="Total guardrail threshold violations triggered.",
            label_names=["guardrail_type", "severity"],
        )

        # 2. Histograms
        self.optimizer_solve_duration_seconds = Histogram(
            name="optimizer_solve_duration_seconds",
            description="Execution latency of SciPy SLSQP non-linear pricing optimizer.",
            label_names=["objective"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )
        self.demand_curve_duration_seconds = Histogram(
            name="demand_curve_duration_seconds",
            description="Latency for counterfactual demand curve generation.",
            label_names=["product_id"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
        )
        self.profit_lift_percentage = Histogram(
            name="profit_lift_percentage",
            description="Distribution of recommended profit lift percentage over baseline.",
            label_names=["product_id"],
            buckets=(0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0),
        )

        # 3. Gauges
        self.audit_chain_valid = Gauge(
            name="audit_chain_valid",
            description="Live cryptographic SHA-256 chain integrity status (1.0 = valid, 0.0 = broken).",
        )
        self.audit_chain_valid.set(1.0)

        self.max_subgroup_disparity_pct = Gauge(
            name="max_subgroup_disparity_pct",
            description="Maximum observed customer group price disparity percentage.",
        )

        self.feature_drift_psi = Gauge(
            name="feature_drift_psi",
            description="Latest Population Stability Index (PSI) score for target features.",
            label_names=["feature_name"],
        )

        self.active_locked_accounts = Gauge(
            name="active_locked_accounts",
            description="Current count of accounts locked due to brute-force protection.",
        )

    def generate_prometheus_text(self) -> str:
        """Export all registered metrics in standard Prometheus exposition format."""
        metrics = [
            self.pricing_requests_total,
            self.pricing_recommendations_total,
            self.pricing_overrides_total,
            self.pricing_approvals_total,
            self.pricing_rollbacks_total,
            self.guardrail_violations_total,
            self.optimizer_solve_duration_seconds,
            self.demand_curve_duration_seconds,
            self.profit_lift_percentage,
            self.audit_chain_valid,
            self.max_subgroup_disparity_pct,
            self.feature_drift_psi,
            self.active_locked_accounts,
        ]
        blocks = [m.format_prometheus() for m in metrics]
        return "\n\n".join(b for b in blocks if b) + "\n"


_registry_instance: MetricsRegistry | None = None
_registry_lock = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    """Return singleton instance of MetricsRegistry."""
    global _registry_instance
    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = MetricsRegistry()
        return _registry_instance


class TimingContext:
    """Context manager for measuring and recording code execution latency into a histogram."""

    def __init__(self, histogram: Histogram, **labels: Any):
        self.histogram = histogram
        self.labels = labels
        self.start_time = 0.0

    def __enter__(self) -> TimingContext:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration = time.perf_counter() - self.start_time
        self.histogram.observe(duration, **self.labels)
