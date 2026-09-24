"""Guardrails Engine: Multi-dimensional validation for dynamic pricing decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailViolation:
    """Detailed record of a specific guardrail boundary violation."""

    rule_name: str
    item_identifier: str
    violation_message: str
    current_value: float
    allowed_threshold: float
    severity: str = "CRITICAL" # "CRITICAL" or "WARNING"


@dataclass
class GuardrailCheckResult:
    """Overall outcome of guardrail policy checks."""

    is_compliant: bool
    violations: list[GuardrailViolation] = field(default_factory=list)
    remediation_suggestions: list[str] = field(default_factory=list)

    @property
    def total_violations(self) -> int:
        """Total count of violations."""
        return len(self.violations)

    @property
    def critical_violations(self) -> int:
        """Count of critical violations."""
        return sum(1 for v in self.violations if v.severity == "CRITICAL")


class GuardrailsEngine:
    """Validates dynamic price proposals against business, safety, and fairness guardrails."""

    def __init__(
        self,
        max_price_change_pct: float = 0.15, # Max 15% swing relative to baseline P0
        min_margin_pct_over_cost: float = 0.15, # Min 15% markup over cost
        absolute_min_margin_currency: float = 2.0, # Minimum absolute $ margin
        max_customer_group_disparity_pct: float = 0.12, # Max 12% price gap across customer groups
        max_location_disparity_pct: float = 0.10, # Max 10% price gap across locations
        max_capacity_utilization: float = 0.98, # Max 98% capacity threshold
    ):
        self.max_price_change_pct = max_price_change_pct
        self.min_margin_pct_over_cost = min_margin_pct_over_cost
        self.absolute_min_margin_currency = absolute_min_margin_currency
        self.max_customer_group_disparity_pct = max_customer_group_disparity_pct
        self.max_location_disparity_pct = max_location_disparity_pct
        self.max_capacity_utilization = max_capacity_utilization

    def validate_item(
        self,
        item_id: str,
        recommended_price: float,
        base_price: float,
        unit_cost: float,
        expected_demand: float | None = None,
        capacity: float | None = None,
    ) -> list[GuardrailViolation]:
        """Validate single product-segment pricing against volatility, margin, and capacity rules."""
        violations: list[GuardrailViolation] = []

        # 1. Profit Margin Floor Check
        min_margin_p = max(
            unit_cost * (1.0 + self.min_margin_pct_over_cost),
            unit_cost + self.absolute_min_margin_currency,
        )
        if recommended_price < (min_margin_p - 1e-4):
            violations.append(
                GuardrailViolation(
                    rule_name="PROFIT_MARGIN_FLOOR",
                    item_identifier=item_id,
                    violation_message=(
                        f"Recommended price ${recommended_price:.2f} is below minimum margin floor "
                        f"${min_margin_p:.2f} (Cost: ${unit_cost:.2f})"
                    ),
                    current_value=recommended_price,
                    allowed_threshold=min_margin_p,
                    severity="CRITICAL",
                )
            )

        # 2. Price Volatility Check
        vol_min = base_price * (1.0 - self.max_price_change_pct)
        vol_max = base_price * (1.0 + self.max_price_change_pct)

        if recommended_price < (vol_min - 1e-4) or recommended_price > (vol_max + 1e-4):
            chg_pct = (recommended_price - base_price) / base_price * 100.0
            violations.append(
                GuardrailViolation(
                    rule_name="PRICE_VOLATILITY_LIMIT",
                    item_identifier=item_id,
                    violation_message=(
                        f"Price change of {chg_pct:+.1f}% exceeds max volatility limit "
                        f"+/-{self.max_price_change_pct * 100.0:.1f}% (Range: [${vol_min:.2f}, ${vol_max:.2f}])"
                    ),
                    current_value=recommended_price,
                    allowed_threshold=vol_max if recommended_price > vol_max else vol_min,
                    severity="CRITICAL",
                )
            )

        # 3. Capacity Ceiling Check
        if capacity is not None and expected_demand is not None:
            max_allowed_demand = capacity * self.max_capacity_utilization
            if expected_demand > (max_allowed_demand + 1e-4):
                violations.append(
                    GuardrailViolation(
                        rule_name="CAPACITY_CEILING",
                        item_identifier=item_id,
                        violation_message=(
                            f"Expected demand {expected_demand:.1f} exceeds maximum allowed capacity "
                            f"{max_allowed_demand:.1f} ({self.max_capacity_utilization * 100:.0f}% of {capacity:.0f})"
                        ),
                        current_value=expected_demand,
                        allowed_threshold=max_allowed_demand,
                        severity="WARNING",
                    )
                )

        return violations

    def validate_batch(
        self,
        pricing_records: list[dict[str, Any]],
    ) -> GuardrailCheckResult:
        """Perform comprehensive batch validation including group and geographic fairness rules.

        Each record in `pricing_records` should contain:
        - `item_id`: str
        - `product_id`: str
        - `customer_group`: str
        - `location_id`: str
        - `recommended_price`: float
        - `base_price`: float
        - `unit_cost`: float
        - `expected_demand`: float (optional)
        - `capacity`: float (optional)
        """
        all_violations: list[GuardrailViolation] = []
        remediations: list[str] = []

        # 1. Item-level checks
        for rec in pricing_records:
            item_violations = self.validate_item(
                item_id=rec.get("item_id", f"{rec.get('product_id')}_{rec.get('customer_group')}"),
                recommended_price=float(rec["recommended_price"]),
                base_price=float(rec["base_price"]),
                unit_cost=float(rec["unit_cost"]),
                expected_demand=rec.get("expected_demand"),
                capacity=rec.get("capacity"),
            )
            all_violations.extend(item_violations)

        # 2. Customer Group Fairness Disparity Check (for same product and location)
        prod_loc_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for rec in pricing_records:
            key = (rec["product_id"], rec.get("location_id", "GLOBAL"))
            prod_loc_groups.setdefault(key, []).append(rec)

        for (prod_id, loc_id), group_recs in prod_loc_groups.items():
            if len(group_recs) > 1:
                prices = [float(r["recommended_price"]) for r in group_recs]
                min_p, max_p = min(prices), max(prices)
                if min_p > 0:
                    disparity = (max_p - min_p) / min_p
                    if disparity > (self.max_customer_group_disparity_pct + 1e-4):
                        all_violations.append(
                            GuardrailViolation(
                                rule_name="CUSTOMER_GROUP_FAIRNESS",
                                item_identifier=f"{prod_id}@{loc_id}",
                                violation_message=(
                                    f"Customer group price disparity of {disparity * 100:.1f}% exceeds threshold "
                                    f"{self.max_customer_group_disparity_pct * 100:.1f}% (Min: ${min_p:.2f}, Max: ${max_p:.2f})"
                                ),
                                current_value=disparity,
                                allowed_threshold=self.max_customer_group_disparity_pct,
                                severity="CRITICAL",
                            )
                        )
                        remediations.append(
                            f"Compress price spread for product {prod_id} at location {loc_id} below "
                            f"{self.max_customer_group_disparity_pct * 100:.1f}%"
                        )

        # 3. Geographic Location Fairness Disparity Check (for same product and customer group)
        prod_grp_locations: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for rec in pricing_records:
            key = (rec["product_id"], rec.get("customer_group", "GLOBAL"))
            prod_grp_locations.setdefault(key, []).append(rec)

        for (prod_id, grp_id), loc_recs in prod_grp_locations.items():
            if len(loc_recs) > 1:
                prices = [float(r["recommended_price"]) for r in loc_recs]
                min_p, max_p = min(prices), max(prices)
                if min_p > 0:
                    disparity = (max_p - min_p) / min_p
                    if disparity > (self.max_location_disparity_pct + 1e-4):
                        all_violations.append(
                            GuardrailViolation(
                                rule_name="GEOGRAPHIC_LOCATION_FAIRNESS",
                                item_identifier=f"{prod_id}@{grp_id}",
                                violation_message=(
                                    f"Geographic price disparity of {disparity * 100:.1f}% exceeds threshold "
                                    f"{self.max_location_disparity_pct * 100:.1f}% (Min: ${min_p:.2f}, Max: ${max_p:.2f})"
                                ),
                                current_value=disparity,
                                allowed_threshold=self.max_location_disparity_pct,
                                severity="CRITICAL",
                            )
                        )
                        remediations.append(
                            f"Harmonize regional pricing for product {prod_id} (segment {grp_id}) within "
                            f"{self.max_location_disparity_pct * 100:.1f}%"
                        )

        is_compliant = len(all_violations) == 0
        return GuardrailCheckResult(
            is_compliant=is_compliant,
            violations=all_violations,
            remediation_suggestions=remediations,
        )
