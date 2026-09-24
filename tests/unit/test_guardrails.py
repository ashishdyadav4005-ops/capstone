"""Unit tests for Guardrails Engine."""

from __future__ import annotations

from src.pricing.guardrails import GuardrailsEngine


class TestGuardrailsEngine:
    """Test suite for GuardrailsEngine policy validator."""

    def test_margin_floor_violation_caught(self) -> None:
        """Verify that prices below cost + 15% margin floor are flagged as CRITICAL violations."""
        engine = GuardrailsEngine(min_margin_pct_over_cost=0.15, absolute_min_margin_currency=2.0)

        # Cost = $60, required price >= max(60 * 1.15 = 69.0, 62.0) = $69.0
        # Recommended price = $65.0 -> Violation
        violations = engine.validate_item(
            item_id="PROD_001_budget",
            recommended_price=65.0,
            base_price=100.0,
            unit_cost=60.0,
        )

        assert len(violations) >= 1
        margin_v = next(v for v in violations if v.rule_name == "PROFIT_MARGIN_FLOOR")
        assert margin_v.severity == "CRITICAL"
        assert margin_v.current_value == 65.0
        assert margin_v.allowed_threshold == 69.0

    def test_volatility_violation_caught(self) -> None:
        """Verify that prices outside +/- 15% of baseline P0 are flagged as CRITICAL violations."""
        engine = GuardrailsEngine(max_price_change_pct=0.15)

        # Baseline = $100. Bounds = [$85, $115]
        # Test upper violation ($125)
        v_high = engine.validate_item(
            item_id="PROD_001_high",
            recommended_price=125.0,
            base_price=100.0,
            unit_cost=50.0,
        )
        assert any(v.rule_name == "PRICE_VOLATILITY_LIMIT" for v in v_high)

        # Test lower violation ($75)
        v_low = engine.validate_item(
            item_id="PROD_001_low",
            recommended_price=75.0,
            base_price=100.0,
            unit_cost=50.0,
        )
        assert any(v.rule_name == "PRICE_VOLATILITY_LIMIT" for v in v_low)

    def test_capacity_ceiling_violation_caught(self) -> None:
        """Verify that demand exceeding 98% capacity threshold triggers a WARNING violation."""
        engine = GuardrailsEngine(max_capacity_utilization=0.98)

        # Capacity = 100, max allowed = 98
        # Expected demand = 105 -> Violation
        violations = engine.validate_item(
            item_id="PROD_001_cap",
            recommended_price=100.0,
            base_price=100.0,
            unit_cost=50.0,
            expected_demand=105.0,
            capacity=100.0,
        )

        assert len(violations) == 1
        cap_v = violations[0]
        assert cap_v.rule_name == "CAPACITY_CEILING"
        assert cap_v.severity == "WARNING"

    def test_customer_group_disparity_violation(self) -> None:
        """Verify that customer group price disparity exceeding 12% is caught in batch validation."""
        engine = GuardrailsEngine(max_customer_group_disparity_pct=0.12)

        # Product PROD_001 at LOC_URBAN: budget=$90, premium=$110 -> (110-90)/90 = 22.2% > 12%
        records = [
            {
                "item_id": "PROD_001_budget_URBAN",
                "product_id": "PROD_001",
                "customer_group": "budget",
                "location_id": "LOC_URBAN",
                "recommended_price": 90.0,
                "base_price": 100.0,
                "unit_cost": 50.0,
            },
            {
                "item_id": "PROD_001_premium_URBAN",
                "product_id": "PROD_001",
                "customer_group": "premium",
                "location_id": "LOC_URBAN",
                "recommended_price": 110.0,
                "base_price": 100.0,
                "unit_cost": 50.0,
            },
        ]

        res = engine.validate_batch(records)
        assert not res.is_compliant
        assert any(v.rule_name == "CUSTOMER_GROUP_FAIRNESS" for v in res.violations)
        assert len(res.remediation_suggestions) >= 1

    def test_geographic_location_disparity_violation(self) -> None:
        """Verify that geographic location price disparity exceeding 10% is caught."""
        engine = GuardrailsEngine(max_location_disparity_pct=0.10)

        # Product PROD_001 for regular group: URBAN=$95, RURAL=$110 -> (110-95)/95 = 15.8% > 10%
        records = [
            {
                "item_id": "PROD_001_regular_URBAN",
                "product_id": "PROD_001",
                "customer_group": "regular",
                "location_id": "LOC_URBAN",
                "recommended_price": 95.0,
                "base_price": 100.0,
                "unit_cost": 50.0,
            },
            {
                "item_id": "PROD_001_regular_RURAL",
                "product_id": "PROD_001",
                "customer_group": "regular",
                "location_id": "LOC_RURAL",
                "recommended_price": 110.0,
                "base_price": 100.0,
                "unit_cost": 50.0,
            },
        ]

        res = engine.validate_batch(records)
        assert not res.is_compliant
        assert any(v.rule_name == "GEOGRAPHIC_LOCATION_FAIRNESS" for v in res.violations)

    def test_fully_compliant_batch_passes(self) -> None:
        """Verify that a compliant set of prices passes validation with zero violations."""
        engine = GuardrailsEngine(
            max_price_change_pct=0.15,
            min_margin_pct_over_cost=0.15,
            max_customer_group_disparity_pct=0.12,
            max_location_disparity_pct=0.10,
        )

        records = [
            {
                "item_id": "PROD_001_budget_URBAN",
                "product_id": "PROD_001",
                "customer_group": "budget",
                "location_id": "LOC_URBAN",
                "recommended_price": 98.0,
                "base_price": 100.0,
                "unit_cost": 50.0,
                "expected_demand": 60.0,
                "capacity": 100.0,
            },
            {
                "item_id": "PROD_001_regular_URBAN",
                "product_id": "PROD_001",
                "customer_group": "regular",
                "location_id": "LOC_URBAN",
                "recommended_price": 102.0,
                "base_price": 100.0,
                "unit_cost": 50.0,
                "expected_demand": 50.0,
                "capacity": 100.0,
            },
        ]

        res = engine.validate_batch(records)
        assert res.is_compliant
        assert res.total_violations == 0
        assert res.critical_violations == 0
