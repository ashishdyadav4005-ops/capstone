"""Unit tests for Governance Reports: Subgroup fairness and Model Card metadata."""

from __future__ import annotations

from src.api.routers.governance import (
    FairnessAuditResponse,
    FairnessProductReport,
    ModelCardResponse,
    get_fairness_report,
    get_model_card,
)
from src.auth.repository import UserRecord
from src.auth.roles import UserRole
from src.pricing.curves import DemandCurveGenerator
from src.pricing.simulator import PricingScenarioSimulator


def test_fairness_report_generation():
    """Verify fairness audit endpoint builds subgroup breakdown within disparity threshold."""
    curve_gen = DemandCurveGenerator()
    simulator = PricingScenarioSimulator(curve_generator=curve_gen)
    mock_user = UserRecord(
        user_id="USR_01",
        username="gov_officer",
        email="gov@example.com",
        full_name="Governance Officer",
        hashed_password="hash",
        role=UserRole.GOVERNANCE,
    )

    report: FairnessAuditResponse = get_fairness_report(simulator=simulator, _user=mock_user)

    assert isinstance(report, FairnessAuditResponse)
    assert report.threshold_pct == 12.0
    assert len(report.products) == 5
    for p in report.products:
        assert isinstance(p, FairnessProductReport)
        assert p.product_id.startswith("PROD_")
        assert len(p.group_breakdown) == 4
        assert p.disparity_threshold_pct == 12.0


def test_model_card_metadata():
    """Verify Model Card returns Robinson DML architecture and sensitivity benchmarks."""
    mock_user = UserRecord(
        user_id="USR_01",
        username="gov_officer",
        email="gov@example.com",
        full_name="Governance Officer",
        hashed_password="hash",
        role=UserRole.GOVERNANCE,
    )

    card: ModelCardResponse = get_model_card(_user=mock_user)

    assert isinstance(card, ModelCardResponse)
    assert "Double Machine Learning" in card.model_name
    assert card.cross_fitting_folds == 5
    assert card.target_treatment == "log(price)"
    assert card.target_outcome == "log(quantity)"
    assert len(card.ground_truth_bias_comparison) >= 3
    assert len(card.sensitivity_stress_tests) >= 4
    assert card.governance_status == "APPROVED_FOR_INTERNAL_DECISION_SUPPORT"
