"""Unit tests for Recommendation Lifecycle State Machine and Manager Override Engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.lifecycle import (
    IllegalStateTransitionError,
    RecommendationLifecycleManager,
)
from src.api.repository import RecommendationRepository
from src.api.schemas import RecommendationRecord, RecommendationState
from src.pricing.guardrails import GuardrailsEngine


@pytest.fixture
def temp_repo(tmp_path: Path) -> RecommendationRepository:
    """Create a temporary SQLite recommendation repository."""
    db_file = tmp_path / "test_lifecycle.db"
    return RecommendationRepository(db_path=db_file)


@pytest.fixture
def sample_record(temp_repo: RecommendationRepository) -> RecommendationRecord:
    """Create and persist a sample recommendation in GENERATED state."""
    rec = RecommendationRecord(
        recommendation_id="REC_TEST_001",
        item_id="PROD_001_regular_URBAN",
        product_id="PROD_001",
        customer_group="regular",
        location_id="LOC_URBAN",
        base_price=100.0,
        recommended_price=115.0,
        final_price=115.0,
        unit_cost=60.0,
        expected_demand=45.0,
        expected_revenue=5175.0,
        expected_profit=2475.0,
        margin_pct=47.83,
        state=RecommendationState.GENERATED,
        created_by="OPTIMIZER",
        created_at=RecommendationRepository.now_iso(),
        updated_at=RecommendationRepository.now_iso(),
    )
    temp_repo.save(rec)
    return rec


class TestRecommendationLifecycle:
    """Test suite for recommendation state transitions and overrides."""

    def test_valid_lifecycle_happy_path(
        self,
        temp_repo: RecommendationRepository,
        sample_record: RecommendationRecord,
    ) -> None:
        """Test complete progression: GENERATED -> UNDER_REVIEW -> APPROVED -> PUBLISHED."""
        lifecycle = RecommendationLifecycleManager(repository=temp_repo)
        rec_id = sample_record.recommendation_id

        # 1. GENERATED -> UNDER_REVIEW
        r1 = lifecycle.transition(rec_id, RecommendationState.UNDER_REVIEW, actor="analyst_1", notes="Reviewing elasticity")
        assert r1.state == RecommendationState.UNDER_REVIEW
        assert "analyst_1" in str(r1.audit_notes)

        # 2. UNDER_REVIEW -> APPROVED
        r2 = lifecycle.transition(rec_id, RecommendationState.APPROVED, actor="manager_1", notes="Approved for Q3")
        assert r2.state == RecommendationState.APPROVED
        assert r2.reviewed_by == "manager_1"

        # 3. APPROVED -> PUBLISHED
        r3 = lifecycle.transition(rec_id, RecommendationState.PUBLISHED, actor="system", notes="Pushed to production")
        assert r3.state == RecommendationState.PUBLISHED

    def test_illegal_state_transition_raises_error(
        self,
        temp_repo: RecommendationRepository,
        sample_record: RecommendationRecord,
    ) -> None:
        """Test that illegal transitions (e.g. GENERATED -> PUBLISHED directly) are blocked."""
        lifecycle = RecommendationLifecycleManager(repository=temp_repo)
        rec_id = sample_record.recommendation_id

        # Directly jumping to PUBLISHED from GENERATED is illegal
        with pytest.raises(IllegalStateTransitionError, match="Illegal state transition"):
            lifecycle.transition(rec_id, RecommendationState.PUBLISHED, actor="user")

    def test_rejection_requires_mandatory_reason(
        self,
        temp_repo: RecommendationRepository,
        sample_record: RecommendationRecord,
    ) -> None:
        """Verify that rejecting a price requires a mandatory reason."""
        lifecycle = RecommendationLifecycleManager(repository=temp_repo)
        rec_id = sample_record.recommendation_id

        # Rejection with no reason/notes raises ValueError
        with pytest.raises(ValueError, match="mandatory explanation"):
            lifecycle.transition(rec_id, RecommendationState.REJECTED, actor="manager_1")

        # Valid rejection
        r = lifecycle.transition(rec_id, RecommendationState.REJECTED, actor="manager_1", reason="Competitor launched promo")
        assert r.state == RecommendationState.REJECTED

    def test_manager_price_override_with_guardrail_check(
        self,
        temp_repo: RecommendationRepository,
        sample_record: RecommendationRecord,
    ) -> None:
        """Verify that manager override updates final price and respects guardrails."""
        guardrails = GuardrailsEngine(max_price_change_pct=0.15, min_margin_pct_over_cost=0.15)
        lifecycle = RecommendationLifecycleManager(repository=temp_repo, guardrails_engine=guardrails)
        rec_id = sample_record.recommendation_id

        # 1. Valid override ($112.0 within [85, 115] and > 69.0)
        r = lifecycle.override_price(
            recommendation_id=rec_id,
            override_price=112.0,
            reason="Market intelligence suggests 112 is optimal",
            actor="senior_manager",
        )
        assert r.final_price == 112.0
        assert r.override_price == 112.0
        assert r.state == RecommendationState.UNDER_REVIEW

        # 2. Invalid override violating volatility (e.g. $130 > 115)
        with pytest.raises(ValueError, match="rejected by guardrail"):
            lifecycle.override_price(
                recommendation_id=rec_id,
                override_price=130.0,
                reason="Aggressive push",
                actor="senior_manager",
            )

        # 3. Invalid override violating margin floor (e.g. $65 < 69)
        with pytest.raises(ValueError, match="rejected by guardrail"):
            lifecycle.override_price(
                recommendation_id=rec_id,
                override_price=65.0,
                reason="Loss leader attempt",
                actor="senior_manager",
            )

    def test_emergency_rollback_to_baseline(
        self,
        temp_repo: RecommendationRepository,
        sample_record: RecommendationRecord,
    ) -> None:
        """Verify emergency rollback restores original base price P0 and transitions to ROLLED_BACK."""
        lifecycle = RecommendationLifecycleManager(repository=temp_repo)
        rec_id = sample_record.recommendation_id

        # Fast forward to PUBLISHED
        lifecycle.transition(rec_id, RecommendationState.APPROVED, actor="manager")
        lifecycle.transition(rec_id, RecommendationState.PUBLISHED, actor="system")

        # Rollback
        r = lifecycle.rollback(rec_id, actor="ops_lead", notes="Customer backlash on dynamic adjustment")
        assert r.state == RecommendationState.ROLLED_BACK
        assert r.final_price == r.base_price == 100.0
