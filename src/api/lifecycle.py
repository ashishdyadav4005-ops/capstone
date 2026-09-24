"""Recommendation Lifecycle State Machine and Manager Override Engine."""

from __future__ import annotations

from src.api.repository import RecommendationRepository
from src.api.schemas import RecommendationRecord, RecommendationState
from src.audit.logger import AuditLogger
from src.common.logger import get_logger
from src.pricing.guardrails import GuardrailsEngine

logger = get_logger("recommendation_lifecycle")


class IllegalStateTransitionError(ValueError):
    """Raised when an illegal lifecycle state transition is attempted."""
    pass


class RecommendationLifecycleManager:
    """Controls price recommendation state transitions, approvals, manager overrides, and rollbacks."""

    VALID_TRANSITIONS: dict[RecommendationState, set[RecommendationState]] = {
        RecommendationState.GENERATED: {
            RecommendationState.UNDER_REVIEW,
            RecommendationState.APPROVED,
            RecommendationState.REJECTED,
        },
        RecommendationState.UNDER_REVIEW: {
            RecommendationState.APPROVED,
            RecommendationState.REJECTED,
        },
        RecommendationState.APPROVED: {
            RecommendationState.PUBLISHED,
            RecommendationState.REJECTED,
            RecommendationState.UNDER_REVIEW,
        },
        RecommendationState.PUBLISHED: {
            RecommendationState.ROLLED_BACK,
        },
        RecommendationState.REJECTED: {
            RecommendationState.UNDER_REVIEW,
        },
        RecommendationState.ROLLED_BACK: {
            RecommendationState.UNDER_REVIEW,
        },
    }

    def __init__(
        self,
        repository: RecommendationRepository | None = None,
        guardrails_engine: GuardrailsEngine | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.repository = repository or RecommendationRepository()
        self.guardrails_engine = guardrails_engine or GuardrailsEngine()
        self.audit_logger = audit_logger or AuditLogger()

    def transition(
        self,
        recommendation_id: str,
        target_state: RecommendationState,
        actor: str = "system",
        notes: str | None = None,
        reason: str | None = None,
    ) -> RecommendationRecord:
        """Execute a state transition for a recommendation."""
        rec = self.repository.get_by_id(recommendation_id)
        if rec is None:
            raise KeyError(f"Recommendation with ID '{recommendation_id}' not found.")

        current_state = rec.state
        allowed = self.VALID_TRANSITIONS.get(current_state, set())

        if target_state not in allowed:
            raise IllegalStateTransitionError(
                f"Illegal state transition from '{current_state.value}' to '{target_state.value}'. "
                f"Allowed transitions: {[s.value for s in allowed]}"
            )

        if target_state == RecommendationState.REJECTED and not (reason or notes):
            raise ValueError("Rejection requires a mandatory explanation note or reason.")

        now = RecommendationRepository.now_iso()
        audit_entry = f"[{now}] State changed {current_state.value} -> {target_state.value} by {actor}"
        if notes:
            audit_entry += f" | Notes: {notes}"
        if reason:
            audit_entry += f" | Reason: {reason}"

        combined_audit = f"{rec.audit_notes}\n{audit_entry}" if rec.audit_notes else audit_entry

        updated_rec = RecommendationRecord(
            recommendation_id=rec.recommendation_id,
            item_id=rec.item_id,
            product_id=rec.product_id,
            customer_group=rec.customer_group,
            location_id=rec.location_id,
            base_price=rec.base_price,
            recommended_price=rec.recommended_price,
            final_price=rec.final_price,
            unit_cost=rec.unit_cost,
            expected_demand=rec.expected_demand,
            expected_revenue=rec.expected_revenue,
            expected_profit=rec.expected_profit,
            margin_pct=rec.margin_pct,
            state=target_state,
            created_by=rec.created_by,
            reviewed_by=actor if target_state in [RecommendationState.APPROVED, RecommendationState.REJECTED, RecommendationState.UNDER_REVIEW] else rec.reviewed_by,
            override_price=rec.override_price,
            override_reason=rec.override_reason,
            audit_notes=combined_audit,
            created_at=rec.created_at,
            updated_at=now,
        )

        self.repository.save(updated_rec)
        logger.info(
            "Recommendation state updated",
            extra={
                "recommendation_id": recommendation_id,
                "from_state": current_state.value,
                "to_state": target_state.value,
                "actor": actor,
            },
        )

        # Append to cryptographic audit trail
        try:
            if target_state == RecommendationState.APPROVED:
                self.audit_logger.log_price_approved(
                    recommendation_id=updated_rec.recommendation_id,
                    item_id=updated_rec.item_id,
                    final_price=updated_rec.final_price,
                    actor_username=actor,
                    actor_role="manager",
                    notes=notes,
                )
            elif target_state == RecommendationState.REJECTED:
                self.audit_logger.log_price_rejected(
                    recommendation_id=updated_rec.recommendation_id,
                    item_id=updated_rec.item_id,
                    reason=reason or notes or "Manager rejection",
                    actor_username=actor,
                    actor_role="manager",
                )
            elif target_state == RecommendationState.PUBLISHED:
                self.audit_logger.log_price_published(
                    recommendation_id=updated_rec.recommendation_id,
                    item_id=updated_rec.item_id,
                    published_price=updated_rec.final_price,
                    actor_username=actor,
                    actor_role="system",
                )
        except Exception as audit_err:
            logger.warning(f"Failed to record audit trail for transition: {audit_err}")

        return updated_rec

    def override_price(
        self,
        recommendation_id: str,
        override_price: float,
        reason: str,
        actor: str = "manager",
        strict_guardrails: bool = True,
    ) -> RecommendationRecord:
        """Apply a human-in-the-loop manager price override with guardrail safety checks."""
        rec = self.repository.get_by_id(recommendation_id)
        if rec is None:
            raise KeyError(f"Recommendation with ID '{recommendation_id}' not found.")

        if override_price <= 0:
            raise ValueError(f"Override price must be strictly positive, got {override_price}")

        if not reason or len(reason.strip()) < 5:
            raise ValueError("A clear, justifiable reason (>= 5 chars) is mandatory for price overrides.")

        # Guardrail check on overridden price
        violations = self.guardrails_engine.validate_item(
            item_id=rec.item_id,
            recommended_price=override_price,
            base_price=rec.base_price,
            unit_cost=rec.unit_cost,
        )

        critical_violations = [v for v in violations if v.severity == "CRITICAL"]
        if strict_guardrails and critical_violations:
            first_v = critical_violations[0]
            raise ValueError(f"Price override ${override_price:.2f} rejected by guardrail: {first_v.violation_message}")

        # Recalculate economic expected demand and profit at override price
        # Elasticity proxy derived from recommended demand vs base demand
        price_ratio = override_price / rec.base_price
        # Using standard elasticity from base
        exp_q = float(max(0.0, rec.expected_demand * (price_ratio ** -1.5)))
        exp_rev = float(override_price * exp_q)
        exp_prof = float((override_price - rec.unit_cost) * exp_q)
        margin_pct = float(((override_price - rec.unit_cost) / override_price * 100.0) if override_price > 0 else 0.0)

        now = RecommendationRepository.now_iso()
        audit_entry = (
            f"[{now}] Price overridden by {actor}: ${rec.final_price:.2f} -> ${override_price:.2f} | Reason: {reason}"
        )
        combined_audit = f"{rec.audit_notes}\n{audit_entry}" if rec.audit_notes else audit_entry

        updated_rec = RecommendationRecord(
            recommendation_id=rec.recommendation_id,
            item_id=rec.item_id,
            product_id=rec.product_id,
            customer_group=rec.customer_group,
            location_id=rec.location_id,
            base_price=rec.base_price,
            recommended_price=rec.recommended_price,
            final_price=override_price,
            unit_cost=rec.unit_cost,
            expected_demand=float(round(exp_q, 2)),
            expected_revenue=float(round(exp_rev, 2)),
            expected_profit=float(round(exp_prof, 2)),
            margin_pct=float(round(margin_pct, 2)),
            state=RecommendationState.UNDER_REVIEW,
            created_by=rec.created_by,
            reviewed_by=actor,
            override_price=override_price,
            override_reason=reason,
            audit_notes=combined_audit,
            created_at=rec.created_at,
            updated_at=now,
        )

        self.repository.save(updated_rec)
        logger.info(
            "Price override applied",
            extra={
                "recommendation_id": recommendation_id,
                "original_price": rec.final_price,
                "override_price": override_price,
                "actor": actor,
            },
        )

        try:
            self.audit_logger.log_price_override(
                recommendation_id=updated_rec.recommendation_id,
                item_id=updated_rec.item_id,
                original_price=rec.final_price,
                override_price=override_price,
                reason=reason,
                actor_username=actor,
                actor_role="manager",
            )
        except Exception as audit_err:
            logger.warning(f"Failed to record audit trail for price override: {audit_err}")

        return updated_rec

    def rollback(
        self,
        recommendation_id: str,
        actor: str = "manager",
        notes: str | None = "Emergency rollback to baseline",
    ) -> RecommendationRecord:
        """Emergency rollback of a published price back to baseline price P0."""
        rec = self.repository.get_by_id(recommendation_id)
        if rec is None:
            raise KeyError(f"Recommendation with ID '{recommendation_id}' not found.")

        now = RecommendationRepository.now_iso()
        audit_entry = f"[{now}] EMERGENCY ROLLBACK executed by {actor}: ${rec.final_price:.2f} -> ${rec.base_price:.2f} | Notes: {notes}"
        combined_audit = f"{rec.audit_notes}\n{audit_entry}" if rec.audit_notes else audit_entry

        updated_rec = RecommendationRecord(
            recommendation_id=rec.recommendation_id,
            item_id=rec.item_id,
            product_id=rec.product_id,
            customer_group=rec.customer_group,
            location_id=rec.location_id,
            base_price=rec.base_price,
            recommended_price=rec.recommended_price,
            final_price=rec.base_price,
            unit_cost=rec.unit_cost,
            expected_demand=rec.expected_demand,
            expected_revenue=float(rec.base_price * rec.expected_demand),
            expected_profit=float((rec.base_price - rec.unit_cost) * rec.expected_demand),
            margin_pct=float(((rec.base_price - rec.unit_cost) / rec.base_price * 100.0) if rec.base_price > 0 else 0.0),
            state=RecommendationState.ROLLED_BACK,
            created_by=rec.created_by,
            reviewed_by=actor,
            override_price=rec.override_price,
            override_reason=rec.override_reason,
            audit_notes=combined_audit,
            created_at=rec.created_at,
            updated_at=now,
        )

        self.repository.save(updated_rec)
        logger.warning(
            "Price rolled back to baseline",
            extra={
                "recommendation_id": recommendation_id,
                "restored_price": rec.base_price,
                "actor": actor,
            },
        )

        try:
            self.audit_logger.log_emergency_rollback(
                recommendation_id=updated_rec.recommendation_id,
                item_id=updated_rec.item_id,
                rolled_back_price=rec.final_price,
                restored_base_price=rec.base_price,
                actor_username=actor,
                actor_role="manager",
                notes=notes,
            )
        except Exception as audit_err:
            logger.warning(f"Failed to record audit trail for emergency rollback: {audit_err}")

        return updated_rec
