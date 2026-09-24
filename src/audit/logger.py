"""Centralized AuditLogger service for system-wide tamper-evident event recording."""

from __future__ import annotations

from typing import Any

from src.audit.models import AuditEntry, AuditEventType
from src.audit.repository import AuditRepository


class AuditLogger:
    """High-level service interface for recording immutable audit trail events."""

    def __init__(self, repository: AuditRepository | None = None):
        self.repository = repository or AuditRepository()

    def log_recommendations_generated(
        self,
        count: int,
        actor_username: str = "OPTIMIZER",
        actor_role: str = "system",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record batch dynamic price recommendations generation."""
        action_data = {"count": count}
        if details:
            action_data.update(details)
        return self.repository.append(
            event_type=AuditEventType.PRICE_RECOMMENDATIONS_GENERATED,
            actor_username=actor_username,
            actor_role=actor_role,
            action_details=action_data,
        )

    def log_price_override(
        self,
        recommendation_id: str,
        item_id: str,
        original_price: float,
        override_price: float,
        reason: str,
        actor_username: str,
        actor_role: str = "manager",
    ) -> AuditEntry:
        """Record human manager price override."""
        return self.repository.append(
            event_type=AuditEventType.PRICE_OVERRIDE_APPLIED,
            actor_username=actor_username,
            actor_role=actor_role,
            action_details={
                "recommendation_id": recommendation_id,
                "item_id": item_id,
                "original_price": original_price,
                "override_price": override_price,
                "price_change_pct": round(((override_price - original_price) / original_price * 100.0), 2) if original_price > 0 else 0.0,
                "reason": reason,
            },
        )

    def log_price_approved(
        self,
        recommendation_id: str,
        item_id: str,
        final_price: float,
        actor_username: str,
        actor_role: str = "manager",
        notes: str | None = None,
    ) -> AuditEntry:
        """Record manager approval of a dynamic price recommendation."""
        return self.repository.append(
            event_type=AuditEventType.PRICE_APPROVED,
            actor_username=actor_username,
            actor_role=actor_role,
            action_details={
                "recommendation_id": recommendation_id,
                "item_id": item_id,
                "final_price": final_price,
                "notes": notes,
            },
        )

    def log_price_rejected(
        self,
        recommendation_id: str,
        item_id: str,
        reason: str,
        actor_username: str,
        actor_role: str = "manager",
    ) -> AuditEntry:
        """Record rejection of a dynamic price recommendation."""
        return self.repository.append(
            event_type=AuditEventType.PRICE_REJECTED,
            actor_username=actor_username,
            actor_role=actor_role,
            action_details={
                "recommendation_id": recommendation_id,
                "item_id": item_id,
                "reason": reason,
            },
        )

    def log_price_published(
        self,
        recommendation_id: str,
        item_id: str,
        published_price: float,
        actor_username: str,
        actor_role: str = "system",
    ) -> AuditEntry:
        """Record live publication of a price recommendation."""
        return self.repository.append(
            event_type=AuditEventType.PRICE_PUBLISHED,
            actor_username=actor_username,
            actor_role=actor_role,
            action_details={
                "recommendation_id": recommendation_id,
                "item_id": item_id,
                "published_price": published_price,
            },
        )

    def log_emergency_rollback(
        self,
        recommendation_id: str,
        item_id: str,
        rolled_back_price: float,
        restored_base_price: float,
        actor_username: str,
        actor_role: str = "manager",
        notes: str | None = None,
    ) -> AuditEntry:
        """Record emergency price rollback back to baseline."""
        return self.repository.append(
            event_type=AuditEventType.EMERGENCY_ROLLBACK,
            actor_username=actor_username,
            actor_role=actor_role,
            action_details={
                "recommendation_id": recommendation_id,
                "item_id": item_id,
                "rolled_back_price": rolled_back_price,
                "restored_base_price": restored_base_price,
                "notes": notes,
            },
        )

    def log_security_event(
        self,
        event_type: AuditEventType,
        actor_username: str,
        actor_role: str = "system",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record user login, lockout, or security configuration change."""
        return self.repository.append(
            event_type=event_type,
            actor_username=actor_username,
            actor_role=actor_role,
            action_details=details or {},
        )
