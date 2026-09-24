"""Audit logging models and canonical SHA-256 cryptographic hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class AuditEventType(StrEnum):
    """Types of auditable actions and lifecycle events in BDS-39."""

    # Dynamic Pricing & Optimization events
    PRICE_RECOMMENDATIONS_GENERATED = "PRICE_RECOMMENDATIONS_GENERATED"
    PRICE_OVERRIDE_APPLIED = "PRICE_OVERRIDE_APPLIED"
    PRICE_APPROVED = "PRICE_APPROVED"
    PRICE_REJECTED = "PRICE_REJECTED"
    PRICE_PUBLISHED = "PRICE_PUBLISHED"
    EMERGENCY_ROLLBACK = "EMERGENCY_ROLLBACK"

    # Model & Data events
    MODEL_TRAINED = "MODEL_TRAINED"
    DATASET_GENERATED = "DATASET_GENERATED"
    GUARDRAIL_VIOLATION_DETECTED = "GUARDRAIL_VIOLATION_DETECTED"

    # User & Security events
    USER_LOGIN_SUCCESS = "USER_LOGIN_SUCCESS"
    USER_LOGIN_FAILED = "USER_LOGIN_FAILED"
    USER_ACCOUNT_LOCKED = "USER_ACCOUNT_LOCKED"
    USER_ACCOUNT_UNLOCKED = "USER_ACCOUNT_UNLOCKED"
    USER_LOCKED_OUT = "USER_ACCOUNT_LOCKED"
    USER_UNLOCKED = "USER_ACCOUNT_UNLOCKED"
    USER_CREATED = "USER_CREATED"
    SYSTEM_CONFIG_UPDATED = "SYSTEM_CONFIG_UPDATED"


GENESIS_HASH = "0" * 64


def canonical_json_dumps(obj: Any) -> str:
    """Serialize object to deterministic, sorted, whitespace-trimmed JSON string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def calculate_entry_hash(
    sequence_number: int,
    timestamp: str,
    event_type: str,
    actor_username: str,
    actor_role: str,
    action_details: dict[str, Any],
    previous_entry_hash: str,
) -> str:
    """Calculate the cryptographic SHA-256 hash for an audit entry chained to its predecessor."""
    canonical_payload = canonical_json_dumps({
        "sequence_number": sequence_number,
        "timestamp": timestamp,
        "event_type": event_type,
        "actor_username": actor_username,
        "actor_role": actor_role,
        "action_details": action_details,
        "previous_entry_hash": previous_entry_hash,
    })
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


# Alias for compatibility
compute_entry_hash = calculate_entry_hash


@dataclass
class AuditEntry:
    """Cryptographically chained audit log entry."""

    entry_id: str
    sequence_number: int
    timestamp: str
    event_type: AuditEventType
    actor_username: str
    actor_role: str
    action_details: dict[str, Any]
    previous_entry_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        d = asdict(self)
        d["event_type"] = self.event_type.value if isinstance(self.event_type, AuditEventType) else str(self.event_type)
        return d
