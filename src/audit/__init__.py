"""Audit package: Immutable cryptographic audit trail and chain verification."""

from src.audit.logger import AuditLogger
from src.audit.models import (
    GENESIS_HASH,
    AuditEntry,
    AuditEventType,
    calculate_entry_hash,
    canonical_json_dumps,
)
from src.audit.repository import AuditRepository
from src.audit.verification import AuditChainVerifier, ChainVerificationResult

__all__ = [
    "AuditEventType",
    "AuditEntry",
    "GENESIS_HASH",
    "calculate_entry_hash",
    "canonical_json_dumps",
    "AuditRepository",
    "ChainVerificationResult",
    "AuditChainVerifier",
    "AuditLogger",
]
