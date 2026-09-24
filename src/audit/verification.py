"""Audit chain cryptographic verification engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from src.audit.models import GENESIS_HASH, AuditEntry, calculate_entry_hash
from src.audit.repository import AuditRepository
from src.common.logger import get_logger

logger = get_logger("audit_verification")


@dataclass
class ChainVerificationResult:
    """Outcome of cryptographic audit chain verification."""

    is_valid: bool
    total_entries: int
    verified_at: str
    corrupted_sequence_number: int | None = None
    error_message: str | None = None
    latest_entry_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class AuditChainVerifier:
    """Verifies cryptographic chain integrity and detects data tampering or deletion."""

    def __init__(self, repository: AuditRepository | None = None):
        self.repository = repository or AuditRepository()

    def verify_chain(
        self,
        entries: list[AuditEntry] | None = None,
    ) -> ChainVerificationResult:
        """Verify the full cryptographic integrity of the audit log chain."""
        now = datetime.now(UTC).isoformat()
        all_entries = entries if entries is not None else self.repository.get_all_entries()

        if not all_entries:
            return ChainVerificationResult(
                is_valid=True,
                total_entries=0,
                verified_at=now,
                error_message="Audit log is empty (0 records).",
            )

        expected_prev_hash = GENESIS_HASH

        for idx, entry in enumerate(all_entries):
            expected_seq = idx + 1

            # 1. Check Sequence Number Continuity
            if entry.sequence_number != expected_seq:
                err = (
                    f"Sequence discontinuity at index {idx}: Expected #{expected_seq}, "
                    f"found #{entry.sequence_number}. Possible record deletion or insertion."
                )
                logger.error(err)
                return ChainVerificationResult(
                    is_valid=False,
                    total_entries=len(all_entries),
                    verified_at=now,
                    corrupted_sequence_number=entry.sequence_number,
                    error_message=err,
                    latest_entry_hash=entry.entry_hash,
                )

            # 2. Check Previous Hash Linkage
            if entry.previous_entry_hash != expected_prev_hash:
                err = (
                    f"Hash link broken at sequence #{entry.sequence_number}: "
                    f"Expected previous hash {expected_prev_hash[:12]}..., found {entry.previous_entry_hash[:12]}..."
                )
                logger.error(err)
                return ChainVerificationResult(
                    is_valid=False,
                    total_entries=len(all_entries),
                    verified_at=now,
                    corrupted_sequence_number=entry.sequence_number,
                    error_message=err,
                    latest_entry_hash=entry.entry_hash,
                )

            # 3. Recompute and Verify Cryptographic SHA-256 Hash
            recalculated_hash = calculate_entry_hash(
                sequence_number=entry.sequence_number,
                timestamp=entry.timestamp,
                event_type=entry.event_type.value if hasattr(entry.event_type, "value") else str(entry.event_type),
                actor_username=entry.actor_username,
                actor_role=entry.actor_role,
                action_details=entry.action_details,
                previous_entry_hash=entry.previous_entry_hash,
            )

            if recalculated_hash != entry.entry_hash:
                err = (
                    f"Cryptographic payload tampering detected at sequence #{entry.sequence_number}: "
                    f"Stored hash {entry.entry_hash[:12]}... != Computed hash {recalculated_hash[:12]}..."
                )
                logger.error(err)
                return ChainVerificationResult(
                    is_valid=False,
                    total_entries=len(all_entries),
                    verified_at=now,
                    corrupted_sequence_number=entry.sequence_number,
                    error_message=err,
                    latest_entry_hash=entry.entry_hash,
                )

            expected_prev_hash = entry.entry_hash

        logger.info(
            f"Audit chain verification PASSED. Verified {len(all_entries)} entries.",
            extra={"total": len(all_entries), "tip_hash": expected_prev_hash[:12]},
        )
        return ChainVerificationResult(
            is_valid=True,
            total_entries=len(all_entries),
            verified_at=now,
            latest_entry_hash=expected_prev_hash,
            error_message="Audit chain cryptographic integrity is 100% verified.",
        )
