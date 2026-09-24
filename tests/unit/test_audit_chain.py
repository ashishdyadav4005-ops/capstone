"""Unit tests for Cryptographic Audit Trail and SHA-256 Hash Chain Verification."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.audit.models import (
    GENESIS_HASH,
    AuditEventType,
    calculate_entry_hash,
)
from src.audit.repository import AuditRepository
from src.audit.verification import AuditChainVerifier


@pytest.fixture
def temp_audit_repo():
    """Create a clean temporary SQLite database for testing audit repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_audit.db"
        repo = AuditRepository(db_path=db_path)
        yield repo


class TestAuditChain:
    """Test cryptographic SHA-256 chaining and tamper detection."""

    def test_genesis_hash_and_entry_computation(self):
        """Verify deterministic canonical hashing for genesis and subsequent entries."""
        h1 = calculate_entry_hash(
            sequence_number=1,
            timestamp="2026-09-22T10:00:00Z",
            event_type=AuditEventType.USER_LOGIN_SUCCESS.value,
            actor_username="admin",
            actor_role="admin",
            action_details={"ip": "127.0.0.1"},
            previous_entry_hash=GENESIS_HASH,
        )
        assert isinstance(h1, str)
        assert len(h1) == 64

        # Same input must produce exact same SHA-256
        h1_repeat = calculate_entry_hash(
            sequence_number=1,
            timestamp="2026-09-22T10:00:00Z",
            event_type=AuditEventType.USER_LOGIN_SUCCESS.value,
            actor_username="admin",
            actor_role="admin",
            action_details={"ip": "127.0.0.1"},
            previous_entry_hash=GENESIS_HASH,
        )
        assert h1 == h1_repeat

        # Different detail must produce completely different hash
        h2 = calculate_entry_hash(
            sequence_number=1,
            timestamp="2026-09-22T10:00:00Z",
            event_type=AuditEventType.USER_LOGIN_SUCCESS.value,
            actor_username="admin",
            actor_role="admin",
            action_details={"ip": "192.168.1.1"},
            previous_entry_hash=GENESIS_HASH,
        )
        assert h1 != h2

    def test_append_and_sequential_chaining(self, temp_audit_repo: AuditRepository):
        """Verify sequential entries link previous hashes correctly."""
        verifier = AuditChainVerifier(repository=temp_audit_repo)

        # Empty chain verifies cleanly
        res_empty = verifier.verify_chain()
        assert res_empty.is_valid is True
        assert res_empty.total_entries == 0

        # Append 3 entries
        e1 = temp_audit_repo.append(
            event_type=AuditEventType.PRICE_RECOMMENDATIONS_GENERATED,
            actor_username="OPTIMIZER",
            actor_role="system",
            action_details={"count": 5},
        )
        assert e1.sequence_number == 1
        assert e1.previous_entry_hash == GENESIS_HASH

        e2 = temp_audit_repo.append(
            event_type=AuditEventType.PRICE_APPROVED,
            actor_username="manager_bob",
            actor_role="manager",
            action_details={"recommendation_id": "REC_001", "price": 105.0},
        )
        assert e2.sequence_number == 2
        assert e2.previous_entry_hash == e1.entry_hash

        e3 = temp_audit_repo.append(
            event_type=AuditEventType.PRICE_PUBLISHED,
            actor_username="system",
            actor_role="system",
            action_details={"recommendation_id": "REC_001", "price": 105.0},
        )
        assert e3.sequence_number == 3
        assert e3.previous_entry_hash == e2.entry_hash

        # Live verification
        res = verifier.verify_chain()
        assert res.is_valid is True
        assert res.total_entries == 3

    def test_tamper_detection_on_mutated_entry_details(self, temp_audit_repo: AuditRepository):
        """Verify that altering even 1 cent in action details invalidates the audit chain."""
        temp_audit_repo.append(
            event_type=AuditEventType.PRICE_RECOMMENDATIONS_GENERATED,
            actor_username="OPTIMIZER",
            actor_role="system",
            action_details={"count": 10},
        )
        temp_audit_repo.append(
            event_type=AuditEventType.PRICE_APPROVED,
            actor_username="manager",
            actor_role="manager",
            action_details={"recommendation_id": "REC_002", "price": 120.00},
        )

        verifier = AuditChainVerifier(repository=temp_audit_repo)
        assert verifier.verify_chain().is_valid is True

        # Malicious actor tampers with SQLite table directly
        conn = sqlite3.connect(temp_audit_repo.db_path)
        try:
            conn.execute(
                "UPDATE audit_logs SET action_details_json = ? WHERE sequence_number = 2",
                ('{"recommendation_id": "REC_002", "price": 999.99}',),
            )
            conn.commit()
        finally:
            conn.close()

        # Chain verification must catch the tamper
        res = verifier.verify_chain()
        assert res.is_valid is False
        assert res.corrupted_sequence_number == 2
        assert "tampering detected" in (res.error_message or "")

    def test_tamper_detection_on_deleted_record(self, temp_audit_repo: AuditRepository):
        """Verify that deleting a record from the middle of the chain breaks linkage."""
        for i in range(1, 6):
            temp_audit_repo.append(
                event_type=AuditEventType.PRICE_RECOMMENDATIONS_GENERATED,
                actor_username="OPTIMIZER",
                actor_role="system",
                action_details={"batch_num": i},
            )

        verifier = AuditChainVerifier(repository=temp_audit_repo)
        assert verifier.verify_chain().is_valid is True
        assert temp_audit_repo.count() == 5

        # Malicious actor deletes record 3
        conn = sqlite3.connect(temp_audit_repo.db_path)
        try:
            conn.execute("DELETE FROM audit_logs WHERE sequence_number = 3")
            conn.commit()
        finally:
            conn.close()

        # Chain verification must detect sequence gap / hash discontinuity
        res = verifier.verify_chain()
        assert res.is_valid is False
        assert res.corrupted_sequence_number in (3, 4)
        assert "discontinuity" in (res.error_message or "").lower() or "broken" in (res.error_message or "").lower()

    def test_list_entries_and_filtering(self, temp_audit_repo: AuditRepository):
        """Verify querying audit logs with event type and actor filters."""
        temp_audit_repo.append(
            event_type=AuditEventType.USER_LOGIN_SUCCESS,
            actor_username="alice",
            actor_role="analyst",
            action_details={"action": "login"},
        )
        temp_audit_repo.append(
            event_type=AuditEventType.PRICE_APPROVED,
            actor_username="bob",
            actor_role="manager",
            action_details={"action": "approve"},
        )
        temp_audit_repo.append(
            event_type=AuditEventType.USER_LOGIN_SUCCESS,
            actor_username="bob",
            actor_role="manager",
            action_details={"action": "login"},
        )

        all_entries = temp_audit_repo.list_entries()
        assert len(all_entries) == 3

        alice_entries = temp_audit_repo.list_entries(actor_username="alice")
        assert len(alice_entries) == 1
        assert alice_entries[0].actor_username == "alice"

        login_entries = temp_audit_repo.list_entries(event_type=AuditEventType.USER_LOGIN_SUCCESS.value)
        assert len(login_entries) == 2
