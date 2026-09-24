"""Audit log persistence repository in SQLite with append-only hash chaining."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.audit.models import (
    GENESIS_HASH,
    AuditEntry,
    AuditEventType,
    calculate_entry_hash,
    canonical_json_dumps,
)
from src.common.config import find_project_root, get_config
from src.common.logger import get_logger

logger = get_logger("audit_repository")


class AuditRepository:
    """Manages append-only cryptographic audit logging in SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            config = get_config()
            db_url = getattr(config.database, "url", "sqlite:///./data/bds39_pricing.db")
            clean_path = db_url.replace("sqlite:///", "").lstrip("./").lstrip("/")
            self.db_path = find_project_root() / clean_path
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create a connection with Row factory and guarantee closure."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create audit_logs table and index if not exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    entry_id TEXT PRIMARY KEY,
                    sequence_number INTEGER UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_username TEXT NOT NULL,
                    actor_role TEXT NOT NULL,
                    action_details_json TEXT NOT NULL,
                    previous_entry_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_seq ON audit_logs (sequence_number ASC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_logs (event_type)
            """)
            conn.commit()

    def append(
        self,
        event_type: AuditEventType | str,
        actor_username: str,
        actor_role: str,
        action_details: dict[str, Any],
        custom_timestamp: str | None = None,
    ) -> AuditEntry:
        """Append a new audit entry cryptographically linked to the previous log tip."""
        now = custom_timestamp or datetime.now(UTC).isoformat()
        evt_str = event_type.value if isinstance(event_type, AuditEventType) else str(event_type)
        entry_id = f"AUD_{uuid.uuid4().hex[:10].upper()}"

        with self._get_connection() as conn:
            # Fetch previous tip
            cur = conn.cursor()
            cur.execute("SELECT sequence_number, entry_hash FROM audit_logs ORDER BY sequence_number DESC LIMIT 1")
            row = cur.fetchone()

            if row is None:
                seq_num = 1
                prev_hash = GENESIS_HASH
            else:
                seq_num = row["sequence_number"] + 1
                prev_hash = row["entry_hash"]

            # Calculate cryptographic hash
            curr_hash = calculate_entry_hash(
                sequence_number=seq_num,
                timestamp=now,
                event_type=evt_str,
                actor_username=actor_username,
                actor_role=actor_role,
                action_details=action_details,
                previous_entry_hash=prev_hash,
            )

            details_json = canonical_json_dumps(action_details)

            cur.execute("""
                INSERT INTO audit_logs (
                    entry_id, sequence_number, timestamp, event_type,
                    actor_username, actor_role, action_details_json,
                    previous_entry_hash, entry_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id,
                seq_num,
                now,
                evt_str,
                actor_username,
                actor_role,
                details_json,
                prev_hash,
                curr_hash,
            ))
            conn.commit()

        entry = AuditEntry(
            entry_id=entry_id,
            sequence_number=seq_num,
            timestamp=now,
            event_type=AuditEventType(evt_str),
            actor_username=actor_username,
            actor_role=actor_role,
            action_details=action_details,
            previous_entry_hash=prev_hash,
            entry_hash=curr_hash,
        )

        logger.info(
            f"Audit log entry #{seq_num} appended [{evt_str}] by {actor_username}",
            extra={"seq": seq_num, "event_type": evt_str, "hash": curr_hash[:12]},
        )
        return entry

    def get_latest_entry(self) -> AuditEntry | None:
        """Return the newest tip of the audit chain."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM audit_logs ORDER BY sequence_number DESC LIMIT 1").fetchone()
            if row is None:
                return None
            return self._row_to_entry(row)

    def get_all_entries(self) -> list[AuditEntry]:
        """Return all audit entries in ascending sequential order for verification."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM audit_logs ORDER BY sequence_number ASC").fetchall()
            return [self._row_to_entry(r) for r in rows]

    def list_entries(
        self,
        event_type: str | None = None,
        actor_username: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Query audit logs in reverse chronological order with optional filters."""
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params: list[Any] = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if actor_username:
            query += " AND actor_username = ?"
            params.append(actor_username)

        query += " ORDER BY sequence_number DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def count(self) -> int:
        """Return total count of audit records."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM audit_logs").fetchone()
            return row["cnt"] if row else 0

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
        """Convert a SQLite row to an AuditEntry model."""
        details = json.loads(row["action_details_json"])
        return AuditEntry(
            entry_id=row["entry_id"],
            sequence_number=row["sequence_number"],
            timestamp=row["timestamp"],
            event_type=AuditEventType(row["event_type"]),
            actor_username=row["actor_username"],
            actor_role=row["actor_role"],
            action_details=details,
            previous_entry_hash=row["previous_entry_hash"],
            entry_hash=row["entry_hash"],
        )
