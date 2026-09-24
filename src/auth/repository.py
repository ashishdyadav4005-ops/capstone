"""User persistence repository and account lockout manager in SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.auth.roles import UserRole
from src.common.config import find_project_root, get_config
from src.common.logger import get_logger

logger = get_logger("user_repository")


@dataclass
class UserRecord:
    """User entity record stored in the database."""

    user_id: str
    username: str
    email: str
    full_name: str
    hashed_password: str
    role: UserRole
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: str | None = None
    created_at: str = ""
    last_login_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary without exposing password hash."""
        d = asdict(self)
        d.pop("hashed_password", None)
        return d


class UserRepository:
    """Manages user persistence, credentials, and account lockout security."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        max_failed_logins: int = 5,
        lockout_duration_minutes: int = 15,
    ):
        if db_path is None:
            config = get_config()
            db_url = getattr(config.database, "url", "sqlite:///./data/bds39_pricing.db")
            clean_path = db_url.replace("sqlite:///", "").lstrip("./").lstrip("/")
            self.db_path = find_project_root() / clean_path
        else:
            self.db_path = Path(db_path)

        self.max_failed_logins = max_failed_logins
        self.lockout_duration_minutes = lockout_duration_minutes

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with Row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create users table if not exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    hashed_password TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until TEXT,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT
                )
            """)
            conn.commit()

    def create_user(self, user: UserRecord) -> UserRecord:
        """Insert a new user record."""
        now = datetime.now(UTC).isoformat()
        if not user.created_at:
            user.created_at = now

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO users (
                    user_id, username, email, full_name, hashed_password,
                    role, is_active, failed_login_attempts, locked_until, created_at, last_login_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user.user_id,
                user.username.lower().strip(),
                user.email,
                user.full_name,
                user.hashed_password,
                user.role.value if isinstance(user.role, UserRole) else str(user.role),
                1 if user.is_active else 0,
                user.failed_login_attempts,
                user.locked_until,
                user.created_at,
                user.last_login_at,
            ))
            conn.commit()
        return user

    def get_by_username(self, username: str) -> UserRecord | None:
        """Find a user by username."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username.lower().strip(),),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def get_by_id(self, user_id: str) -> UserRecord | None:
        """Find a user by user_id."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def list_users(self) -> list[UserRecord]:
        """List all users."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY username ASC").fetchall()
            return [self._row_to_record(r) for r in rows]

    def record_login_success(self, username: str) -> None:
        """Reset failed login attempts and update last_login_at timestamp."""
        now = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE users
                SET failed_login_attempts = 0,
                    locked_until = NULL,
                    last_login_at = ?
                WHERE username = ?
            """, (now, username.lower().strip()))
            conn.commit()

    def record_login_failure(self, username: str) -> tuple[int, bool]:
        """Increment failed attempts and apply account lockout if threshold reached.

        Returns
        -------
        tuple[int, bool]
            (new_failed_count, is_locked)
        """
        user = self.get_by_username(username)
        if user is None:
            return 0, False

        new_count = user.failed_login_attempts + 1
        locked_until_val = user.locked_until

        is_locked = False
        if new_count >= self.max_failed_logins:
            is_locked = True
            lock_exp = datetime.now(UTC) + timedelta(minutes=self.lockout_duration_minutes)
            locked_until_val = lock_exp.isoformat()
            logger.warning(
                f"Account '{username}' locked due to {new_count} consecutive failed login attempts.",
                extra={"username": username, "locked_until": locked_until_val},
            )

        with self._get_connection() as conn:
            conn.execute("""
                UPDATE users
                SET failed_login_attempts = ?,
                    locked_until = ?
                WHERE username = ?
            """, (new_count, locked_until_val, username.lower().strip()))
            conn.commit()

        return new_count, is_locked

    def is_locked(self, user: UserRecord) -> bool:
        """Check if an account is currently in a locked state."""
        if not user.locked_until:
            return False
        try:
            lock_time = datetime.fromisoformat(user.locked_until)
            if lock_time.tzinfo is None:
                lock_time = lock_time.replace(tzinfo=UTC)
            return datetime.now(UTC) < lock_time
        except Exception:
            return False

    def unlock_user(self, user_id: str) -> bool:
        """Manually unlock an account."""
        with self._get_connection() as conn:
            cur = conn.execute("""
                UPDATE users
                SET failed_login_attempts = 0,
                    locked_until = NULL
                WHERE user_id = ?
            """, (user_id,))
            conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> UserRecord:
        """Convert SQLite row to UserRecord."""
        return UserRecord(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            full_name=row["full_name"],
            hashed_password=row["hashed_password"],
            role=UserRole(row["role"]),
            is_active=bool(row["is_active"]),
            failed_login_attempts=int(row["failed_login_attempts"]),
            locked_until=row["locked_until"],
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
        )
