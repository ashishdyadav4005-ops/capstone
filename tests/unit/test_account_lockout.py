"""Unit tests for failed login tracking and account lockout mechanism."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.auth.repository import UserRecord, UserRepository
from src.auth.roles import UserRole
from src.auth.security import hash_password


@pytest.fixture
def temp_user_repo(tmp_path: Path) -> UserRepository:
    """Create a test UserRepository with a lower threshold for fast lockout testing."""
    db_file = tmp_path / "test_auth.db"
    return UserRepository(
        db_path=db_file,
        max_failed_logins=3, # 3 failed attempts to lock
        lockout_duration_minutes=10,
    )


@pytest.fixture
def sample_user(temp_user_repo: UserRepository) -> UserRecord:
    """Create and persist a test user."""
    user = UserRecord(
        user_id="USR_TEST_001",
        username="testanalyst",
        email="test@company.internal",
        full_name="Test Analyst",
        hashed_password=hash_password("ValidPassword@123"),
        role=UserRole.ANALYST,
        is_active=True,
    )
    temp_user_repo.create_user(user)
    return user


class TestAccountLockout:
    """Test suite for account lockout and failed attempt tracking."""

    def test_failed_attempts_increment(
        self,
        temp_user_repo: UserRepository,
        sample_user: UserRecord,
    ) -> None:
        """Verify failed login attempt counter increments."""
        username = sample_user.username

        count1, is_locked1 = temp_user_repo.record_login_failure(username)
        assert count1 == 1
        assert not is_locked1

        count2, is_locked2 = temp_user_repo.record_login_failure(username)
        assert count2 == 2
        assert not is_locked2

        user_db = temp_user_repo.get_by_username(username)
        assert user_db is not None
        assert user_db.failed_login_attempts == 2
        assert not temp_user_repo.is_locked(user_db)

    def test_lockout_triggered_at_threshold(
        self,
        temp_user_repo: UserRepository,
        sample_user: UserRecord,
    ) -> None:
        """Verify that reaching max_failed_logins triggers account lockout."""
        username = sample_user.username

        temp_user_repo.record_login_failure(username) # 1
        temp_user_repo.record_login_failure(username) # 2
        count3, is_locked3 = temp_user_repo.record_login_failure(username) # 3 -> threshold!

        assert count3 == 3
        assert is_locked3

        user_db = temp_user_repo.get_by_username(username)
        assert user_db is not None
        assert temp_user_repo.is_locked(user_db)
        assert user_db.locked_until is not None

    def test_successful_login_resets_failure_counter(
        self,
        temp_user_repo: UserRepository,
        sample_user: UserRecord,
    ) -> None:
        """Verify that a successful login resets failed attempts to zero."""
        username = sample_user.username

        temp_user_repo.record_login_failure(username) # 1
        temp_user_repo.record_login_failure(username) # 2

        temp_user_repo.record_login_success(username)

        user_db = temp_user_repo.get_by_username(username)
        assert user_db is not None
        assert user_db.failed_login_attempts == 0
        assert user_db.last_login_at is not None

    def test_admin_manual_unlock(
        self,
        temp_user_repo: UserRepository,
        sample_user: UserRecord,
    ) -> None:
        """Verify admin can manually unlock a locked account."""
        username = sample_user.username

        # Trigger lockout
        for _ in range(3):
            temp_user_repo.record_login_failure(username)

        user_db = temp_user_repo.get_by_username(username)
        assert user_db is not None
        assert temp_user_repo.is_locked(user_db)

        # Admin unlocks
        success = temp_user_repo.unlock_user(sample_user.user_id)
        assert success

        user_unlocked = temp_user_repo.get_by_username(username)
        assert user_unlocked is not None
        assert not temp_user_repo.is_locked(user_unlocked)
        assert user_unlocked.failed_login_attempts == 0
