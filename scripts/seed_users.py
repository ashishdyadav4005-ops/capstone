"""Seed script to populate initial role-based users in SQLite."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root workspace directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.auth.repository import UserRecord, UserRepository
from src.auth.roles import UserRole
from src.auth.security import hash_password
from src.common.logger import get_logger

logger = get_logger("seed_users")


def seed_users() -> None:
    """Populate default demo users for all 4 roles."""
    logger.info("Seeding default BDS-39 system users...")
    repo = UserRepository()

    default_users = [
        UserRecord(
            user_id="USR_ADMIN_01",
            username="admin",
            email="admin@company.internal",
            full_name="Alex Vance (System Admin)",
            hashed_password=hash_password("Admin@12345"),
            role=UserRole.ADMIN,
            is_active=True,
        ),
        UserRecord(
            user_id="USR_ANALYST_01",
            username="analyst",
            email="analyst@company.internal",
            full_name="Sarah Chen (Pricing Analyst)",
            hashed_password=hash_password("Analyst@12345"),
            role=UserRole.ANALYST,
            is_active=True,
        ),
        UserRecord(
            user_id="USR_MANAGER_01",
            username="manager",
            email="manager@company.internal",
            full_name="Marcus Brody (Pricing Manager)",
            hashed_password=hash_password("Manager@12345"),
            role=UserRole.MANAGER,
            is_active=True,
        ),
        UserRecord(
            user_id="USR_GOVERNANCE_01",
            username="governance",
            email="governance@company.internal",
            full_name="Elena Rostova (Compliance & Governance Lead)",
            hashed_password=hash_password("Governance@12345"),
            role=UserRole.GOVERNANCE,
            is_active=True,
        ),
    ]

    for user in default_users:
        existing = repo.get_by_username(user.username)
        if existing is None:
            repo.create_user(user)
            logger.info(f"Created user '{user.username}' with role '{user.role.value}'")
        else:
            logger.info(f"User '{user.username}' already exists, skipping.")

    logger.info("User seeding complete. 4 demo accounts available.")
    print("\n" + "=" * 70)
    print("BDS-39 DEMO USERS SEEDED SUCCESSFULLY")
    print("=" * 70)
    print("  1. Username: admin      | Password: Admin@12345      | Role: admin")
    print("  2. Username: analyst    | Password: Analyst@12345    | Role: analyst")
    print("  3. Username: manager    | Password: Manager@12345    | Role: manager")
    print("  4. Username: governance | Password: Governance@12345 | Role: governance")
    print("=" * 70)


if __name__ == "__main__":
    seed_users()
