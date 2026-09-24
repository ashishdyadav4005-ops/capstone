"""Persistence repository for dynamic price recommendations in SQLite."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.api.schemas import RecommendationRecord, RecommendationState
from src.common.config import find_project_root, get_config
from src.common.logger import get_logger

logger = get_logger("recommendation_repository")


class RecommendationRepository:
    """Manages CRUD operations and state persistence for dynamic pricing recommendations."""

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

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with Row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create price_recommendations table if it does not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_recommendations (
                    recommendation_id TEXT PRIMARY KEY,
                    item_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    customer_group TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    base_price REAL NOT NULL,
                    recommended_price REAL NOT NULL,
                    final_price REAL NOT NULL,
                    unit_cost REAL NOT NULL,
                    expected_demand REAL NOT NULL,
                    expected_revenue REAL NOT NULL,
                    expected_profit REAL NOT NULL,
                    margin_pct REAL NOT NULL,
                    state TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    reviewed_by TEXT,
                    override_price REAL,
                    override_reason TEXT,
                    audit_notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def save(self, rec: RecommendationRecord) -> None:
        """Insert or replace a recommendation record."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO price_recommendations (
                    recommendation_id, item_id, product_id, customer_group, location_id,
                    base_price, recommended_price, final_price, unit_cost, expected_demand,
                    expected_revenue, expected_profit, margin_pct, state, created_by,
                    reviewed_by, override_price, override_reason, audit_notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec.recommendation_id,
                rec.item_id,
                rec.product_id,
                rec.customer_group,
                rec.location_id,
                rec.base_price,
                rec.recommended_price,
                rec.final_price,
                rec.unit_cost,
                rec.expected_demand,
                rec.expected_revenue,
                rec.expected_profit,
                rec.margin_pct,
                rec.state.value if isinstance(rec.state, RecommendationState) else str(rec.state),
                rec.created_by,
                rec.reviewed_by,
                rec.override_price,
                rec.override_reason,
                rec.audit_notes,
                rec.created_at,
                rec.updated_at,
            ))
            conn.commit()

    def bulk_save(self, records: list[RecommendationRecord]) -> None:
        """Insert multiple recommendation records in a single transaction."""
        with self._get_connection() as conn:
            for rec in records:
                conn.execute("""
                    INSERT OR REPLACE INTO price_recommendations (
                        recommendation_id, item_id, product_id, customer_group, location_id,
                        base_price, recommended_price, final_price, unit_cost, expected_demand,
                        expected_revenue, expected_profit, margin_pct, state, created_by,
                        reviewed_by, override_price, override_reason, audit_notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rec.recommendation_id,
                    rec.item_id,
                    rec.product_id,
                    rec.customer_group,
                    rec.location_id,
                    rec.base_price,
                    rec.recommended_price,
                    rec.final_price,
                    rec.unit_cost,
                    rec.expected_demand,
                    rec.expected_revenue,
                    rec.expected_profit,
                    rec.margin_pct,
                    rec.state.value if isinstance(rec.state, RecommendationState) else str(rec.state),
                    rec.created_by,
                    rec.reviewed_by,
                    rec.override_price,
                    rec.override_reason,
                    rec.audit_notes,
                    rec.created_at,
                    rec.updated_at,
                ))
            conn.commit()

    def get_by_id(self, recommendation_id: str) -> RecommendationRecord | None:
        """Retrieve a recommendation by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM price_recommendations WHERE recommendation_id = ?",
                (recommendation_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def list_recommendations(
        self,
        product_id: str | None = None,
        customer_group: str | None = None,
        location_id: str | None = None,
        state: RecommendationState | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RecommendationRecord]:
        """List recommendations with optional filters."""
        query = "SELECT * FROM price_recommendations WHERE 1=1"
        params: list[Any] = []

        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)
        if customer_group:
            query += " AND customer_group = ?"
            params.append(customer_group)
        if location_id:
            query += " AND location_id = ?"
            params.append(location_id)
        if state:
            query += " AND state = ?"
            params.append(state.value if isinstance(state, RecommendationState) else str(state))

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> RecommendationRecord:
        """Convert a SQLite row to a RecommendationRecord Pydantic model."""
        return RecommendationRecord(
            recommendation_id=row["recommendation_id"],
            item_id=row["item_id"],
            product_id=row["product_id"],
            customer_group=row["customer_group"],
            location_id=row["location_id"],
            base_price=row["base_price"],
            recommended_price=row["recommended_price"],
            final_price=row["final_price"],
            unit_cost=row["unit_cost"],
            expected_demand=row["expected_demand"],
            expected_revenue=row["expected_revenue"],
            expected_profit=row["expected_profit"],
            margin_pct=row["margin_pct"],
            state=RecommendationState(row["state"]),
            created_by=row["created_by"],
            reviewed_by=row["reviewed_by"],
            override_price=row["override_price"],
            override_reason=row["override_reason"],
            audit_notes=row["audit_notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def now_iso() -> str:
        """Generate current UTC timestamp in ISO 8601 format."""
        return datetime.now(UTC).isoformat()
