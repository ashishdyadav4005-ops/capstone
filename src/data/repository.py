"""Data repository module providing persistence to Parquet and SQLite with cryptographic dataset hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


class DataRepository:
    """Handles dataset persistence to SQLite and Parquet files alongside provenance tracking."""

    @staticmethod
    def compute_dataset_hash(df: pd.DataFrame) -> str:
        """Compute deterministic SHA-256 hash representing dataset content."""
        # Normalize and serialize dataframe to byte stream for deterministic hashing
        sorted_df = df.sort_index(axis=1).sort_values(by=list(df.columns[:min(4, len(df.columns))]))
        content_bytes = sorted_df.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(content_bytes).hexdigest()

    @classmethod
    def save_parquet(cls, df: pd.DataFrame, file_path: str | Path) -> str:
        """Save DataFrame to Parquet file and return its SHA-256 content hash."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save parquet with snappy compression
        df.to_parquet(path, index=False, engine="pyarrow")
        return cls.compute_dataset_hash(df)

    @classmethod
    def load_parquet(cls, file_path: str | Path) -> pd.DataFrame:
        """Load DataFrame from Parquet file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file does not exist at {path}")
        return pd.read_parquet(path, engine="pyarrow")

    @classmethod
    def save_sqlite(
        cls,
        df: pd.DataFrame,
        db_url: str,
        table_name: str = "sales_transactions",
        if_exists: str = "replace",
    ) -> None:
        """Save DataFrame to a SQLite database table."""
        if db_url.startswith("sqlite:///"):
            raw_path = db_url.replace("sqlite:///", "")
            Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
            import sqlite3
            conn = sqlite3.connect(raw_path)
            try:
                df.to_sql(table_name, con=conn, if_exists=if_exists, index=False)
            finally:
                conn.close()
        else:
            from sqlalchemy import create_engine
            engine = create_engine(db_url)
            with engine.begin() as connection:
                df.to_sql(table_name, con=connection, if_exists=if_exists, index=False)
            engine.dispose()

    @classmethod
    def load_sqlite(
        cls,
        db_url: str,
        table_name: str = "sales_transactions",
        query_filter: str | None = None,
    ) -> pd.DataFrame:
        """Query DataFrame from SQLite database table."""
        sql = f"SELECT * FROM {table_name}"
        if query_filter:
            sql += f" WHERE {query_filter}"

        if db_url.startswith("sqlite:///"):
            raw_path = db_url.replace("sqlite:///", "")
            import sqlite3
            conn = sqlite3.connect(raw_path)
            try:
                res_df = pd.read_sql(sql, con=conn)
            finally:
                conn.close()
            return res_df
        else:
            from sqlalchemy import create_engine
            engine = create_engine(db_url)
            try:
                with engine.connect() as connection:
                    res_df = pd.read_sql(sql, con=connection)
            finally:
                engine.dispose()
            return res_df

    @classmethod
    def get_dataset_metadata(cls, df: pd.DataFrame, file_path: str | Path | None = None) -> dict[str, Any]:
        """Generate provenance and summary metadata for dataset."""
        dataset_hash = cls.compute_dataset_hash(df)
        metadata = {
            "version_hash": dataset_hash,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "memory_usage_mb": float(round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)),
        }
        if file_path:
            metadata["storage_path"] = str(file_path)
        if "date" in df.columns:
            metadata["date_range"] = {
                "start": str(df["date"].min()),
                "end": str(df["date"].max()),
            }
        return metadata
