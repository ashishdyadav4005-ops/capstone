"""Unit tests for DataRepository (Parquet, SQLite, hash verification)."""

import tempfile
from pathlib import Path

import pandas as pd

from src.data.generator import SyntheticDataGenerator
from src.data.repository import DataRepository


def test_parquet_save_and_load():
    """Verify saving to Parquet and reloading returns identical DataFrame."""
    gen = SyntheticDataGenerator(n_days=10, random_seed=42)
    df = gen.generate()

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "test_transactions.parquet"
        content_hash = DataRepository.save_parquet(df, file_path)

        assert file_path.exists()
        assert len(content_hash) == 64 # SHA-256 length

        loaded_df = DataRepository.load_parquet(file_path)
        pd.testing.assert_frame_equal(df, loaded_df)


def test_sqlite_save_and_load():
    """Verify saving to SQLite table and loading with SQL queries."""
    gen = SyntheticDataGenerator(n_days=10, random_seed=42)
    df = gen.generate()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_pricing.db"
        db_url = f"sqlite:///{db_path}"

        DataRepository.save_sqlite(df, db_url=db_url, table_name="test_sales")
        loaded_df = DataRepository.load_sqlite(db_url=db_url, table_name="test_sales")

        assert len(loaded_df) == len(df)
        assert set(loaded_df["product_id"].unique()) == set(df["product_id"].unique())

        # Test with query filter
        filtered_df = DataRepository.load_sqlite(
            db_url=db_url, table_name="test_sales", query_filter="product_id = 'P001'"
        )
        assert len(filtered_df) > 0
        assert (filtered_df["product_id"] == "P001").all()


def test_dataset_hash_determinism():
    """Verify hash is identical for identical content and changes when data changes."""
    gen1 = SyntheticDataGenerator(n_days=5, random_seed=42)
    df1 = gen1.generate()

    gen2 = SyntheticDataGenerator(n_days=5, random_seed=42)
    df2 = gen2.generate()

    hash1 = DataRepository.compute_dataset_hash(df1)
    hash2 = DataRepository.compute_dataset_hash(df2)
    assert hash1 == hash2

    df_modified = df1.copy()
    df_modified.loc[0, "price"] = df_modified.loc[0, "price"] + 5.0
    hash_mod = DataRepository.compute_dataset_hash(df_modified)
    assert hash1 != hash_mod
