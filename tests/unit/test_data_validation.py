"""Unit tests for data validation and cleaning."""

import pandas as pd

from src.data.generator import SyntheticDataGenerator
from src.data.validation import DataValidator


def test_validation_passes_on_clean_data():
    """Verify standard generated data passes validation."""
    gen = SyntheticDataGenerator(n_days=10, random_seed=42)
    df = gen.generate()

    res = DataValidator.validate(df)
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert res.total_rows == len(df)


def test_validation_fails_on_missing_column():
    """Verify validation detects missing required columns."""
    gen = SyntheticDataGenerator(n_days=5, random_seed=42)
    df = gen.generate().drop(columns=["price"])

    res = DataValidator.validate(df)
    assert res.is_valid is False
    assert any("price" in err for err in res.errors)


def test_validation_fails_on_negative_price():
    """Verify validation detects negative price."""
    gen = SyntheticDataGenerator(n_days=5, random_seed=42)
    df = gen.generate()
    df.loc[0, "price"] = -10.0

    res = DataValidator.validate(df)
    assert res.is_valid is False
    assert any("price" in err for err in res.errors)


def test_validation_fails_on_negative_quantity():
    """Verify validation detects negative quantities."""
    gen = SyntheticDataGenerator(n_days=5, random_seed=42)
    df = gen.generate()
    df.loc[0, "quantity"] = -5

    res = DataValidator.validate(df)
    assert res.is_valid is False
    assert any("quantity" in err for err in res.errors)


def test_validation_fails_on_invalid_customer_group():
    """Verify validation catches unauthorized customer categories."""
    gen = SyntheticDataGenerator(n_days=5, random_seed=42)
    df = gen.generate()
    df.loc[0, "customer_group"] = "vip_unauthorized"

    res = DataValidator.validate(df)
    assert res.is_valid is False
    assert any("customer_group" in err for err in res.errors)


def test_data_cleaner_removes_duplicates():
    """Verify cleaner eliminates duplicate keys."""
    gen = SyntheticDataGenerator(n_days=5, random_seed=42)
    df = gen.generate()

    # Duplicate first row
    duplicate_row = df.iloc[[0]].copy()
    dirty_df = pd.concat([df, duplicate_row], ignore_index=True)

    cleaned = DataValidator.clean(dirty_df)
    assert len(cleaned) == len(df)
