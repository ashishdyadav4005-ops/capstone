"""Unit tests for chronological splitting and temporal leakage assertions."""

import pytest

from src.data.generator import SyntheticDataGenerator
from src.data.split import TimeSeriesSplitter


def test_time_series_split_partitions():
    """Verify train, val, and test splits partition the dataset without overlap."""
    gen = SyntheticDataGenerator(n_days=100, random_seed=42)
    df = gen.generate()

    train_df, val_df, test_df = TimeSeriesSplitter.split(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert len(train_df) > len(val_df)
    assert len(val_df) == len(test_df)


def test_leakage_verification_clean():
    """Verify leakage verification succeeds on chronological splits."""
    gen = SyntheticDataGenerator(n_days=50, random_seed=42)
    df = gen.generate()

    train_df, val_df, test_df = TimeSeriesSplitter.split(df)
    res = TimeSeriesSplitter.verify_no_leakage(train_df, val_df, test_df)

    assert res.is_leakage_free is True
    assert len(res.violations) == 0


def test_leakage_verification_catches_overlap():
    """Verify leakage check catches intentional date overlaps."""
    gen = SyntheticDataGenerator(n_days=50, random_seed=42)
    df = gen.generate()

    train_df, val_df, test_df = TimeSeriesSplitter.split(df)

    # Intentionally corrupt train set with validation date
    train_corrupt = train_df.copy()
    train_corrupt.loc[0, "date"] = val_df.loc[0, "date"]

    res = TimeSeriesSplitter.verify_no_leakage(train_corrupt, val_df, test_df)
    assert res.is_leakage_free is False
    assert len(res.violations) > 0


def test_invalid_ratios_raise_error():
    """Verify invalid split ratios raise a ValueError."""
    gen = SyntheticDataGenerator(n_days=10, random_seed=42)
    df = gen.generate()

    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        TimeSeriesSplitter.split(df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)
