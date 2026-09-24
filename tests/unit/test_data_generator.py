"""Unit tests for synthetic data generator."""

import pandas as pd

from src.data.generator import SyntheticDataGenerator


def test_generator_reproducibility():
    """Verify that identical seeds produce identical datasets."""
    gen1 = SyntheticDataGenerator(n_days=10, random_seed=42)
    df1 = gen1.generate()

    gen2 = SyntheticDataGenerator(n_days=10, random_seed=42)
    df2 = gen2.generate()

    pd.testing.assert_frame_equal(df1, df2)


def test_generator_schema_and_non_empty():
    """Verify generated dataset contains expected columns and non-empty rows."""
    gen = SyntheticDataGenerator(n_days=5, random_seed=123)
    df = gen.generate()

    expected_cols = [
        "date",
        "product_id",
        "location_id",
        "customer_group",
        "price",
        "quantity",
        "capacity",
        "cost",
        "holiday",
        "event",
        "weather_temp",
        "competitor_price",
        "unobserved_shock",
        "is_random_policy",
        "true_elasticity",
    ]

    for col in expected_cols:
        assert col in df.columns

    assert len(df) > 0
    assert (df["price"] > 0).all()
    assert (df["quantity"] >= 0).all()
    assert (df["true_elasticity"] < 0).all()


def test_confounding_presence():
    """Verify that holiday/event days on average have higher prices in observational policy."""
    gen = SyntheticDataGenerator(n_days=100, random_seed=42)
    df = gen.generate()

    # On non-random policy days, holiday prices should be higher on average
    obs_df = df[df["is_random_policy"] == 0]
    holiday_prices = obs_df[obs_df["holiday"] == 1]["price"].mean()
    non_holiday_prices = obs_df[obs_df["holiday"] == 0]["price"].mean()

    assert holiday_prices > non_holiday_prices
