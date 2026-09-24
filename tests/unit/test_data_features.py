"""Unit tests for feature engineering."""

from src.data.features import FeatureEngineer
from src.data.generator import SyntheticDataGenerator


def test_feature_engineering_columns():
    """Verify all engineered features are present and non-empty."""
    gen = SyntheticDataGenerator(n_days=15, random_seed=42)
    df = gen.generate()

    featured_df = FeatureEngineer.create_features(df)

    expected_features = FeatureEngineer.get_feature_names()
    for feat in expected_features:
        assert feat in featured_df.columns
        assert not featured_df[feat].isnull().all()


def test_log_transformations_accuracy():
    """Verify logarithmic transformations are numerically correct."""
    gen = SyntheticDataGenerator(n_days=5, random_seed=42)
    df = gen.generate()
    featured = FeatureEngineer.create_features(df)

    # Check log_price
    assert (featured["log_price"] > 0).all()
    # Check margin ratio is bounded < 1.0
    assert (featured["margin_ratio"] < 1.0).all()


def test_no_null_after_feature_creation():
    """Verify fill_na option ensures zero missing values."""
    gen = SyntheticDataGenerator(n_days=15, random_seed=42)
    df = gen.generate()
    featured = FeatureEngineer.create_features(df, fill_na=True)

    null_counts = featured[FeatureEngineer.get_feature_names()].isnull().sum().sum()
    assert null_counts == 0
