"""Unit tests for sensitivity analysis against unobserved confounding."""

from src.data.features import FeatureEngineer
from src.data.generator import SyntheticDataGenerator
from src.models.sensitivity import SensitivityAnalyzer


def test_omitted_confounder_sensitivity():
    gen = SyntheticDataGenerator(n_days=20, random_seed=42)
    df = FeatureEngineer.create_features(gen.generate())
    features = FeatureEngineer.get_feature_names()

    res = SensitivityAnalyzer.evaluate_omitted_confounder(
        df, all_confounder_cols=features, omitted_col="holiday"
    )

    assert "full_model_elasticity" in res
    assert "omitted_model_elasticity" in res
    assert "percentage_shift" in res
    assert res["remains_negative"] is True


def test_synthetic_confounder_stress_test():
    gen = SyntheticDataGenerator(n_days=15, random_seed=42)
    df = FeatureEngineer.create_features(gen.generate())
    features = FeatureEngineer.get_feature_names()

    stress_results = SensitivityAnalyzer.stress_test_synthetic_confounder(
        df, confounder_cols=features, strengths=[0.0, 0.25, 0.50]
    )

    assert len(stress_results) == 3
    for step in stress_results:
        assert "confounder_strength" in step
        assert "estimated_elasticity" in step
        assert "is_economically_valid" in step
