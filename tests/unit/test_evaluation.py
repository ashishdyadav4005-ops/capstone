"""Unit tests for model evaluation, calibration, and subgroup performance metrics."""

import numpy as np
import pandas as pd

from src.data.features import FeatureEngineer
from src.data.generator import SyntheticDataGenerator
from src.models.baselines import NaiveDemandModel
from src.models.causal_dml import DoubleMachineLearningEstimator
from src.models.evaluation import ModelEvaluator


def test_regression_metrics_calculation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 31.0, 38.0])

    metrics = ModelEvaluator.compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["mape"] > 0
    assert metrics["r2"] > 0.90


def test_evaluate_demand_model():
    gen = SyntheticDataGenerator(n_days=15, random_seed=42)
    df = FeatureEngineer.create_features(gen.generate())
    features = FeatureEngineer.get_feature_names()

    model = NaiveDemandModel(max_iter=20)
    model.fit(df, feature_cols=features)

    res = ModelEvaluator.evaluate_demand_model(model, df)
    assert "overall_metrics" in res
    assert res["overall_metrics"]["mae"] > 0


def test_evaluate_calibration_coverage():
    gen = SyntheticDataGenerator(n_days=20, random_seed=42)
    df = FeatureEngineer.create_features(gen.generate())
    features = FeatureEngineer.get_feature_names()

    dml = DoubleMachineLearningEstimator(n_splits=2, y_max_iter=20, t_max_iter=20)
    dml.fit(df, confounder_cols=features)

    cal_res = ModelEvaluator.evaluate_calibration(dml, df, n_bins=5)
    assert "empirical_coverage_95" in cal_res
    assert 0 <= cal_res["empirical_coverage_95"] <= 100.0
    assert len(cal_res["calibration_curve"]) > 0


def test_subgroup_performance_breakdown():
    gen = SyntheticDataGenerator(n_days=15, random_seed=42)
    df = FeatureEngineer.create_features(gen.generate())
    features = FeatureEngineer.get_feature_names()

    model = NaiveDemandModel(max_iter=20)
    model.fit(df, feature_cols=features)

    sub_res = ModelEvaluator.evaluate_subgroup_performance(model, df, group_by_col="customer_group")
    assert len(sub_res) == 4
    for entry in sub_res:
        assert entry["subgroup"] in ["budget", "standard", "premium", "enterprise"]
        assert entry["sample_size"] > 0


def test_elasticity_bias_comparison():
    gen = SyntheticDataGenerator(n_days=20, random_seed=42)
    df = FeatureEngineer.create_features(gen.generate())
    features = FeatureEngineer.get_feature_names()

    naive = NaiveDemandModel(max_iter=20)
    naive.fit(df, feature_cols=features)

    dml = DoubleMachineLearningEstimator(n_splits=2, y_max_iter=20, t_max_iter=20)
    dml.fit(df, confounder_cols=features)

    bias_df = ModelEvaluator.compare_elasticity_bias(dml, naive)
    assert isinstance(bias_df, pd.DataFrame)
    assert "ground_truth_elasticity" in bias_df.columns
    assert "causal_dml_elasticity" in bias_df.columns
    assert "naive_elasticity" in bias_df.columns
    assert "bias_reduction_pct" in bias_df.columns
