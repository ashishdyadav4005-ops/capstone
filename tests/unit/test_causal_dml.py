"""Unit tests for Double Machine Learning (DML) Causal Elasticity Estimator."""

import numpy as np

from src.data.features import FeatureEngineer
from src.data.generator import SyntheticDataGenerator
from src.models.causal_dml import DoubleMachineLearningEstimator


def test_dml_training_and_elasticity_estimates():
    gen = SyntheticDataGenerator(n_days=25, random_seed=42)
    df = FeatureEngineer.create_features(gen.generate())
    features = FeatureEngineer.get_feature_names()

    dml = DoubleMachineLearningEstimator(n_splits=3, y_max_iter=30, t_max_iter=30, random_state=42)
    dml.fit(df, confounder_cols=features)

    # Global elasticity must be negative
    assert dml.overall_elasticity < 0

    # Segment elasticities must exist and be negative on average
    assert len(dml.elasticities) > 0
    all_estimates = [est.point_estimate for est in dml.elasticities.values()]
    assert np.mean(all_estimates) < 0

    # Check structure of single segment estimate
    sample_est = list(dml.elasticities.values())[0]
    assert sample_est.ci_lower_95 <= sample_est.point_estimate <= sample_est.ci_upper_95
    assert sample_est.std_error > 0


def test_counterfactual_demand_prediction():
    gen = SyntheticDataGenerator(n_days=20, random_seed=42)
    df = FeatureEngineer.create_features(gen.generate())
    features = FeatureEngineer.get_feature_names()

    dml = DoubleMachineLearningEstimator(n_splits=3, y_max_iter=30, t_max_iter=30, random_state=42)
    dml.fit(df, confounder_cols=features)

    # When price increases (+20%), expected demand should decrease (law of demand)
    pred_base = dml.predict_counterfactual_demand(
        base_demand=100.0, base_price=50.0, candidate_price=50.0, product_id="P001", customer_group="standard"
    )
    pred_higher_price = dml.predict_counterfactual_demand(
        base_demand=100.0, base_price=50.0, candidate_price=60.0, product_id="P001", customer_group="standard"
    )
    pred_lower_price = dml.predict_counterfactual_demand(
        base_demand=100.0, base_price=50.0, candidate_price=40.0, product_id="P001", customer_group="standard"
    )

    assert pred_base["expected_demand"] == 100.0
    assert pred_higher_price["expected_demand"] < pred_base["expected_demand"]
    assert pred_lower_price["expected_demand"] > pred_base["expected_demand"]
    assert pred_higher_price["demand_ci_lower"] <= pred_higher_price["expected_demand"] <= pred_higher_price["demand_ci_upper"]


def test_cold_start_fallback():
    dml = DoubleMachineLearningEstimator(n_splits=2)
    dml.overall_elasticity = -1.45

    # Unseen product/group key
    est = dml.get_elasticity(product_id="P_UNSEEN", customer_group="vip_unseen")
    assert est.point_estimate == -1.45
    assert est.ci_lower_95 < -1.45 < est.ci_upper_95
