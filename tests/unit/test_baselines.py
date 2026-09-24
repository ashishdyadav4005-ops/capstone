"""Unit tests for baseline pricing models and naive demand regression."""

from src.data.features import FeatureEngineer
from src.data.generator import SyntheticDataGenerator
from src.models.baselines import CostPlusPricingModel, LastPriceModel, NaiveDemandModel


def test_cost_plus_pricing_model():
    model = CostPlusPricingModel(default_markup=0.25)
    rec_price = model.predict_price(cost=40.0)
    assert rec_price == 50.0

    rec_custom = model.predict_price(cost=40.0, markup=0.50)
    assert rec_custom == 60.0


def test_last_price_model():
    model = LastPriceModel()
    price = model.predict_price(current_price=79.99)
    assert price == 79.99


def test_naive_demand_model_fit_and_predict():
    gen = SyntheticDataGenerator(n_days=15, random_seed=42)
    df = FeatureEngineer.create_features(gen.generate())
    features = FeatureEngineer.get_feature_names()

    model = NaiveDemandModel(max_iter=30)
    model.fit(df, feature_cols=features)

    assert len(model.naive_elasticities) > 0

    preds = model.predict(df)
    assert len(preds) == len(df)

    quantities = model.predict_quantity(df)
    assert len(quantities) == len(df)
    assert (quantities >= 0).all()
