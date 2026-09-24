"""Model explainability module computing feature importances and SHAP values for demand drivers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.models.causal_dml import DoubleMachineLearningEstimator


class ModelExplainer:
    """Provides interpretability and feature importance analysis for demand nuisance models."""

    @classmethod
    def compute_feature_importance(
        cls,
        dml_model: DoubleMachineLearningEstimator,
        val_df: pd.DataFrame,
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compute permutation feature importance on the Stage 1 Outcome (demand) model."""
        if dml_model.outcome_model is None:
            raise ValueError("Outcome model is not fitted.")

        feature_cols = dml_model.confounder_features
        X_val = val_df[feature_cols].fillna(0.0).values
        y_val = val_df["log_quantity"].values

        perm_res = permutation_importance(
            dml_model.outcome_model,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="neg_mean_squared_error",
        )

        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance_mean": np.round(perm_res.importances_mean, 5),
            "importance_std": np.round(perm_res.importances_std, 5),
        }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

        return importance_df

    @classmethod
    def get_top_demand_drivers(
        cls,
        importance_df: pd.DataFrame,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Extract top K influential market drivers with plain-English descriptions."""
        top_features = importance_df.head(top_k)

        driver_descriptions = {
            "holiday": "Official holiday calendar effect",
            "event": "Major sports/entertainment event in market",
            "competitor_price": "Substitute product pricing pressure",
            "price_ratio_competitor": "Relative pricing versus competitors",
            "weather_temp": "Temperature & weather fluctuations",
            "rolling_demand_7d": "Recent 7-day sales momentum",
            "rolling_demand_14d": "Medium-term demand trend",
            "price_lag_1": "Previous day reference price",
            "day_of_week": "Day-of-week demand pattern",
            "is_weekend": "Weekend surge effect",
            "cost": "Procurement / cost baseline",
        }

        results = []
        for _, row in top_features.iterrows():
            feat = row["feature"]
            results.append({
                "feature": feat,
                "importance_score": float(row["importance_mean"]),
                "description": driver_descriptions.get(feat, "Contextual market indicator"),
            })

        return results
