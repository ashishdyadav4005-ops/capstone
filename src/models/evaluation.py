"""Model evaluation, calibration analysis, subgroup error breakdowns, and elasticity bias reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.baselines import NaiveDemandModel
from src.models.causal_dml import DoubleMachineLearningEstimator


class ModelEvaluator:
    """Evaluates demand models and causal elasticity estimators on held-out test periods."""

    @staticmethod
    def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Compute standard regression metrics (MAE, RMSE, MAPE, R2)."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        # Safe MAPE calculation avoiding division by zero
        non_zero_mask = y_true > 0
        if np.sum(non_zero_mask) > 0:
            mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0)
        else:
            mape = 0.0

        r2 = float(r2_score(y_true, y_pred))

        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "r2": round(r2, 4),
        }

    @classmethod
    def evaluate_demand_model(
        cls,
        model: Any,
        test_df: pd.DataFrame,
        target_col: str = "quantity",
    ) -> dict[str, Any]:
        """Evaluate out-of-sample demand predictions."""
        if hasattr(model, "predict_quantity"):
            y_pred = model.predict_quantity(test_df)
        elif hasattr(model, "predict"):
            log_preds = model.predict(test_df)
            y_pred = np.clip(np.exp(log_preds) - 1.0, 0.0, None)
        else:
            raise ValueError("Model does not implement predict or predict_quantity method.")

        y_true = test_df[target_col].values
        overall_metrics = cls.compute_regression_metrics(y_true, y_pred)

        return {
            "overall_metrics": overall_metrics,
            "predictions_summary": {
                "mean_actual": float(round(np.mean(y_true), 2)),
                "mean_predicted": float(round(np.mean(y_pred), 2)),
                "sample_size": len(test_df),
            },
        }

    @classmethod
    def evaluate_calibration(
        cls,
        dml_model: DoubleMachineLearningEstimator,
        test_df: pd.DataFrame,
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """Analyze calibration (predicted vs actual) and 95% confidence interval coverage."""
        predictions: list[float] = []
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        actuals = test_df["quantity"].values

        for _, row in test_df.iterrows():
            pred_dict = dml_model.predict_counterfactual_demand(
                base_demand=float(row.get("rolling_demand_7d", row["quantity"])),
                base_price=float(row.get("price_lag_1", row["price"])),
                candidate_price=float(row["price"]),
                product_id=str(row["product_id"]),
                customer_group=str(row["customer_group"]),
            )
            predictions.append(pred_dict["expected_demand"])
            lower_bounds.append(pred_dict["demand_ci_lower"])
            upper_bounds.append(pred_dict["demand_ci_upper"])

        preds_arr = np.array(predictions)
        low_arr = np.array(lower_bounds)
        up_arr = np.array(upper_bounds)

        # Empirical Coverage: fraction of actual values within 95% CI
        covered = ((actuals >= low_arr) & (actuals <= up_arr)).mean() * 100.0

        # Binning for calibration curves
        df_cal = pd.DataFrame({"actual": actuals, "predicted": preds_arr})
        df_cal["bin"] = pd.qcut(df_cal["predicted"], q=min(n_bins, df_cal["predicted"].nunique()), duplicates="drop")

        cal_table = df_cal.groupby("bin", observed=True).agg({
            "predicted": "mean",
            "actual": "mean",
        }).reset_index()

        calibration_points = [
            {"bin": str(row["bin"]), "pred_mean": round(float(row["predicted"]), 2), "actual_mean": round(float(row["actual"]), 2)}
            for _, row in cal_table.iterrows()
        ]

        return {
            "empirical_coverage_95": round(float(covered), 2),
            "mean_prediction_interval_width": round(float(np.mean(up_arr - low_arr)), 2),
            "calibration_curve": calibration_points,
        }

    @classmethod
    def evaluate_subgroup_performance(
        cls,
        model: Any,
        test_df: pd.DataFrame,
        group_by_col: str = "customer_group",
    ) -> list[dict[str, Any]]:
        """Compute error metrics broken down by customer segment or product."""
        results: list[dict[str, Any]] = []

        for grp, sub_df in test_df.groupby(group_by_col):
            if hasattr(model, "predict_quantity"):
                y_pred = model.predict_quantity(sub_df)
            else:
                log_preds = model.predict(sub_df)
                y_pred = np.clip(np.exp(log_preds) - 1.0, 0.0, None)

            y_true = sub_df["quantity"].values
            metrics = cls.compute_regression_metrics(y_true, y_pred)

            results.append({
                "subgroup": str(grp),
                "sample_size": len(sub_df),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "r2": metrics["r2"],
            })

        return results

    @classmethod
    def compare_elasticity_bias(
        cls,
        dml_model: DoubleMachineLearningEstimator,
        naive_model: NaiveDemandModel,
    ) -> pd.DataFrame:
        """Construct comparative elasticity table between Naive OLS, Causal DML, and Ground Truth."""
        comparison_rows: list[dict[str, Any]] = []

        for key, est in dml_model.elasticities.items():
            naive_val = naive_model.naive_elasticities.get(key, -0.5)
            gt_val = est.ground_truth_elasticity or -1.5

            causal_bias = abs(est.point_estimate - gt_val)
            naive_bias = abs(naive_val - gt_val)
            bias_reduction_pct = (
                ((naive_bias - causal_bias) / naive_bias) * 100.0 if naive_bias > 0 else 0.0
            )

            comparison_rows.append({
                "product_id": est.product_id,
                "customer_group": est.customer_group,
                "ground_truth_elasticity": round(gt_val, 4),
                "naive_elasticity": round(naive_val, 4),
                "causal_dml_elasticity": round(est.point_estimate, 4),
                "ci_95_range": f"[{est.ci_lower_95:.2f}, {est.ci_upper_95:.2f}]",
                "naive_bias": round(naive_bias, 4),
                "causal_bias": round(causal_bias, 4),
                "bias_reduction_pct": round(bias_reduction_pct, 1),
            })

        return pd.DataFrame(comparison_rows)
