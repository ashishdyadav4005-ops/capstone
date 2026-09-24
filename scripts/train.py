"""CLI Script: Train baseline models and Double Machine Learning causal elasticity models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path for standalone script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.config import get_config
from src.common.logger import get_logger, setup_logging
from src.data.features import FeatureEngineer
from src.data.repository import DataRepository
from src.data.split import TimeSeriesSplitter
from src.models.baselines import CostPlusPricingModel, LastPriceModel, NaiveDemandModel
from src.models.causal_dml import DoubleMachineLearningEstimator
from src.models.evaluation import ModelEvaluator
from src.models.explainability import ModelExplainer
from src.models.mlflow_tracker import MLflowTracker
from src.models.sensitivity import SensitivityAnalyzer

logger = get_logger("scripts.train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train dynamic pricing models (Baselines + DML)")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/pricing_dataset.parquet",
        help="Path to processed feature-engineered data",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-fitting folds for DML (default: 5)",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="data/models/causal_dml_model.pkl",
        help="Target output path for trained model artifact",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()
    setup_logging(log_level=config.logging.level, json_format=(config.logging.format == "json"))

    logger.info("Initializing model training pipeline", extra={"extra_data": {"data_path": args.data_path}})

    # 1. Load Dataset
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.warning(f"Data file not found at {data_path}. Running generator first...")
        from src.data.generator import SyntheticDataGenerator
        from src.data.validation import DataValidator
        raw_df = SyntheticDataGenerator(n_days=365, random_seed=42).generate()
        cleaned_df = DataValidator.clean(raw_df)
        df = FeatureEngineer.create_features(cleaned_df)
        DataRepository.save_parquet(df, data_path)
    else:
        df = DataRepository.load_parquet(data_path)

    # 2. Chronological Train / Val / Test Partition
    train_df, val_df, test_df = TimeSeriesSplitter.split(df)
    logger.info("Split data chronologically into train/val/test", extra={
        "extra_data": {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    })

    feature_cols = FeatureEngineer.get_feature_names()

    # 3. Train Baselines
    logger.info("Training baseline pricing heuristics and naive demand model...")
    _ = CostPlusPricingModel(default_markup=0.30)
    _ = LastPriceModel()

    naive_model = NaiveDemandModel()
    naive_model.fit(train_df, feature_cols=feature_cols)

    # 4. Train Causal DML Model
    logger.info(f"Training Double Machine Learning estimator with {args.n_splits}-fold cross-fitting...")
    dml_model = DoubleMachineLearningEstimator(n_splits=args.n_splits, random_state=42)
    dml_model.fit(train_df, confounder_cols=feature_cols)

    # 5. Out-of-Sample Evaluation
    logger.info("Evaluating models on held-out test period...")
    naive_eval = ModelEvaluator.evaluate_demand_model(naive_model, test_df)
    calibration_eval = ModelEvaluator.evaluate_calibration(dml_model, test_df)
    _ = ModelEvaluator.evaluate_subgroup_performance(naive_model, test_df)
    bias_df = ModelEvaluator.compare_elasticity_bias(dml_model, naive_model)

    # 6. Sensitivity Analysis
    logger.info("Running sensitivity analysis against unobserved confounding...")
    omitted_res = SensitivityAnalyzer.evaluate_omitted_confounder(
        train_df, all_confounder_cols=feature_cols, omitted_col="holiday"
    )
    stress_res = SensitivityAnalyzer.stress_test_synthetic_confounder(
        train_df, confounder_cols=feature_cols
    )

    # 7. Explainability
    logger.info("Computing demand driver feature importances...")
    importance_df = ModelExplainer.compute_feature_importance(dml_model, val_df)
    top_drivers = ModelExplainer.get_top_demand_drivers(importance_df, top_k=5)

    # 8. MLflow Tracking & Serialization
    tracker = MLflowTracker()
    metrics = {
        "test_mae": naive_eval["overall_metrics"]["mae"],
        "test_rmse": naive_eval["overall_metrics"]["rmse"],
        "test_mape": naive_eval["overall_metrics"]["mape"],
        "test_r2": naive_eval["overall_metrics"]["r2"],
        "calibration_coverage_95": calibration_eval["empirical_coverage_95"],
        "mean_elasticity_causal": float(bias_df["causal_dml_elasticity"].mean()),
        "mean_elasticity_naive": float(bias_df["naive_elasticity"].mean()),
        "mean_bias_reduction_pct": float(bias_df["bias_reduction_pct"].mean()),
    }

    params = {
        "n_splits": args.n_splits,
        "n_estimators": 150,
        "learning_rate": 0.05,
        "confounders_count": len(feature_cols),
    }

    artifacts = {
        "elasticity_comparison": bias_df,
        "feature_importances": importance_df,
        "sensitivity_stress_test": pd.DataFrame(stress_res),
    }

    tracker.log_training_run(
        run_name="causal_dml_production",
        params=params,
        metrics=metrics,
        artifacts=artifacts,
        model_obj=dml_model,
        model_artifact_path=args.model_output,
    )

    # 9. Output Summary Report to Console
    print("=================================================================")
    print(" BDS-39 MODEL TRAINING & CAUSAL EVALUATION SUMMARY")
    print("=================================================================")
    print(f" - Out-of-Sample Demand Test MAE:   {metrics['test_mae']:.4f}")
    print(f" - Out-of-Sample Demand Test MAPE:  {metrics['test_mape']:.2f}%")
    print(f" - 95% Interval Empirical Coverage: {metrics['calibration_coverage_95']:.1f}%")
    print(f" - Avg Causal Elasticity Bias Drop: {metrics['mean_bias_reduction_pct']:.1f}% reduction vs naive")
    print("\n--- ELASTICITY COMPARISON (SAMPLE) ---")
    print(bias_df[["product_id", "customer_group", "ground_truth_elasticity", "naive_elasticity", "causal_dml_elasticity", "bias_reduction_pct"]].head(8).to_string(index=False))
    print("\n--- TOP DEMAND DRIVERS ---")
    for d in top_drivers:
        print(f" • {d['feature']:<22} (Score: {d['importance_score']:.4f}) - {d['description']}")
    print("\n--- SENSITIVITY CHECK (OMITTED HOLIDAY CONFOUNDER) ---")
    print(f" • Baseline Elasticity: {omitted_res['full_model_elasticity']:.4f} -> Shift with Omitted Factor: {omitted_res['omitted_model_elasticity']:.4f} ({omitted_res['percentage_shift']:+.1f}%)")
    print("=================================================================")


if __name__ == "__main__":
    main()
