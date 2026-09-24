"""CLI Script: Generate synthetic sales data with confounding, run validation, feature engineering, and store datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path for standalone script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.config import get_config
from src.common.logger import get_logger, setup_logging
from src.data.features import FeatureEngineer
from src.data.generator import SyntheticDataGenerator
from src.data.repository import DataRepository
from src.data.split import TimeSeriesSplitter
from src.data.validation import DataValidator

logger = get_logger("scripts.generate_data")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction data with confounding and ground truth elasticity."
    )
    parser.add_argument(
        "--days", type=int, default=365, help="Number of simulated days (default: 365)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default="data/raw/transactions.parquet",
        help="Target path for raw dataset",
    )
    parser.add_argument(
        "--processed-output",
        type=str,
        default="data/processed/pricing_dataset.parquet",
        help="Target path for processed & feature-engineered dataset",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Optional SQLite database URL (defaults to configs/app.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()
    setup_logging(log_level=config.logging.level, json_format=(config.logging.format == "json"))

    logger.info("Starting synthetic transaction generation pipeline", extra={
        "extra_data": {"days": args.days, "seed": args.seed}
    })

    # 1. Generate Raw Synthetic Data
    generator = SyntheticDataGenerator(n_days=args.days, random_seed=args.seed)
    raw_df = generator.generate()
    logger.info("Generated raw synthetic dataset", extra={
        "extra_data": {"rows": len(raw_df), "columns": len(raw_df.columns)}
    })

    # 2. Data Validation & Quality Checks
    validation_res = DataValidator.validate(raw_df)
    if not validation_res.is_valid:
        logger.error("Data validation failed", extra={"extra_data": {"errors": validation_res.errors}})
        raise ValueError(f"Data validation failed: {validation_res.errors}")
    logger.info("Data validation passed successfully", extra={
        "extra_data": {"statistics": validation_res.statistics, "warnings": validation_res.warnings}
    })

    # 3. Clean and Standardize Data
    cleaned_df = DataValidator.clean(raw_df)

    # 4. Feature Engineering (Strictly Leakage-Free)
    processed_df = FeatureEngineer.create_features(cleaned_df)
    logger.info("Engineered time-series, lag, and ratio features", extra={
        "extra_data": {"features_count": len(processed_df.columns)}
    })

    # 5. Chronological Train / Val / Test Split & Leakage Verification
    train_df, val_df, test_df = TimeSeriesSplitter.split(processed_df)
    leakage_check = TimeSeriesSplitter.verify_no_leakage(train_df, val_df, test_df)
    if not leakage_check.is_leakage_free:
        logger.error("Data leakage detected in chronological split", extra={
            "extra_data": {"violations": leakage_check.violations}
        })
        raise ValueError(f"Data leakage detected: {leakage_check.violations}")
    logger.info("Verified chronological train/val/test split (leakage-free)", extra={
        "extra_data": {
            "train_dates": leakage_check.train_dates,
            "val_dates": leakage_check.val_dates,
            "test_dates": leakage_check.test_dates,
            "train_rows": leakage_check.train_rows,
            "val_rows": leakage_check.val_rows,
            "test_rows": leakage_check.test_rows,
        }
    })

    # 6. Save Datasets to Parquet
    raw_hash = DataRepository.save_parquet(raw_df, args.raw_output)
    processed_hash = DataRepository.save_parquet(processed_df, args.processed_output)
    logger.info("Saved Parquet datasets with provenance hashes", extra={
        "extra_data": {
            "raw_output": args.raw_output,
            "raw_sha256": raw_hash,
            "processed_output": args.processed_output,
            "processed_sha256": processed_hash,
        }
    })

    # 7. Persist to SQLite Database
    db_url = args.db_url or config.database.url
    DataRepository.save_sqlite(processed_df, db_url=db_url, table_name="sales_transactions")
    logger.info("Persisted dataset into SQLite database table 'sales_transactions'", extra={
        "extra_data": {"database_url": db_url}
    })

    print("=================================================================")
    print(" BDS-39 DATA GENERATION PIPELINE COMPLETED")
    print(f" - Raw Data Rows:       {len(raw_df):,}")
    print(f" - Processed Data Rows: {len(processed_df):,}")
    print(f" - Features Created:    {len(processed_df.columns)}")
    print(f" - Raw Hash (SHA-256):  {raw_hash[:16]}...")
    print(f" - Proc Hash (SHA-256): {processed_hash[:16]}...")
    print(f" - Train / Val / Test:  {len(train_df)} / {len(val_df)} / {len(test_df)} rows")
    print(f" - Saved Parquet:       {args.processed_output}")
    print(f" - Saved SQLite DB:     {db_url}")
    print("=================================================================")


if __name__ == "__main__":
    main()
