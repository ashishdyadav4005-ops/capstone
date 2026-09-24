"""Time-based chronological dataset splitting with strict temporal leakage verification."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field


class SplitVerificationResult(BaseModel):
    """Detailed result of chronological split and leakage verification."""
    is_leakage_free: bool
    train_dates: tuple[str, str]
    val_dates: tuple[str, str]
    test_dates: tuple[str, str]
    train_rows: int
    val_rows: int
    test_rows: int
    violations: list[str] = Field(default_factory=list)


class TimeSeriesSplitter:
    """Performs leakage-free chronological train/val/test splits."""

    @classmethod
    def split(
        cls,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train, validation, and test sets strictly by calendar date."""
        if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5:
            raise ValueError(f"Ratios must sum to 1.0. Got {train_ratio + val_ratio + test_ratio}")

        unique_dates = sorted(df["date"].unique())
        n_dates = len(unique_dates)

        if n_dates < 3:
            raise ValueError(f"Need at least 3 distinct dates to create time splits. Got {n_dates}")

        train_cutoff_idx = int(n_dates * train_ratio)
        val_cutoff_idx = int(n_dates * (train_ratio + val_ratio))

        train_dates = set(unique_dates[:train_cutoff_idx])
        val_dates = set(unique_dates[train_cutoff_idx:val_cutoff_idx])
        test_dates = set(unique_dates[val_cutoff_idx:])

        train_df = df[df["date"].isin(train_dates)].copy().reset_index(drop=True)
        val_df = df[df["date"].isin(val_dates)].copy().reset_index(drop=True)
        test_df = df[df["date"].isin(test_dates)].copy().reset_index(drop=True)

        return train_df, val_df, test_df

    @classmethod
    def verify_no_leakage(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> SplitVerificationResult:
        """Assert and verify that no future information is leaked across splits."""
        violations: list[str] = []

        max_train_date = str(train_df["date"].max())
        min_val_date = str(val_df["date"].min())
        max_val_date = str(val_df["date"].max())
        min_test_date = str(test_df["date"].min())

        # 1. Temporal Monotonicity Check
        if max_train_date >= min_val_date:
            violations.append(
                f"Train/Val date overlap or inversion: max_train_date ({max_train_date}) >= min_val_date ({min_val_date})"
            )

        if max_val_date >= min_test_date:
            violations.append(
                f"Val/Test date overlap or inversion: max_val_date ({max_val_date}) >= min_test_date ({min_test_date})"
            )

        # 2. Date Set Disjointness
        train_dates = set(train_df["date"].unique())
        val_dates = set(val_df["date"].unique())
        test_dates = set(test_df["date"].unique())

        if train_dates.intersection(val_dates):
            violations.append("Train and Validation sets share identical dates.")
        if val_dates.intersection(test_dates):
            violations.append("Validation and Test sets share identical dates.")
        if train_dates.intersection(test_dates):
            violations.append("Train and Test sets share identical dates.")

        is_clean = len(violations) == 0

        return SplitVerificationResult(
            is_leakage_free=is_clean,
            train_dates=(str(train_df["date"].min()), max_train_date),
            val_dates=(min_val_date, max_val_date),
            test_dates=(min_test_date, str(test_df["date"].max())),
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
            violations=violations,
        )
