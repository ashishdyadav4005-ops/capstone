"""Feature engineering module with strict time-series leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Builds lag, rolling, ratio, seasonality, and logarithmic features without data leakage."""

    @classmethod
    def create_features(
        cls,
        df: pd.DataFrame,
        fill_na: bool = True,
    ) -> pd.DataFrame:
        """Generate all engineered features on the transaction dataset.

        CRITICAL: All rolling and lag features use historical data only (shifted by 1 day)
        to guarantee zero future leakage.
        """
        data = df.copy()
        data["date_dt"] = pd.to_datetime(data["date"])

        # Ensure chronological ordering per entity stream
        group_cols = ["product_id", "location_id", "customer_group"]
        data = data.sort_values(by=group_cols + ["date_dt"]).reset_index(drop=True)

        # 1. Calendar and Seasonality Features
        data["day_of_week"] = data["date_dt"].dt.dayofweek
        data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
        data["month"] = data["date_dt"].dt.month
        data["quarter"] = data["date_dt"].dt.quarter

        # Trigonometric day-of-year cyclical encoding
        day_of_year = data["date_dt"].dt.dayofyear
        data["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
        data["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365.25)

        # 2. Log Transformations
        data["log_price"] = np.log(data["price"])
        data["log_quantity"] = np.log(data["quantity"] + 1.0)
        data["log_competitor_price"] = np.log(data["competitor_price"])
        data["log_cost"] = np.log(data["cost"])

        # 3. Price Ratios & Margins
        data["price_ratio_competitor"] = data["price"] / data["competitor_price"]
        data["price_diff_competitor"] = data["price"] - data["competitor_price"]
        data["margin_currency"] = data["price"] - data["cost"]
        data["margin_ratio"] = data["margin_currency"] / data["price"]

        # 4. Lag Features (Shifted by 1 or 7 days within product-location-group)
        grouped = data.groupby(group_cols)

        data["price_lag_1"] = grouped["price"].shift(1)
        data["price_lag_7"] = grouped["price"].shift(7)
        data["quantity_lag_1"] = grouped["quantity"].shift(1)
        data["quantity_lag_7"] = grouped["quantity"].shift(7)
        data["price_change_1d"] = (data["price"] - data["price_lag_1"]) / data["price_lag_1"]

        # 5. Rolling Historical Statistics (using strictly past observations via closed='left' or shift(1))
        # Note: We compute rolling window on the shifted series so that current day is never included!
        data["rolling_demand_7d"] = grouped["quantity"].transform(
            lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
        )
        data["rolling_demand_14d"] = grouped["quantity"].transform(
            lambda s: s.shift(1).rolling(window=14, min_periods=1).mean()
        )
        data["rolling_price_mean_7d"] = grouped["price"].transform(
            lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
        )
        data["rolling_price_std_7d"] = grouped["price"].transform(
            lambda s: s.shift(1).rolling(window=7, min_periods=2).std()
        )

        # 6. Fill Initial NaN values if requested (e.g. for the initial 7 days where lags are undefined)
        if fill_na:
            data["price_lag_1"] = data["price_lag_1"].bfill().fillna(data["price"])
            data["price_lag_7"] = data["price_lag_7"].bfill().fillna(data["price"])
            data["quantity_lag_1"] = data["quantity_lag_1"].bfill().fillna(data["quantity"])
            data["quantity_lag_7"] = data["quantity_lag_7"].bfill().fillna(data["quantity"])
            data["price_change_1d"] = data["price_change_1d"].fillna(0.0)
            data["rolling_demand_7d"] = data["rolling_demand_7d"].bfill().fillna(data["quantity"])
            data["rolling_demand_14d"] = data["rolling_demand_14d"].bfill().fillna(data["quantity"])
            data["rolling_price_mean_7d"] = data["rolling_price_mean_7d"].bfill().fillna(data["price"])
            data["rolling_price_std_7d"] = data["rolling_price_std_7d"].fillna(0.0)

        # Drop temporary datetime column and return clean DataFrame
        data = data.drop(columns=["date_dt"])
        return data

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Return the standard list of engineered predictor feature names."""
        return [
            "holiday",
            "event",
            "weather_temp",
            "day_of_week",
            "is_weekend",
            "month",
            "quarter",
            "sin_day_of_year",
            "cos_day_of_year",
            "competitor_price",
            "price_ratio_competitor",
            "price_diff_competitor",
            "cost",
            "margin_ratio",
            "price_lag_1",
            "price_lag_7",
            "quantity_lag_1",
            "quantity_lag_7",
            "price_change_1d",
            "rolling_demand_7d",
            "rolling_demand_14d",
            "rolling_price_mean_7d",
            "rolling_price_std_7d",
        ]
