"""Data cleaning and schema validation module using strict type, range, and business rules."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class TransactionSchema(BaseModel):
    """Pydantic model representing a single validated transaction record."""
    date: str
    product_id: str
    location_id: str
    customer_group: str
    price: float = Field(gt=0, description="Price must be strictly positive")
    quantity: int = Field(ge=0, description="Quantity sold must be non-negative")
    capacity: int = Field(gt=0, description="Capacity must be strictly positive")
    cost: float = Field(gt=0, description="Unit cost must be strictly positive")
    holiday: int = Field(ge=0, le=1)
    event: int = Field(ge=0, le=1)
    weather_temp: float
    competitor_price: float = Field(gt=0)


class ValidationResult(BaseModel):
    """Structured report containing validation status, violations, and summary metrics."""
    is_valid: bool
    total_rows: int
    valid_rows: int
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)


class DataValidator:
    """Validates and cleans transaction datasets against integrity and economic consistency rules."""

    REQUIRED_COLUMNS = [
        "date",
        "product_id",
        "location_id",
        "customer_group",
        "price",
        "quantity",
        "capacity",
        "cost",
        "holiday",
        "event",
        "weather_temp",
        "competitor_price",
    ]

    ALLOWED_CUSTOMER_GROUPS = {"budget", "standard", "premium", "enterprise"}

    @classmethod
    def validate(cls, df: pd.DataFrame, strict: bool = False) -> ValidationResult:
        """Run all schema, range, outlier, and business logic validations on DataFrame."""
        errors: list[str] = []
        warnings: list[str] = []

        total_rows = len(df)
        if total_rows == 0:
            return ValidationResult(
                is_valid=False,
                total_rows=0,
                valid_rows=0,
                errors=["Dataset is empty (0 rows)."],
            )

        # 1. Missing Required Columns
        missing_cols = [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(
                is_valid=False,
                total_rows=total_rows,
                valid_rows=0,
                errors=errors,
            )

        # 2. Null Value Check
        null_counts = df[cls.REQUIRED_COLUMNS].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        if cols_with_nulls:
            errors.append(f"Columns contain null values: {cols_with_nulls}")

        # 3. Duplicate Key Check (date, product_id, location_id, customer_group)
        key_cols = ["date", "product_id", "location_id", "customer_group"]
        duplicates = df.duplicated(subset=key_cols).sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate transaction keys on {key_cols}")

        # 4. Data Type and Range Checks
        # Price > 0
        invalid_prices = (df["price"] <= 0).sum()
        if invalid_prices > 0:
            errors.append(f"Found {invalid_prices} rows with non-positive price (<= 0).")

        # Quantity >= 0
        invalid_qty = (df["quantity"] < 0).sum()
        if invalid_qty > 0:
            errors.append(f"Found {invalid_qty} rows with negative quantity (< 0).")

        # Capacity > 0
        invalid_capacity = (df["capacity"] <= 0).sum()
        if invalid_capacity > 0:
            errors.append(f"Found {invalid_capacity} rows with non-positive capacity (<= 0).")

        # Cost > 0
        invalid_cost = (df["cost"] <= 0).sum()
        if invalid_cost > 0:
            errors.append(f"Found {invalid_cost} rows with non-positive cost (<= 0).")

        # Holiday and Event binary checks
        invalid_holiday = (~df["holiday"].isin([0, 1])).sum()
        if invalid_holiday > 0:
            errors.append(f"Found {invalid_holiday} rows with non-binary holiday flag.")

        invalid_event = (~df["event"].isin([0, 1])).sum()
        if invalid_event > 0:
            errors.append(f"Found {invalid_event} rows with non-binary event flag.")

        # Customer group categories
        invalid_groups = (~df["customer_group"].isin(cls.ALLOWED_CUSTOMER_GROUPS)).sum()
        if invalid_groups > 0:
            errors.append(f"Found {invalid_groups} rows with invalid customer_group.")

        # 5. Business Consistency Checks
        # Quantity <= Capacity
        exceeded_capacity = (df["quantity"] > df["capacity"]).sum()
        if exceeded_capacity > 0:
            warnings.append(f"Found {exceeded_capacity} rows where realized quantity exceeded capacity.")

        # Price < Cost warning
        below_cost = (df["price"] < df["cost"]).sum()
        if below_cost > 0:
            warnings.append(f"Found {below_cost} rows where price was below unit cost.")

        # 6. Statistical Outlier Detection (Z-score > 4 on price or quantity)
        price_mean = df["price"].mean()
        price_std = df["price"].std()
        if price_std > 0:
            price_outliers = ((df["price"] - price_mean).abs() / price_std > 4.5).sum()
            if price_outliers > 0:
                warnings.append(f"Detected {price_outliers} statistical price outliers (|z| > 4.5).")

        is_valid = len(errors) == 0 if not strict else (len(errors) == 0 and len(warnings) == 0)

        stats = {
            "num_products": int(df["product_id"].nunique()),
            "num_locations": int(df["location_id"].nunique()),
            "num_customer_groups": int(df["customer_group"].nunique()),
            "date_range": [str(df["date"].min()), str(df["date"].max())],
            "mean_price": float(round(df["price"].mean(), 2)),
            "mean_quantity": float(round(df["quantity"].mean(), 2)),
            "total_revenue": float(round((df["price"] * df["quantity"]).sum(), 2)),
        }

        return ValidationResult(
            is_valid=is_valid,
            total_rows=total_rows,
            valid_rows=total_rows if is_valid else 0,
            errors=errors,
            warnings=warnings,
            statistics=stats,
        )

    @classmethod
    def clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by removing invalid records, standardizing types, and sorting chronologically."""
        df_clean = df.copy()

        # Drop exact duplicates on primary keys
        key_cols = ["date", "product_id", "location_id", "customer_group"]
        if all(c in df_clean.columns for c in key_cols):
            df_clean = df_clean.drop_duplicates(subset=key_cols, keep="first")

        # Filter out invalid numeric rows
        df_clean = df_clean[
            (df_clean["price"] > 0) &
            (df_clean["quantity"] >= 0) &
            (df_clean["capacity"] > 0) &
            (df_clean["cost"] > 0)
        ]

        # Ensure date format and chronological sorting
        df_clean["date"] = pd.to_datetime(df_clean["date"]).dt.strftime("%Y-%m-%d")
        df_clean = df_clean.sort_values(by=["date", "product_id", "location_id", "customer_group"]).reset_index(drop=True)

        return df_clean
