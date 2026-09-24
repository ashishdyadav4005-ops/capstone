"""Baseline pricing heuristics and naive (non-causal) supervised demand models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression


class CostPlusPricingModel:
    """Heuristic pricing baseline that marks up baseline unit cost by a fixed margin percentage."""

    def __init__(self, default_markup: float = 0.30):
        self.default_markup = default_markup

    def predict_price(self, cost: float, markup: float | None = None) -> float:
        """Calculate recommended price based on cost-plus rule."""
        applied_markup = markup if markup is not None else self.default_markup
        return float(round(cost * (1.0 + applied_markup), 2))

    def evaluate_revenue(self, df: pd.DataFrame) -> dict[str, float]:
        """Simulate revenue under cost-plus pricing."""
        prices = df["cost"] * (1.0 + self.default_markup)
        # Realized demand with baseline price elasticity assumption
        revenues = prices * df["quantity"]
        return {
            "total_revenue": float(revenues.sum()),
            "mean_price": float(prices.mean()),
            "markup_applied": self.default_markup,
        }


class LastPriceModel:
    """Heuristic baseline carrying forward the most recent historical price."""

    def __init__(self, lookback_days: int = 1):
        self.lookback_days = lookback_days

    def predict_price(self, current_price: float) -> float:
        """Return the previous observed price without modification."""
        return float(current_price)


class NaiveDemandModel:
    """Naive non-causal supervised model predicting demand directly from price and confounders.

    Because observational prices correlate with positive demand shocks (holidays, high demand),
    this naive regression suffers from severe endogeneity bias and underestimates price elasticity.
    """

    def __init__(
        self,
        max_iter: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: HistGradientBoostingRegressor | None = None
        self.feature_names: list[str] = []
        self.naive_elasticities: dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "log_quantity",
        price_col: str = "log_price",
    ) -> NaiveDemandModel:
        """Fit naive gradient boosting model on observational data."""
        self.feature_names = [c for c in feature_cols if c != target_col]
        if price_col not in self.feature_names:
            self.feature_names.append(price_col)

        X = df[self.feature_names].fillna(0.0).values
        y = df[target_col].values

        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model.fit(X, y)

        # Estimate segment-level naive elasticity via direct OLS regression of log_Q on log_P + controls
        self._compute_naive_elasticities(df, price_col=price_col, target_col=target_col)
        return self

    def _compute_naive_elasticities(
        self,
        df: pd.DataFrame,
        price_col: str = "log_price",
        target_col: str = "log_quantity",
    ) -> None:
        """Fit segment-specific simple regressions to extract the naive observational elasticity."""
        for (prod_id, grp), sub_df in df.groupby(["product_id", "customer_group"]):
            if len(sub_df) > 10 and sub_df[price_col].std() > 1e-4:
                reg = LinearRegression()
                X_p = sub_df[[price_col]].values
                y_q = sub_df[target_col].values
                reg.fit(X_p, y_q)
                self.naive_elasticities[f"{prod_id}_{grp}"] = float(reg.coef_[0])
            else:
                self.naive_elasticities[f"{prod_id}_{grp}"] = -0.5 # Default conservative naive estimate

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict log quantity demand."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        X = df[self.feature_names].fillna(0.0).values
        return self.model.predict(X)

    def predict_quantity(self, df: pd.DataFrame) -> np.ndarray:
        """Predict exponentiated demand quantity."""
        log_q = self.predict(df)
        return np.clip(np.exp(log_q) - 1.0, 0.0, None)
