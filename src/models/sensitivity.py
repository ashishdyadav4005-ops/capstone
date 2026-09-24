"""Sensitivity analysis to quantify the robustness of causal elasticity estimates to unobserved confounding."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.causal_dml import DoubleMachineLearningEstimator


class SensitivityAnalyzer:
    """Evaluates how unobserved confounders and omitted variables impact elasticity estimates."""

    @classmethod
    def evaluate_omitted_confounder(
        cls,
        df: pd.DataFrame,
        all_confounder_cols: list[str],
        omitted_col: str = "holiday",
        treatment_col: str = "log_price",
        outcome_col: str = "log_quantity",
    ) -> dict[str, Any]:
        """Compare DML elasticity estimates when a known confounder is included vs omitted."""
        # 1. Full Model (All observed confounders)
        dml_full = DoubleMachineLearningEstimator(n_splits=3, y_max_iter=100, t_max_iter=100)
        dml_full.fit(df, confounder_cols=all_confounder_cols, treatment_col=treatment_col, outcome_col=outcome_col)

        # 2. Reduced Model (Confounder omitted)
        reduced_cols = [c for c in all_confounder_cols if c != omitted_col]
        dml_reduced = DoubleMachineLearningEstimator(n_splits=3, y_max_iter=100, t_max_iter=100)
        dml_reduced.fit(df, confounder_cols=reduced_cols, treatment_col=treatment_col, outcome_col=outcome_col)

        full_elasticity = dml_full.overall_elasticity
        reduced_elasticity = dml_reduced.overall_elasticity
        shift = reduced_elasticity - full_elasticity
        shift_pct = (shift / abs(full_elasticity)) * 100.0 if full_elasticity != 0 else 0.0

        return {
            "omitted_variable": omitted_col,
            "full_model_elasticity": float(round(full_elasticity, 4)),
            "omitted_model_elasticity": float(round(reduced_elasticity, 4)),
            "absolute_shift": float(round(shift, 4)),
            "percentage_shift": float(round(shift_pct, 2)),
            "remains_negative": bool(reduced_elasticity < 0),
        }

    @classmethod
    def stress_test_synthetic_confounder(
        cls,
        df: pd.DataFrame,
        confounder_cols: list[str],
        strengths: list[float] | None = None,
        random_seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Inject synthetic unobserved confounder shocks of increasing strength and evaluate elasticity stability."""
        strengths = strengths or [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
        rng = np.random.default_rng(random_seed)

        results: list[dict[str, Any]] = []

        # Base DML estimate
        dml_base = DoubleMachineLearningEstimator(n_splits=3, y_max_iter=80, t_max_iter=80)
        dml_base.fit(df, confounder_cols=confounder_cols)
        baseline_beta = dml_base.overall_elasticity

        for gamma in strengths:
            # Add synthetic confounder that correlates with both price and demand
            df_stress = df.copy()
            u_shock = rng.normal(0, 1.0, size=len(df))

            # Perturb price and demand by gamma * u_shock
            df_stress["log_price"] = df_stress["log_price"] + gamma * 0.20 * u_shock
            df_stress["log_quantity"] = df_stress["log_quantity"] + gamma * 0.35 * u_shock

            # Re-estimate DML without including u_shock in confounders
            dml_stress = DoubleMachineLearningEstimator(n_splits=3, y_max_iter=80, t_max_iter=80)
            dml_stress.fit(df_stress, confounder_cols=confounder_cols)

            est_beta = dml_stress.overall_elasticity
            diff = est_beta - baseline_beta

            results.append({
                "confounder_strength": float(gamma),
                "estimated_elasticity": float(round(est_beta, 4)),
                "baseline_elasticity": float(round(baseline_beta, 4)),
                "elasticity_bias": float(round(diff, 4)),
                "is_economically_valid": bool(est_beta < 0),
            })

        return results
