"""MLflow experiment tracking, artifact storage, and model registry manager."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import get_config
from src.common.logger import get_logger

logger = get_logger("src.models.mlflow_tracker")


class MLflowTracker:
    """Manages MLflow runs, metric logging, model artifact serialization, and production staging."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ):
        config = get_config()
        self.tracking_uri = tracking_uri or config.mlflow.tracking_uri
        self.experiment_name = experiment_name or config.mlflow.experiment_name
        self._mlflow_available = False

        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_available = True
            logger.info("MLflow tracking initialized", extra={"extra_data": {"tracking_uri": self.tracking_uri}})
        except Exception as e:
            logger.warning(f"MLflow connection unavailable, operating in local fallback mode: {e}")
            self._mlflow_available = False

    def log_training_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: dict[str, Any] | None = None,
        model_obj: Any | None = None,
        model_artifact_path: str = "data/models/causal_dml_model.pkl",
    ) -> str | None:
        """Log parameters, evaluation metrics, and model artifact to MLflow and local filesystem."""
        # 1. Always save local artifact copy for reliability
        local_path = Path(model_artifact_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if model_obj is not None:
            with open(local_path, "wb") as f:
                pickle.dump(model_obj, f)
            logger.info(f"Saved local model artifact to {local_path}")

        # 2. Log to MLflow if available
        if not self._mlflow_available:
            logger.info("Local run saved (MLflow tracking offline).")
            return "local_run_id"

        try:
            import mlflow

            with mlflow.start_run(run_name=run_name) as run:
                # Log hyperparameters
                for k, v in params.items():
                    mlflow.log_param(k, v)

                # Log evaluation metrics
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))

                # Log artifact files/tables
                if artifacts:
                    for art_name, art_val in artifacts.items():
                        if isinstance(art_val, pd.DataFrame):
                            csv_path = local_path.parent / f"{art_name}.csv"
                            art_val.to_csv(csv_path, index=False)
                            mlflow.log_artifact(str(csv_path))
                        elif isinstance(art_val, dict):
                            json_path = local_path.parent / f"{art_name}.json"
                            with open(json_path, "w", encoding="utf-8") as jf:
                                json.dump(art_val, jf, indent=2)
                            mlflow.log_artifact(str(json_path))

                # Log model binary artifact
                if model_obj is not None and local_path.exists():
                    mlflow.log_artifact(str(local_path), artifact_path="model")

                run_id = run.info.run_id
                logger.info(f"Successfully logged MLflow run: {run_id}")
                return run_id
        except Exception as e:
            logger.warning(f"Error logging to MLflow: {e}")
            return None

    @classmethod
    def load_local_model(cls, model_path: str = "data/models/causal_dml_model.pkl") -> Any:
        """Load a persisted model artifact from local storage."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
