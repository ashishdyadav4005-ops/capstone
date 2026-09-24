"""Unit tests for MLflow tracker and local artifact persistence."""

import tempfile
from pathlib import Path

from src.models.causal_dml import DoubleMachineLearningEstimator
from src.models.mlflow_tracker import MLflowTracker


def test_local_model_serialization_and_loading():
    dml = DoubleMachineLearningEstimator(n_splits=2)
    dml.overall_elasticity = -1.75

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.pkl"

        tracker = MLflowTracker(tracking_uri="sqlite:///:memory:")
        run_id = tracker.log_training_run(
            run_name="test_run",
            params={"n_splits": 2},
            metrics={"test_metric": 0.95},
            model_obj=dml,
            model_artifact_path=str(model_path),
        )

        assert run_id is not None
        assert model_path.exists()

        loaded_dml = MLflowTracker.load_local_model(str(model_path))
        assert loaded_dml.overall_elasticity == -1.75
