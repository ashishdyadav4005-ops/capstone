"""Health check and service status router."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src import __version__
from src.api.dependencies import get_causal_model, get_recommendation_repository
from src.api.repository import RecommendationRepository
from src.api.schemas import HealthResponse
from src.models.causal_dml import DoubleMachineLearningEstimator

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
@router.get("/api/v1/health", response_model=HealthResponse)
def health_check(
    repo: RecommendationRepository = Depends(get_recommendation_repository),
    model: DoubleMachineLearningEstimator | None = Depends(get_causal_model),
) -> HealthResponse:
    """Return health status, database connectivity, and ML model status."""
    db_status = "connected"
    try:
        with repo._get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception:
        db_status = "error"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        version=__version__,
        database=db_status,
        model_loaded=(model is not None),
    )
