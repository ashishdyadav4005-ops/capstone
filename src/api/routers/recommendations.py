"""Recommendations Lifecycle API Router: Transitions, manager overrides, and approvals."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_lifecycle_manager, get_recommendation_repository
from src.api.lifecycle import (
    IllegalStateTransitionError,
    RecommendationLifecycleManager,
)
from src.api.repository import RecommendationRepository
from src.api.schemas import (
    ActionRequest,
    OverrideRequest,
    RecommendationRecord,
    RecommendationState,
)

router = APIRouter(prefix="/api/v1/recommendations", tags=["Recommendation Lifecycle"])


@router.get("", response_model=list[RecommendationRecord])
def list_recommendations(
    product_id: str | None = Query(None, description="Filter by product ID"),
    customer_group: str | None = Query(None, description="Filter by customer group"),
    location_id: str | None = Query(None, description="Filter by location ID"),
    state: RecommendationState | None = Query(None, description="Filter by lifecycle state"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    repo: RecommendationRepository = Depends(get_recommendation_repository),
) -> list[RecommendationRecord]:
    """List price recommendations with optional status and segment filters."""
    return repo.list_recommendations(
        product_id=product_id,
        customer_group=customer_group,
        location_id=location_id,
        state=state,
        limit=limit,
        offset=offset,
    )


@router.get("/{recommendation_id}", response_model=RecommendationRecord)
def get_recommendation(
    recommendation_id: str,
    repo: RecommendationRepository = Depends(get_recommendation_repository),
) -> RecommendationRecord:
    """Retrieve a single price recommendation by ID."""
    rec = repo.get_by_id(recommendation_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Recommendation '{recommendation_id}' not found.")
    return rec


@router.post("/{recommendation_id}/review", response_model=RecommendationRecord)
def mark_under_review(
    recommendation_id: str,
    req: ActionRequest = ActionRequest(),
    lifecycle: RecommendationLifecycleManager = Depends(get_lifecycle_manager),
) -> RecommendationRecord:
    """Mark a recommendation as UNDER_REVIEW."""
    try:
        return lifecycle.transition(
            recommendation_id=recommendation_id,
            target_state=RecommendationState.UNDER_REVIEW,
            actor=req.actor,
            notes=req.notes,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except IllegalStateTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{recommendation_id}/approve", response_model=RecommendationRecord)
def approve_recommendation(
    recommendation_id: str,
    req: ActionRequest = ActionRequest(),
    lifecycle: RecommendationLifecycleManager = Depends(get_lifecycle_manager),
) -> RecommendationRecord:
    """Approve a recommendation for production deployment."""
    try:
        return lifecycle.transition(
            recommendation_id=recommendation_id,
            target_state=RecommendationState.APPROVED,
            actor=req.actor,
            notes=req.notes,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except IllegalStateTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{recommendation_id}/reject", response_model=RecommendationRecord)
def reject_recommendation(
    recommendation_id: str,
    req: ActionRequest,
    lifecycle: RecommendationLifecycleManager = Depends(get_lifecycle_manager),
) -> RecommendationRecord:
    """Reject a recommendation with a mandatory explanation reason."""
    try:
        return lifecycle.transition(
            recommendation_id=recommendation_id,
            target_state=RecommendationState.REJECTED,
            actor=req.actor,
            notes=req.notes,
            reason=req.reason,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except IllegalStateTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{recommendation_id}/override", response_model=RecommendationRecord)
def override_price(
    recommendation_id: str,
    req: OverrideRequest,
    lifecycle: RecommendationLifecycleManager = Depends(get_lifecycle_manager),
) -> RecommendationRecord:
    """Manager override: manually set a price with mandatory reasoning and guardrail validation."""
    try:
        return lifecycle.override_price(
            recommendation_id=recommendation_id,
            override_price=req.override_price,
            reason=req.reason,
            actor=req.reviewed_by,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{recommendation_id}/publish", response_model=RecommendationRecord)
def publish_recommendation(
    recommendation_id: str,
    req: ActionRequest = ActionRequest(),
    lifecycle: RecommendationLifecycleManager = Depends(get_lifecycle_manager),
) -> RecommendationRecord:
    """Publish an approved price recommendation to the production price catalog."""
    try:
        return lifecycle.transition(
            recommendation_id=recommendation_id,
            target_state=RecommendationState.PUBLISHED,
            actor=req.actor,
            notes=req.notes,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except IllegalStateTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{recommendation_id}/rollback", response_model=RecommendationRecord)
def rollback_recommendation(
    recommendation_id: str,
    req: ActionRequest = ActionRequest(),
    lifecycle: RecommendationLifecycleManager = Depends(get_lifecycle_manager),
) -> RecommendationRecord:
    """Execute an emergency rollback restoring baseline price P0."""
    try:
        return lifecycle.rollback(
            recommendation_id=recommendation_id,
            actor=req.actor,
            notes=req.notes or "Emergency rollback to baseline",
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
