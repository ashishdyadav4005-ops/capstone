"""Audit API Router: Query audit trail and verify cryptographic chain integrity."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from src.audit.models import AuditEntry
from src.audit.repository import AuditRepository
from src.audit.verification import AuditChainVerifier, ChainVerificationResult
from src.auth.dependencies import get_current_user, require_roles
from src.auth.repository import UserRecord
from src.auth.roles import UserRole

router = APIRouter(prefix="/api/v1/audit", tags=["Cryptographic Audit Trail"])


class AuditSummaryResponse(BaseModel):
    """Statistical summary of audit trail."""

    total_records: int
    latest_entry_hash: str | None
    latest_event_type: str | None
    latest_timestamp: str | None


def get_audit_repository() -> AuditRepository:
    """Provide singleton AuditRepository."""
    return AuditRepository()


def get_audit_verifier(
    repo: AuditRepository = Depends(get_audit_repository),
) -> AuditChainVerifier:
    """Provide singleton AuditChainVerifier."""
    return AuditChainVerifier(repository=repo)


@router.get(
    "/logs",
    response_model=list[dict],
    dependencies=[Depends(require_roles([UserRole.ADMIN, UserRole.GOVERNANCE, UserRole.MANAGER]))],
)
def get_audit_logs(
    event_type: str | None = Query(None, description="Filter by event type"),
    actor_username: str | None = Query(None, description="Filter by actor"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    repo: AuditRepository = Depends(get_audit_repository),
    _user: UserRecord = Depends(get_current_user),
) -> list[dict]:
    """Retrieve audit log entries in reverse chronological order."""
    entries: list[AuditEntry] = repo.list_entries(
        event_type=event_type,
        actor_username=actor_username,
        limit=limit,
        offset=offset,
    )
    return [e.to_dict() for e in entries]


@router.get(
    "/verify",
    response_model=dict,
    dependencies=[Depends(require_roles([UserRole.ADMIN, UserRole.GOVERNANCE]))],
)
def verify_audit_chain(
    verifier: AuditChainVerifier = Depends(get_audit_verifier),
    _user: UserRecord = Depends(get_current_user),
) -> dict:
    """Perform live cryptographic verification of the SHA-256 chained audit trail."""
    res: ChainVerificationResult = verifier.verify_chain()
    return res.to_dict()


@router.get(
    "/summary",
    response_model=AuditSummaryResponse,
    dependencies=[Depends(require_roles([UserRole.ADMIN, UserRole.GOVERNANCE, UserRole.MANAGER]))],
)
def get_audit_summary(
    repo: AuditRepository = Depends(get_audit_repository),
    _user: UserRecord = Depends(get_current_user),
) -> AuditSummaryResponse:
    """Return high-level summary metadata of the audit chain."""
    total = repo.count()
    latest = repo.get_latest_entry()
    return AuditSummaryResponse(
        total_records=total,
        latest_entry_hash=latest.entry_hash if latest else None,
        latest_event_type=latest.event_type.value if latest else None,
        latest_timestamp=latest.timestamp if latest else None,
    )
