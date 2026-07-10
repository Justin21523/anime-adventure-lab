from __future__ import annotations

import logging
import hashlib
import os
import re
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.exc import IntegrityError, OperationalError

from core.application import (
    StoryApplicationService,
    StoryTurnInProgressError,
    WorldApplicationService,
    DocumentApplicationService,
    ReviewApplicationService,
    JobMaintenanceService,
    JobEventService,
    SystemStatusService,
)
from core.storage.object_store import get_object_store
from schemas.v2 import (
    V2StorySessionCreate,
    V2StorySessionResponse,
    V2JobResponse,
    V2JobEventResponse,
    V2StoryTurnCreate,
    V2StoryTurnResponse,
    V2DocumentResponse,
    V2DocumentUploadResponse,
    V2CapabilitiesResponse,
    V2WorldCreate,
    V2WorldResponse,
    V2WorldUpdate,
    V2ReviewApprovalResponse,
    V2ReviewProposalCreate,
    V2ReviewProposalResponse,
    V2ReconcileResponse,
    V2SystemStatusResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _world_response(record) -> V2WorldResponse:
    return V2WorldResponse(
        world_id=record.id,
        name=record.name,
        pack=record.pack,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _session_response(record) -> V2StorySessionResponse:
    return V2StorySessionResponse(
        session_id=record.id,
        world_id=record.world_id,
        player_name=record.player_name,
        persona_id=record.persona_id,
        runtime_preset_id=record.runtime_preset_id,
        state=record.state,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _turn_response(record) -> V2StoryTurnResponse:
    return V2StoryTurnResponse(
        turn_id=record.id,
        session_id=record.session_id,
        turn_number=record.turn_number,
        status=record.status,
        player_input=record.player_input,
        narrative=record.narrative,
        choices=record.choices,
        citations=record.citations,
        state_delta=record.state_delta,
        trace=record.trace,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _job_response(record, *, replayed: bool = False) -> V2JobResponse:
    error = None
    if record.error_code or record.error_message:
        error = {
            "code": record.error_code or "JOB_FAILED",
            "message": record.error_message or "Job failed",
        }
    duration_ms = None
    if record.started_at and record.finished_at:
        duration_ms = max(
            0, int((record.finished_at - record.started_at).total_seconds() * 1000)
        )
    return V2JobResponse(
        job_id=record.id,
        kind=record.kind,
        status=record.status,
        progress=record.progress,
        attempt_count=record.attempt_count,
        session_id=record.session_id,
        turn_id=record.turn_id,
        execution_id=record.execution_id,
        request_id=record.request_id,
        dispatch_status=record.dispatch_status,
        lease_expires_at=record.lease_expires_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        duration_ms=duration_ms,
        replayed=replayed,
        result=record.result,
        error=error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _job_event_response(record) -> V2JobEventResponse:
    return V2JobEventResponse(
        event_id=record.id,
        job_id=record.job_id,
        event_type=record.event_type,
        from_status=record.from_status,
        to_status=record.to_status,
        progress=record.progress,
        attempt_count=record.attempt_count,
        execution_id=record.execution_id,
        request_id=record.request_id,
        actor=record.actor,
        details=record.details,
        occurred_at=record.occurred_at,
    )


def _document_response(record) -> V2DocumentResponse:
    return V2DocumentResponse(
        document_id=record.id,
        world_id=record.world_id,
        filename=record.filename,
        content_type=record.content_type,
        checksum=record.checksum,
        status=record.status,
        metadata=record.metadata_json,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _proposal_response(record) -> V2ReviewProposalResponse:
    return V2ReviewProposalResponse(
        proposal_id=record.id,
        world_id=record.world_id,
        session_id=record.session_id,
        status=record.status,
        patch=record.patch,
        reasoning=record.reasoning,
        created_at=record.created_at,
        reviewed_at=record.reviewed_at,
    )


def _dispatch_story_turn(job_id: str) -> None:
    """Dispatch only after the database transaction has committed."""
    from workers.tasks.story_v2 import story_turn_v2_task

    story_turn_v2_task.delay(job_id)


def _dispatch_document_index(job_id: str) -> None:
    from workers.tasks.rag_v2 import document_index_v2_task

    document_index_v2_task.delay(job_id)


def _mark_dispatch(job_id: str, *, deferred: bool = False) -> None:
    try:
        JobMaintenanceService().mark_dispatched(job_id, deferred=deferred)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not persist dispatch status for %s: %s", job_id, exc)


@router.get("/system/capabilities", response_model=V2CapabilitiesResponse)
def capabilities():
    return {
        "api_version": "v2",
        "product": "story-first",
        "persistence": "postgresql+pgvector",
        "object_store": "minio",
        "auth_required": os.getenv("ENABLE_API_AUTH", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        "experimental_features_enabled": os.getenv("ENABLE_EXPERIMENTAL_ROUTERS", "0")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"},
        "experimental_features": ["vlm", "training", "model-export", "postprocess"],
    }


@router.get("/system/status", response_model=V2SystemStatusResponse)
def system_status():
    return SystemStatusService().check()


@router.post(
    "/worlds", response_model=V2WorldResponse, status_code=status.HTTP_201_CREATED
)
def create_world(request: V2WorldCreate):
    try:
        pack = {**request.pack, "world_id": request.world_id, "name": request.name}
        return _world_response(
            WorldApplicationService().create(request.world_id, request.name, pack)
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.get("/worlds", response_model=list[V2WorldResponse])
def list_worlds():
    try:
        return [_world_response(item) for item in WorldApplicationService().list()]
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.get("/worlds/{world_id}", response_model=V2WorldResponse)
def get_world(world_id: str, response: Response):
    try:
        record = WorldApplicationService().get(world_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"World not found: {world_id}")
        response.headers["ETag"] = f'"{record.version}"'
        return _world_response(record)
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.put("/worlds/{world_id}", response_model=V2WorldResponse)
def update_world(
    world_id: str,
    request: V2WorldUpdate,
    response: Response,
    if_match: str = Header(..., alias="If-Match"),
):
    try:
        expected_version = int(if_match.strip().strip('"'))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_IF_MATCH",
                "message": "If-Match must be a version",
            },
        ) from exc
    try:
        pack = {**request.pack, "world_id": world_id, "name": request.name}
        record = WorldApplicationService().update(world_id, expected_version, pack)
        response.headers["ETag"] = f'"{record.version}"'
        return _world_response(record)
    except LookupError as exc:
        raise HTTPException(
            status_code=404, detail=f"World not found: {world_id}"
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=412,
            detail={
                "code": "WORLD_VERSION_CONFLICT",
                "message": "Reload the world and reapply the change",
            },
        ) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.post(
    "/story-sessions",
    response_model=V2StorySessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_story_session(request: V2StorySessionCreate):
    try:
        record = StoryApplicationService().create_session(**request.model_dump())
        return _session_response(record)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.get("/story-sessions", response_model=list[V2StorySessionResponse])
def list_story_sessions():
    try:
        return [
            _session_response(item)
            for item in StoryApplicationService().list_sessions()
        ]
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.get("/story-sessions/{session_id}", response_model=V2StorySessionResponse)
def get_story_session(session_id: str, response: Response):
    try:
        record = StoryApplicationService().get_session(session_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Story session not found: {session_id}"
            )
        response.headers["ETag"] = f'"{record.version}"'
        return _session_response(record)
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.post(
    "/story-sessions/{session_id}/turns",
    response_model=V2JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def enqueue_story_turn(
    session_id: str,
    request: V2StoryTurnCreate,
    http_request: Request,
    response: Response,
    idempotency_key: str = Header(
        ...,
        alias="Idempotency-Key",
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    ),
):
    service = StoryApplicationService()
    try:
        enqueued = service.enqueue_turn(
            session_id=session_id,
            idempotency_key=idempotency_key,
            **request.model_dump(),
            request_id=getattr(http_request.state, "request_id", None),
        )
        if enqueued.replayed:
            response.headers["Idempotent-Replayed"] = "true"
        else:
            try:
                _dispatch_story_turn(enqueued.job.id)
                _mark_dispatch(enqueued.job.id)
            except Exception as exc:  # noqa: BLE001
                # The job remains durable and can be retried by a worker/reconciler.
                logger.warning(
                    "Story job %s dispatch deferred: %s", enqueued.job.id, exc
                )
                response.headers["Job-Dispatch"] = "deferred"
                _mark_dispatch(enqueued.job.id, deferred=True)
        response.headers["Location"] = f"/api/v2/jobs/{enqueued.job.id}"
        return _job_response(enqueued.job, replayed=enqueued.replayed)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StoryTurnInProgressError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "STORY_TURN_IN_PROGRESS", "job_id": exc.job_id},
        ) from exc
    except IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "STORY_TURN_CONFLICT",
                "message": "Retry with the same key",
            },
        ) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.get(
    "/story-sessions/{session_id}/turns",
    response_model=list[V2StoryTurnResponse],
)
def list_story_turns(session_id: str):
    try:
        return [
            _turn_response(item)
            for item in StoryApplicationService().list_turns(session_id)
        ]
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.get("/jobs/{job_id}", response_model=V2JobResponse)
def get_job(job_id: str):
    try:
        record = StoryApplicationService().get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return _job_response(record)
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.get("/jobs/{job_id}/events", response_model=list[V2JobEventResponse])
def list_job_events(job_id: str, limit: int = Query(default=100, ge=1, le=200)):
    try:
        return [
            _job_event_response(item)
            for item in JobEventService().list(job_id, limit=limit)
        ]
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.get("/jobs", response_model=list[V2JobResponse])
def list_jobs(
    status_filter: str | None = Query(default=None, alias="status"),
    kind: str | None = None,
    session_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
):
    allowed = {None, "queued", "running", "completed", "failed", "cancelled"}
    if status_filter not in allowed:
        raise HTTPException(status_code=400, detail="Invalid job status")
    return [
        _job_response(item)
        for item in JobMaintenanceService().list(
            status=status_filter, kind=kind, session_id=session_id, limit=limit
        )
    ]


def _dispatch_job(job_id: str, kind: str) -> None:
    from workers.tasks.maintenance_v2 import dispatch_v2_job

    dispatch_v2_job(job_id, kind)


@router.post("/jobs/{job_id}/retry", response_model=V2JobResponse)
def retry_job(job_id: str):
    service = JobMaintenanceService()
    try:
        job = service.retry(job_id)
        try:
            _dispatch_job(job.id, job.kind)
            _mark_dispatch(job.id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Job %s retry dispatch deferred: %s", job.id, exc)
            _mark_dispatch(job.id, deferred=True)
        refreshed = StoryApplicationService().get_job(job.id)
        return _job_response(refreshed or job)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "JOB_NOT_RETRYABLE", "message": str(exc)},
        ) from exc


@router.post("/jobs/reconcile", response_model=V2ReconcileResponse)
def reconcile_jobs():
    service = JobMaintenanceService()
    result = service.reconcile()
    dispatched: list[str] = []
    deferred: list[str] = []
    for job_id, kind in result.requeued:
        try:
            _dispatch_job(job_id, kind)
            _mark_dispatch(job_id)
            dispatched.append(job_id)
        except Exception:  # noqa: BLE001
            _mark_dispatch(job_id, deferred=True)
            deferred.append(job_id)
    return V2ReconcileResponse(
        dispatched=dispatched, deferred=deferred, failed=result.failed
    )


@router.post(
    "/worlds/{world_id}/documents",
    response_model=V2DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    world_id: str,
    request: Request,
    response: Response,
    file: UploadFile = File(...),
):
    allowed_types = {"text/plain", "text/markdown", "application/json"}
    content_type = str(file.content_type or "application/octet-stream").split(";", 1)[0]
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail={"code": "DOCUMENT_TYPE_UNSUPPORTED", "message": content_type},
        )
    try:
        max_bytes = max(1, int(os.getenv("DOCUMENT_MAX_BYTES", str(10 * 1024 * 1024))))
    except ValueError:
        max_bytes = 10 * 1024 * 1024
    payload = await file.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "DOCUMENT_TOO_LARGE",
                "message": f"Limit is {max_bytes} bytes",
            },
        )
    if not payload:
        raise HTTPException(
            status_code=422,
            detail={"code": "DOCUMENT_EMPTY", "message": "Document is empty"},
        )
    filename = Path(file.filename or "document.txt").name
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:200] or "document.txt"
    checksum = hashlib.sha256(payload).hexdigest()
    object_key = f"worlds/{world_id}/documents/{checksum}/{filename}"
    try:
        store = get_object_store()
        store.put_bytes("uploads", object_key, payload, content_type)
        registered = DocumentApplicationService().register(
            world_id=world_id,
            filename=filename,
            object_key=object_key,
            content_type=content_type,
            checksum=checksum,
            size_bytes=len(payload),
            request_id=getattr(request.state, "request_id", None),
        )
        if not registered.replayed:
            try:
                _dispatch_document_index(registered.job.id)
                _mark_dispatch(registered.job.id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Document job %s dispatch deferred: %s", registered.job.id, exc
                )
                response.headers["Job-Dispatch"] = "deferred"
                _mark_dispatch(registered.job.id, deferred=True)
        else:
            response.headers["Idempotent-Replayed"] = "true"
        response.headers["Location"] = f"/api/v2/jobs/{registered.job.id}"
        return V2DocumentUploadResponse(
            document=_document_response(registered.document),
            job=_job_response(registered.job, replayed=registered.replayed),
            replayed=registered.replayed,
        )
    except LookupError as exc:
        store.delete("uploads", object_key)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="Object storage unavailable"
        ) from exc
    except IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "DOCUMENT_CONFLICT", "message": "Retry the upload"},
        ) from exc


@router.get("/worlds/{world_id}/documents", response_model=list[V2DocumentResponse])
def list_documents(world_id: str):
    try:
        return [
            _document_response(item)
            for item in DocumentApplicationService().list(world_id)
        ]
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.post(
    "/review-proposals",
    response_model=V2ReviewProposalResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_review_proposal(request: V2ReviewProposalCreate):
    try:
        return _proposal_response(
            ReviewApplicationService().create(**request.model_dump())
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.get("/review-proposals", response_model=list[V2ReviewProposalResponse])
def list_review_proposals(
    status_filter: str | None = None,
    world_id: str | None = None,
    session_id: str | None = None,
):
    if status_filter not in {None, "pending", "approved", "rejected"}:
        raise HTTPException(status_code=400, detail="Invalid proposal status")
    try:
        return [
            _proposal_response(item)
            for item in ReviewApplicationService().list(
                status=status_filter, world_id=world_id, session_id=session_id
            )
        ]
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.post(
    "/review-proposals/{proposal_id}/approve",
    response_model=V2ReviewApprovalResponse,
)
def approve_review_proposal(
    proposal_id: str,
    if_match: str = Header(..., alias="If-Match"),
):
    try:
        expected_version = int(if_match.strip().strip('"'))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid If-Match version") from exc
    try:
        approved = ReviewApplicationService().approve(proposal_id, expected_version)
        return V2ReviewApprovalResponse(
            proposal=_proposal_response(approved.proposal),
            world=_world_response(approved.world),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        code = (
            "WORLD_VERSION_CONFLICT"
            if str(exc) == "WORLD_VERSION_CONFLICT"
            else "PROPOSAL_CONFLICT"
        )
        raise HTTPException(
            status_code=412 if code == "WORLD_VERSION_CONFLICT" else 409,
            detail={"code": code, "message": str(exc)},
        ) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc


@router.post(
    "/review-proposals/{proposal_id}/reject",
    response_model=V2ReviewProposalResponse,
)
def reject_review_proposal(proposal_id: str):
    try:
        return _proposal_response(ReviewApplicationService().reject(proposal_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "PROPOSAL_CONFLICT", "message": str(exc)},
        ) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=503, detail="Persistence service unavailable"
        ) from exc
