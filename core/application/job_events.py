from __future__ import annotations

from typing import Any

from sqlalchemy import select

from core.persistence.models import JobEventRecord, JobRecord
from core.persistence.unit_of_work import UnitOfWork


ALLOWED_ACTORS = frozenset({"api", "worker", "scheduler", "admin"})
SENSITIVE_KEYS = frozenset(
    {"password", "api_key", "authorization", "cookie", "database_url", "secret"}
)


def _sanitize(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated]"
    if isinstance(value, dict):
        return {
            str(key)[:120]: (
                "[redacted]"
                if str(key).lower() in SENSITIVE_KEYS
                else _sanitize(child, depth=depth + 1)
            )
            for key, child in list(value.items())[:30]
        }
    if isinstance(value, list):
        return [_sanitize(item, depth=depth + 1) for item in value[:30]]
    if isinstance(value, str):
        return value[:1000]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)[:1000]


def append_job_event(
    session,
    job: JobRecord,
    event_type: str,
    *,
    actor: str,
    from_status: str | None = None,
    to_status: str | None = None,
    details: dict[str, Any] | None = None,
) -> JobEventRecord:
    if actor not in ALLOWED_ACTORS:
        raise ValueError(f"Unsupported job event actor: {actor}")
    event = JobEventRecord(
        job_id=job.id,
        event_type=event_type[:80],
        from_status=from_status,
        to_status=to_status,
        progress=job.progress,
        attempt_count=job.attempt_count,
        execution_id=job.execution_id,
        request_id=job.request_id,
        actor=actor,
        details=_sanitize(details or {}),
    )
    session.add(event)
    return event


class JobEventService:
    def __init__(self, uow_factory=UnitOfWork) -> None:
        self.uow_factory = uow_factory

    def list(self, job_id: str, *, limit: int = 100) -> list[JobEventRecord]:
        with self.uow_factory() as uow:
            assert uow.session is not None
            if uow.session.get(JobRecord, job_id) is None:
                raise LookupError(f"Job not found: {job_id}")
            return list(
                uow.session.scalars(
                    select(JobEventRecord)
                    .where(JobEventRecord.job_id == job_id)
                    .order_by(JobEventRecord.occurred_at, JobEventRecord.id)
                    .limit(min(200, max(1, limit)))
                )
            )
