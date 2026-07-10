from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os

from sqlalchemy import or_, select

from core.persistence.models import DocumentRecord, JobRecord, StoryTurnRecord
from core.persistence.unit_of_work import UnitOfWork
from core.application.job_events import append_job_event


@dataclass(frozen=True)
class ReconcileResult:
    requeued: list[tuple[str, str]]
    failed: list[str]


class JobMaintenanceService:
    def __init__(self, uow_factory=UnitOfWork) -> None:
        self.uow_factory = uow_factory

    def list(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[JobRecord]:
        with self.uow_factory() as uow:
            assert uow.session is not None
            statement = select(JobRecord).order_by(JobRecord.created_at.desc())
            if status:
                statement = statement.where(JobRecord.status == status)
            if kind:
                statement = statement.where(JobRecord.kind == kind)
            if session_id:
                statement = statement.where(JobRecord.session_id == session_id)
            return list(uow.session.scalars(statement.limit(min(100, max(1, limit)))))

    def mark_dispatched(self, job_id: str, *, deferred: bool = False) -> None:
        with self.uow_factory() as uow:
            assert uow.session is not None
            job = uow.session.get(JobRecord, job_id)
            if job is not None and job.status == "queued":
                job.dispatch_status = "deferred" if deferred else "dispatched"
                job.updated_at = datetime.now(timezone.utc)
                append_job_event(
                    uow.session,
                    job,
                    "dispatch_deferred" if deferred else "dispatch_succeeded",
                    actor="api",
                    from_status="queued",
                    to_status="queued",
                    details={"dispatch_status": job.dispatch_status},
                )

    def update_progress(
        self, job_id: str, progress: int, *, execution_id: str | None = None
    ) -> None:
        with self.uow_factory() as uow:
            assert uow.session is not None
            job = uow.session.get(JobRecord, job_id)
            if job is None or job.status != "running":
                return
            if execution_id and job.execution_id != execution_id:
                return
            next_progress = min(99, max(1, progress))
            if job.progress == next_progress:
                return
            job.progress = next_progress
            job.updated_at = datetime.now(timezone.utc)
            append_job_event(
                uow.session,
                job,
                "progress_updated",
                actor="worker",
                from_status="running",
                to_status="running",
            )

    def retry(self, job_id: str) -> JobRecord:
        with self.uow_factory() as uow:
            assert uow.session is not None
            job = uow.session.scalar(
                select(JobRecord).where(JobRecord.id == job_id).with_for_update()
            )
            if job is None:
                raise LookupError(f"Job not found: {job_id}")
            if job.status != "failed":
                raise RuntimeError(f"JOB_NOT_RETRYABLE:{job.status}")
            previous_attempts = job.attempt_count
            job.status = "queued"
            job.progress = 0
            job.attempt_count = 0
            job.execution_id = None
            job.lease_expires_at = None
            job.started_at = None
            job.finished_at = None
            job.error_code = None
            job.error_message = None
            job.dispatch_status = "pending"
            job.updated_at = datetime.now(timezone.utc)
            self._sync_dependent(uow.session, job, "queued")
            append_job_event(
                uow.session,
                job,
                "retry_requested",
                actor="admin",
                from_status="failed",
                to_status="queued",
                details={"previous_attempt_count": previous_attempts},
            )
            uow.session.flush()
            return job

    def reconcile(self) -> ReconcileResult:
        now = datetime.now(timezone.utc)
        grace_seconds = max(1, int(os.getenv("JOB_DISPATCH_GRACE_SECONDS", "10")))
        max_attempts = max(1, int(os.getenv("JOB_MAX_ATTEMPTS", "3")))
        queued_before = now - timedelta(seconds=grace_seconds)
        requeued: list[tuple[str, str]] = []
        failed: list[str] = []

        with self.uow_factory() as uow:
            assert uow.session is not None
            statement = (
                select(JobRecord)
                .where(
                    or_(
                        (JobRecord.status == "queued")
                        & (JobRecord.updated_at <= queued_before),
                        (JobRecord.status == "running")
                        & (JobRecord.lease_expires_at <= now),
                    )
                )
                .order_by(JobRecord.updated_at)
                .limit(100)
                .with_for_update(skip_locked=True)
            )
            for job in uow.session.scalars(statement):
                previous_status = job.status
                previous_execution = job.execution_id
                if job.attempt_count >= max_attempts:
                    job.status = "failed"
                    job.error_code = "JOB_RETRY_EXHAUSTED"
                    job.error_message = f"Exceeded {max_attempts} attempts"
                    job.finished_at = now
                    job.lease_expires_at = None
                    self._sync_dependent(uow.session, job, "failed")
                    append_job_event(
                        uow.session,
                        job,
                        "retry_exhausted",
                        actor="scheduler",
                        from_status=previous_status,
                        to_status="failed",
                        details={"max_attempts": max_attempts},
                    )
                    failed.append(job.id)
                    continue
                job.status = "queued"
                job.progress = 0
                job.execution_id = None
                job.lease_expires_at = None
                job.dispatch_status = "pending"
                job.updated_at = now
                self._sync_dependent(uow.session, job, "queued")
                append_job_event(
                    uow.session,
                    job,
                    "lease_reconciled",
                    actor="scheduler",
                    from_status=previous_status,
                    to_status="queued",
                    details={
                        "reason": (
                            "lease_expired"
                            if previous_status == "running"
                            else "dispatch_grace_expired"
                        ),
                        "superseded_execution_id": previous_execution,
                    },
                )
                requeued.append((job.id, job.kind))
        return ReconcileResult(requeued=requeued, failed=failed)

    @staticmethod
    def _sync_dependent(session, job: JobRecord, status: str) -> None:
        if job.turn_id:
            turn = session.get(StoryTurnRecord, job.turn_id)
            if turn is not None:
                turn.status = status
        if job.document_id:
            document = session.get(DocumentRecord, job.document_id)
            if document is not None:
                document.status = "queued" if status == "queued" else "failed"
