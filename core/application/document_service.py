from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import uuid
from typing import Any

from sqlalchemy import delete, select

from core.persistence.models import (
    DocumentChunkRecord,
    DocumentRecord,
    JobRecord,
    WorldRecord,
)
from core.persistence.unit_of_work import UnitOfWork
from core.application.job_events import append_job_event


@dataclass(frozen=True)
class RegisteredDocument:
    document: DocumentRecord
    job: JobRecord
    replayed: bool


class DocumentApplicationService:
    def __init__(self, uow_factory=UnitOfWork) -> None:
        self.uow_factory = uow_factory

    def register(
        self,
        *,
        world_id: str,
        filename: str,
        object_key: str,
        content_type: str | None,
        checksum: str,
        size_bytes: int,
        request_id: str | None = None,
    ) -> RegisteredDocument:
        with self.uow_factory() as uow:
            assert uow.session is not None
            if uow.session.get(WorldRecord, world_id) is None:
                raise LookupError(f"World not found: {world_id}")
            existing = uow.session.scalar(
                select(DocumentRecord).where(
                    DocumentRecord.world_id == world_id,
                    DocumentRecord.checksum == checksum,
                )
            )
            if existing is not None:
                job = uow.session.scalar(
                    select(JobRecord)
                    .where(
                        JobRecord.kind == "document_index",
                        JobRecord.document_id == existing.id,
                    )
                    .order_by(JobRecord.created_at.desc())
                )
                if job is None:
                    raise RuntimeError("DOCUMENT_JOB_MISSING")
                return RegisteredDocument(existing, job, True)

            document = DocumentRecord(
                world_id=world_id,
                filename=filename,
                object_key=object_key,
                content_type=content_type,
                checksum=checksum,
                status="queued",
                metadata_json={"size_bytes": size_bytes},
            )
            uow.session.add(document)
            uow.session.flush()
            job = JobRecord(
                kind="document_index",
                status="queued",
                progress=0,
                document_id=document.id,
                payload={"document_id": document.id, "world_id": world_id},
                request_id=request_id,
            )
            uow.session.add(job)
            uow.session.flush()
            append_job_event(
                uow.session,
                job,
                "queued",
                actor="api",
                to_status="queued",
                details={"kind": job.kind, "document_id": document.id},
            )
            return RegisteredDocument(document, job, False)

    def list(self, world_id: str) -> list[DocumentRecord]:
        with self.uow_factory() as uow:
            assert uow.session is not None
            if uow.session.get(WorldRecord, world_id) is None:
                raise LookupError(f"World not found: {world_id}")
            return list(
                uow.session.scalars(
                    select(DocumentRecord)
                    .where(DocumentRecord.world_id == world_id)
                    .order_by(DocumentRecord.created_at.desc())
                )
            )

    def claim_job(
        self, job_id: str, execution_id: str | None = None
    ) -> tuple[JobRecord, DocumentRecord, bool]:
        with self.uow_factory() as uow:
            assert uow.session is not None
            job = uow.session.scalar(
                select(JobRecord).where(JobRecord.id == job_id).with_for_update()
            )
            if job is None or job.kind != "document_index":
                raise LookupError(f"Document job not found: {job_id}")
            document_id = str((job.payload or {}).get("document_id") or "")
            document = uow.session.get(DocumentRecord, document_id)
            if document is None:
                raise RuntimeError("DOCUMENT_JOB_CORRUPT")
            if job.status == "completed":
                return job, document, False
            if job.status == "running":
                now = datetime.now(timezone.utc)
                lease = job.lease_expires_at
                if lease is not None and lease.tzinfo is None:
                    lease = lease.replace(tzinfo=timezone.utc)
                if execution_id and job.execution_id == execution_id:
                    return job, document, True
                if lease is not None and lease > now:
                    return job, document, False
            if job.status != "queued":
                if job.status != "running":
                    raise RuntimeError(f"JOB_NOT_CLAIMABLE:{job.status}")
            previous_status = job.status
            job.status = "running"
            job.progress = 5
            job.attempt_count += 1
            job.execution_id = execution_id or str(uuid.uuid4())
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=35)
            job.started_at = datetime.now(timezone.utc)
            job.finished_at = None
            document.status = "indexing"
            append_job_event(
                uow.session,
                job,
                "claimed",
                actor="worker",
                from_status=previous_status,
                to_status="running",
                details={"lease_minutes": 35},
            )
            return job, document, True

    def complete_job(
        self,
        job_id: str,
        chunks: list[dict[str, Any]],
        execution_id: str | None = None,
    ) -> JobRecord:
        with self.uow_factory() as uow:
            assert uow.session is not None
            job = uow.session.scalar(
                select(JobRecord).where(JobRecord.id == job_id).with_for_update()
            )
            if job is None or job.kind != "document_index":
                raise LookupError(f"Document job not found: {job_id}")
            if job.status == "completed":
                return job
            if execution_id and job.execution_id != execution_id:
                raise RuntimeError("JOB_EXECUTION_SUPERSEDED")
            document_id = str((job.payload or {}).get("document_id") or "")
            document = uow.session.get(DocumentRecord, document_id)
            if document is None:
                raise RuntimeError("DOCUMENT_JOB_CORRUPT")
            uow.session.execute(
                delete(DocumentChunkRecord).where(
                    DocumentChunkRecord.document_id == document.id
                )
            )
            for position, chunk in enumerate(chunks):
                uow.session.add(
                    DocumentChunkRecord(
                        document_id=document.id,
                        world_id=document.world_id,
                        position=position,
                        content=str(chunk["content"]),
                        embedding=chunk.get("embedding"),
                        metadata_json=dict(chunk.get("metadata") or {}),
                    )
                )
            now = datetime.now(timezone.utc)
            previous_status = job.status
            document.status = "ready"
            document.metadata_json = {
                **dict(document.metadata_json or {}),
                "chunk_count": len(chunks),
            }
            document.updated_at = now
            job.status = "completed"
            job.progress = 100
            job.result = {"document_id": document.id, "chunk_count": len(chunks)}
            job.updated_at = now
            job.lease_expires_at = None
            job.finished_at = now
            append_job_event(
                uow.session,
                job,
                "completed",
                actor="worker",
                from_status=previous_status,
                to_status="completed",
                details={"chunk_count": len(chunks)},
            )
            return job

    def fail_job(
        self,
        job_id: str,
        error_code: str,
        message: str,
        execution_id: str | None = None,
    ) -> None:
        with self.uow_factory() as uow:
            assert uow.session is not None
            job = uow.session.get(JobRecord, job_id)
            if job is None:
                return
            if execution_id and job.execution_id != execution_id:
                raise RuntimeError("JOB_EXECUTION_SUPERSEDED")
            previous_status = job.status
            document_id = str((job.payload or {}).get("document_id") or "")
            document = uow.session.get(DocumentRecord, document_id)
            if document is not None:
                document.status = "failed"
            job.status = "failed"
            job.error_code = error_code[:120]
            job.error_message = message[:4000]
            job.lease_expires_at = None
            job.finished_at = datetime.now(timezone.utc)
            append_job_event(
                uow.session,
                job,
                "failed",
                actor="worker",
                from_status=previous_status,
                to_status="failed",
                details={"error_code": job.error_code},
            )
