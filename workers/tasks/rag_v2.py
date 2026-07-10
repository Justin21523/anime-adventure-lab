from __future__ import annotations

import os
import time
from typing import Iterable

from core.application.document_service import DocumentApplicationService
from core.application.job_service import JobMaintenanceService
from core.application.embedding_service import embed_texts
from core.application.embedding_service import (
    deterministic_embedding as _deterministic_embedding,  # noqa: F401 - compatibility export
)
from core.config import get_config
from core.storage.object_store import get_object_store
from workers.celery_app import celery_app


def _chunks(text: str, size: int, overlap: int) -> Iterable[str]:
    start = 0
    while start < len(text):
        value = text[start : start + size].strip()
        if value:
            yield value
        if start + size >= len(text):
            break
        start += max(1, size - overlap)


@celery_app.task(bind=True, name="document_index_v2_task")
def document_index_v2_task(self, job_id: str):
    service = DocumentApplicationService()
    execution_id = str(self.request.id or "")
    try:
        job, document, claimed = service.claim_job(job_id, execution_id=execution_id)
        if not claimed:
            return job.result or {"job_id": job.id, "status": job.status}
        delay = max(0, int(os.getenv("DEMO_JOB_DELAY_MS", "0"))) / 1000
        payload = get_object_store().get_bytes("uploads", document.object_key)
        JobMaintenanceService().update_progress(job_id, 30, execution_id=execution_id)
        if delay:
            time.sleep(delay / 2)
        text = payload.decode("utf-8-sig")
        config = get_config().rag
        contents = list(_chunks(text, config.chunk_size, config.chunk_overlap))
        vectors = embed_texts(contents)
        JobMaintenanceService().update_progress(job_id, 70, execution_id=execution_id)
        if delay:
            time.sleep(delay / 2)
        chunks = [
            {"content": content, "embedding": vector, "metadata": {"position": index}}
            for index, (content, vector) in enumerate(zip(contents, vectors))
        ]
        completed = service.complete_job(job_id, chunks, execution_id=execution_id)
        return completed.result
    except Exception as exc:  # noqa: BLE001
        service.fail_job(
            job_id, type(exc).__name__, str(exc), execution_id=execution_id
        )
        raise
