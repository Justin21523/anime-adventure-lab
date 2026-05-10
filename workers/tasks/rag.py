from __future__ import annotations

from typing import Any, Dict

from workers.celery_app import celery_app

from core.rag.job_runner import run_rag_rebuild_job, run_rag_upload_job


@celery_app.task(bind=True, name="rag_rebuild_task")
def rag_rebuild_task(self, job_data: dict):
    """Rebuild RAG index/embeddings in background (Celery wrapper)."""
    job_id = str(job_data.get("job_id") or "").strip()
    payload: Dict[str, Any] = dict(job_data.get("payload") or {})
    if not payload:
        payload = dict(job_data)
    return run_rag_rebuild_job(job_id, payload)


@celery_app.task(bind=True, name="rag_upload_task")
def rag_upload_task(self, job_data: dict):
    """Ingest uploaded files into the RAG index as a background job (Celery wrapper)."""
    job_id = str(job_data.get("job_id") or "").strip()
    payload: Dict[str, Any] = dict(job_data.get("payload") or {})
    if not payload:
        payload = dict(job_data)
    return run_rag_upload_job(job_id, payload)

