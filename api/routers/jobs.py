# api/routers/jobs.py
"""
Jobs Router

Provides a centralized API for polling/canceling background jobs (training, RAG rebuild,
story scene image, dataset caption, etc.).

Paths are intentionally stable for frontend polling:
- GET /jobs
- GET /jobs/{job_id}
- DELETE /jobs/{job_id}
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException

from core.train.job_manager import TrainJobManager

router = APIRouter()
job_manager = TrainJobManager()


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status by id (auto-progress for simulated jobs)."""
    job = job_manager.get_job(job_id, auto_progress=True)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return {"job_id": job_id, "status": job.get("status"), **job}


@router.get("/jobs")
async def list_jobs():
    """List all jobs (with simulated progress)."""
    return {"jobs": job_manager.list_jobs()}


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Best-effort cancel a job (revokes celery task if tracked)."""
    job = job_manager.get_job(job_id, auto_progress=False)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    task_id = str(job.get("celery_task_id") or "").strip() or None
    revoked = False
    if task_id:
        try:
            from workers.celery_app import celery_app

            celery_app.control.revoke(task_id, terminate=True)
            revoked = True
        except Exception:
            revoked = False

    updated = job_manager.update_job(
        job_id,
        status="cancelled",
        progress=float(job.get("progress", 0.0) or 0.0),
        cancelled_at=datetime.utcnow().isoformat(),
        cancelled_task_revoked=revoked,
    )
    return {"job_id": job_id, "cancelled": True, "revoked": revoked, "job": updated or job}

