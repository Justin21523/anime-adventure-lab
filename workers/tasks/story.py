from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from workers.celery_app import celery_app

from core.train.job_manager import TrainJobManager
from core.train.job_context import job_context


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


@celery_app.task(bind=True, name="story_turn_task")
def story_turn_task(self, job_data: dict):
    """Execute a full Story turn pipeline and persist updates into the session file."""
    job_id = str(job_data.get("job_id") or "").strip()
    payload: Dict[str, Any] = dict(job_data.get("payload") or {})
    if not payload:
        payload = dict(job_data)

    if not job_id:
        raise ValueError("job_id is required")

    job_manager = TrainJobManager()
    existing = job_manager.get_job(job_id, auto_progress=False) or {}
    if str(existing.get("status") or "").lower() == "cancelled":
        return existing

    job_manager.update_job(
        job_id,
        status="running",
        progress=1.0,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    started = time.time()
    try:
        from schemas.story import StoryTurnRequest
        from api.routers.story import process_story_turn

        request = StoryTurnRequest(**payload)
        with job_context(job_id, "story_turn"):
            response = _run_async(process_story_turn(request))

        result = (
            response.model_dump()  # type: ignore[attr-defined]
            if hasattr(response, "model_dump")
            else response.dict()  # type: ignore[attr-defined]
        )

        duration = time.time() - started
        latest = job_manager.get_job(job_id, auto_progress=False) or {}
        latest_status = str(latest.get("status") or "").lower()
        if latest_status == "cancelled" or bool(latest.get("cancel_requested")):
            job_manager.update_job(
                job_id,
                status="cancelled",
                progress=float(latest.get("progress") or 0.0),
                result=result,
                duration_seconds=duration,
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            return {"job_id": job_id, "cancelled": True, "duration_seconds": duration}

        job_manager.update_job(
            job_id,
            status="completed",
            progress=100.0,
            result=result,
            duration_seconds=duration,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return {"job_id": job_id, "result": result, "duration_seconds": duration}

    except Exception as exc:  # noqa: BLE001
        duration = time.time() - started
        job_manager.update_job(
            job_id,
            status="failed",
            progress=0.0,
            error=str(exc)[:4000],
            duration_seconds=duration,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        raise
