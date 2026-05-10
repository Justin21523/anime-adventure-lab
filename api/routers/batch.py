# api/routers/batch.py
"""
Batch Processing Router
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from core.shared_cache import get_shared_cache
from schemas.batch import (
    BatchJobRequest,
    BatchJobResponse,
    BatchStatus,
    BatchJobList,
    TaskProgress,
)
from core.batch.manager import BatchManager, DEFAULT_CACHE_ROOT
from core.batch.queue import QueueManager

router = APIRouter(prefix="/batch", tags=["batch"])
batch_manager = BatchManager()
queue_manager = QueueManager(batch_manager)

try:
    from workers.tasks import (
        batch_caption_task,
        batch_vqa_task,
        batch_chat_task,
        get_task_status,
        cancel_task,
    )
except Exception as exc:  # noqa: BLE001
    logging.warning("Celery batch tasks unavailable; batch API will use local fallback: %s", exc)
    batch_caption_task = None
    batch_vqa_task = None
    batch_chat_task = None

    def get_task_status(task_id: str) -> Dict[str, Any]:
        return {
            "task_id": task_id,
            "status": "PENDING",
            "error": "Celery is not installed or not available in this environment",
        }

    def cancel_task(task_id: str) -> bool:
        return str(task_id).startswith(("local-", "sim-"))


def _celery_available() -> bool:
    return all((batch_caption_task, batch_vqa_task, batch_chat_task))


def _parse_dt(ts: Any) -> Optional[datetime]:
    """Safely parse sqlite timestamp/iso/datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None


def _map_status(status: str) -> BatchStatus:
    """Map Celery/worker states to BatchStatus."""
    normalized = (status or "").upper()
    if normalized in {"PENDING", "RECEIVED"}:
        return BatchStatus.PENDING
    if normalized in {"STARTED", "PROGRESS", "RUNNING", "PROCESSING"}:
        return BatchStatus.PROCESSING
    if normalized in {"SUCCESS", "COMPLETED"}:
        return BatchStatus.COMPLETED
    if normalized in {"RETRY", "RETRYING"}:
        return BatchStatus.RETRYING
    if normalized in {"REVOKED", "CANCELLED"}:
        return BatchStatus.CANCELLED
    if normalized in {"FAILURE", "FAILED"}:
        return BatchStatus.FAILED
    return BatchStatus.PENDING


def _write_simulated_results(job_id: str, payload: Dict[str, Any]) -> str:
    """Persist lightweight simulated results for testing/offline runs."""
    cache = get_shared_cache()
    out_dir = Path(cache.get_path("OUTPUT_BATCH")) / "batch_results"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        out_dir = Path("/tmp/ai_output/batch/batch_results")
        out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / f"simulated_results_{job_id}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(results_file)


@router.post("/submit", response_model=BatchJobResponse)
async def submit_batch_job(request: BatchJobRequest):
    """Submit a batch processing job"""
    try:
        job_id = str(uuid.uuid4())

        # Supported job types
        supported = {"caption", "vqa", "chat"}
        if request.job_type not in supported:
            raise HTTPException(
                status_code=400, detail=f"Unsupported job type: {request.job_type}"
            )

        # Simulation path (no Celery/broker required)
        if request.simulate:
            await batch_manager.create_job(
                job_id=job_id,
                task_id=f"sim-{job_id}",
                job_type=request.job_type,
                total_items=len(request.inputs),
                config=request.config,
            )
            results = [
                {"input": inp, "result": f"simulated_result_{idx}"}
                for idx, inp in enumerate(request.inputs)
            ]
            results_path = _write_simulated_results(
                job_id,
                {
                    "job_id": job_id,
                    "job_type": request.job_type,
                    "results": results,
                    "config": request.config,
                },
            )
            await batch_manager.update_job_status(
                job_id,
                BatchStatus.COMPLETED,
                processed_items=len(request.inputs),
                failed_items=0,
                results_path=results_path,
            )
            return BatchJobResponse(
                job_id=job_id,
                task_id=f"sim-{job_id}",
                status=BatchStatus.COMPLETED,
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                total_items=len(request.inputs),
                processed_items=len(request.inputs),
                failed_items=0,
                results_path=results_path,
            )

        if not _celery_available():
            task_id = f"local-{job_id}"
            logging.warning(
                "Celery is unavailable; creating local pending batch job. "
                "Use request.simulate=true for immediate local results."
            )
            await batch_manager.create_job(
                job_id=job_id,
                task_id=task_id,
                job_type=request.job_type,
                total_items=len(request.inputs),
                config=request.config,
            )
            await batch_manager.update_job_status(
                job_id, BatchStatus.PENDING, processed_items=0
            )
            return BatchJobResponse(
                job_id=job_id,
                task_id=task_id,
                status=BatchStatus.PENDING,
                created_at=datetime.utcnow(),
                total_items=len(request.inputs),
                processed_items=0,
                failed_items=0,
            )

        # Route to Celery task
        try:
            if request.job_type == "caption":
                task = batch_caption_task.delay(
                    job_id=job_id, image_paths=request.inputs, config=request.config
                )
            elif request.job_type == "vqa":
                task = batch_vqa_task.delay(
                    job_id=job_id, inputs=request.inputs, config=request.config
                )
            else:
                task = batch_chat_task.delay(
                    job_id=job_id, messages_batch=request.inputs, config=request.config
                )
            task_id = task.id
        except Exception as exc:  # noqa: BLE001
            # Fallback to synchronous execution of a stub result
            task_id = f"local-{job_id}"
            logging.warning(
                "Falling back to local batch simulation because Celery submission failed: %s",
                exc,
            )
            await batch_manager.create_job(
                job_id=job_id,
                task_id=task_id,
                job_type=request.job_type,
                total_items=len(request.inputs),
                config=request.config,
            )
            await batch_manager.update_job_status(
                job_id, BatchStatus.PENDING, processed_items=0
            )
            return BatchJobResponse(
                job_id=job_id,
                task_id=task_id,
                status=BatchStatus.PENDING,
                created_at=datetime.utcnow(),
                total_items=len(request.inputs),
                processed_items=0,
                failed_items=0,
            )

        # Store job metadata
        await batch_manager.create_job(
            job_id=job_id,
            task_id=task_id,
            job_type=request.job_type,
            total_items=len(request.inputs),
            config=request.config,
        )

        return BatchJobResponse(
            job_id=job_id,
            task_id=task_id,
            status=BatchStatus.PENDING,
            created_at=datetime.utcnow(),
            total_items=len(request.inputs),
            processed_items=0,
            failed_items=0,
        )

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logging.error("Batch submit failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/status/{job_id}", response_model=BatchJobResponse)
async def get_batch_status(job_id: str):
    """Get batch job status and progress"""
    try:
        job_info = await batch_manager.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")

        # Simulated jobs: trust stored state
        if str(job_info.get("task_id", "")).startswith("sim-"):
            stored_status = _map_status(job_info.get("status", "COMPLETED"))
            return BatchJobResponse(
                job_id=job_id,
                task_id=job_info["task_id"],
                status=stored_status,
                created_at=_parse_dt(job_info["created_at"]) or datetime.utcnow(),
                completed_at=_parse_dt(job_info.get("completed_at"))
                or datetime.utcnow(),
                total_items=job_info["total_items"],
                processed_items=job_info.get("processed_items", 0),
                failed_items=job_info.get("failed_items", 0),
                results_path=job_info.get("results_path"),
                error_message=job_info.get("error_message"),
            )

        # Get Celery task status
        task_status = get_task_status(job_info["task_id"])
        mapped_status = _map_status(task_status.get("status"))
        stored_status = _map_status(job_info.get("status", "PENDING"))
        # Prefer non-pending stored status
        final_status = (
            stored_status
            if stored_status in {BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED}
            else mapped_status
        )

        # Optionally update DB status
        await batch_manager.update_job_status(
            job_id,
            final_status,
            processed_items=task_status.get("processed_items"),
            failed_items=task_status.get("failed_items"),
            results_path=task_status.get("results_file")
            or task_status.get("results_path"),
            error_message=task_status.get("error"),
        )

        return BatchJobResponse(
            job_id=job_id,
            task_id=job_info["task_id"],
            status=final_status,
            created_at=_parse_dt(job_info["created_at"]) or datetime.utcnow(),
            completed_at=_parse_dt(
                task_status.get("completed_at") or job_info.get("completed_at")
            ),
            total_items=job_info["total_items"],
            processed_items=task_status.get("processed_items", 0),
            failed_items=task_status.get("failed_items", 0),
            results_path=task_status.get("results_file")
            or task_status.get("results_path")
            or job_info.get("results_path"),
            error_message=task_status.get("error"),
        )

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logging.error("Failed to get batch status: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/list", response_model=BatchJobList)
async def list_batch_jobs(
    status: Optional[BatchStatus] = None, limit: int = 20, offset: int = 0
):
    """List batch jobs with optional filtering"""
    try:
        raw_jobs = await batch_manager.list_jobs(
            status=status, limit=limit, offset=offset
        )
        jobs: list[BatchJobResponse] = []
        for job in raw_jobs:
            jobs.append(
                BatchJobResponse(
                    job_id=job["job_id"],
                    task_id=job["task_id"],
                    status=_map_status(job["status"]),
                    created_at=_parse_dt(job.get("created_at")) or datetime.utcnow(),
                    completed_at=_parse_dt(job.get("completed_at")),
                    total_items=job["total_items"],
                    processed_items=job.get("processed_items", 0),
                    failed_items=job.get("failed_items", 0),
                    results_path=job.get("results_path"),
                    error_message=job.get("error_message"),
                )
            )

        total = await batch_manager.count_jobs(status=status)

        return BatchJobList(jobs=jobs, total=total, limit=limit, offset=offset)

    except Exception as e:  # noqa: BLE001
        logging.error("Failed to list batch jobs: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/cancel/{job_id}")
async def cancel_batch_job(job_id: str):
    """Cancel a running batch job"""
    try:
        job_info = await batch_manager.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")

        # Cancel Celery task
        success = cancel_task(job_info["task_id"])

        if success:
            await batch_manager.update_job_status(job_id, BatchStatus.CANCELLED)
            return {"message": "Job cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel job")

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logging.error("Failed to cancel batch job: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/results/{job_id}")
async def get_batch_results(job_id: str):
    """Download batch job results"""
    try:
        job_info = await batch_manager.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")

        results_path = job_info.get("results_path")
        if not results_path:
            raise HTTPException(status_code=404, detail="Results not available")

        # Return file or JSON results
        # Implementation depends on result format
        return {"results_path": results_path, "download_url": f"/download/{job_id}"}

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logging.error("Failed to get batch results: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/download/{job_id}")
async def download_batch_results(job_id: str):
    """Serve batch result file if available."""
    try:
        job_info = await batch_manager.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")

        results_path = job_info.get("results_path")
        if not results_path or not os.path.isfile(results_path):
            raise HTTPException(status_code=404, detail="Results file not found")

        filename = os.path.basename(results_path)
        return FileResponse(
            results_path,
            media_type="application/json",
            filename=filename,
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logging.error("Failed to download batch results: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/queues")
async def get_queue_stats():
    """Get Celery queue statistics."""
    try:
        stats = await queue_manager.get_queue_stats()
        return {"queues": stats, "count": len(stats)}
    except Exception as e:  # noqa: BLE001
        logging.error("Failed to get queue stats: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/progress/{task_id}", response_model=TaskProgress)
async def get_task_progress(task_id: str):
    """Get detailed task progress from Redis/Celery."""
    try:
        status = get_task_status(task_id)
        total = status.get("total_items") or status.get("total") or 0
        processed = status.get("processed_items") or status.get("current") or 0
        failed = status.get("failed_items") or status.get("failed") or 0
        progress_percent = (
            (processed / total) * 100 if total else status.get("progress_percent", 0)
        )
        start_time = _parse_dt(status.get("created_at")) or datetime.utcnow()
        estimated = status.get("estimated_completion")
        estimated_dt = _parse_dt(estimated) if estimated else None
        return TaskProgress(
            task_id=task_id,
            total_items=total,
            processed_items=processed,
            failed_items=failed,
            progress_percent=progress_percent,
            current_item=status.get("current_item"),
            start_time=start_time,
            estimated_completion=estimated_dt,
            partial_results=status.get("partial_results"),
        )
    except Exception as e:  # noqa: BLE001
        logging.error("Failed to get task progress: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/cleanup")
async def cleanup_completed_jobs(days: int = 30):
    """Cleanup completed/failed/cancelled jobs older than N days."""
    try:
        removed = await batch_manager.cleanup_old_jobs(days=days)
        return {"removed": removed, "days": days}
    except Exception as e:  # noqa: BLE001
        logging.error("Failed to cleanup jobs: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# Example usage:
"""
# Submit batch caption job
POST /api/v1/batch/submit
{
  "job_type": "caption",
  "inputs": ["/path/img1.jpg", "/path/img2.jpg"],
  "config": {
    "max_length": 50,
    "num_beams": 3
  }
}

# Response
{
  "job_id": "uuid-here",
  "task_id": "celery-task-id",
  "status": "PENDING",
  "created_at": "2025-01-01T00:00:00Z",
  "total_items": 2,
  "processed_items": 0
}

# Get status
GET /api/v1/batch/status/uuid-here
{
  "job_id": "uuid-here",
  "status": "PROCESSING",
  "total_items": 2,
  "processed_items": 1,
  "failed_items": 0
}
"""
