from fastapi import APIRouter, HTTPException
from api.schemas import BatchJobRequest, BatchJobStatus

router = APIRouter(prefix="/batch", tags=["Batch"])


def _lazy_workers():
    """Import workers on demand to avoid import-time failures when Celery is not configured."""
    try:
        from workers.tasks.batch import BatchJobManager, process_batch_job  # type: ignore

        return BatchJobManager, process_batch_job
    except Exception as e:  # ImportError or others
        raise HTTPException(status_code=503, detail=f"Workers unavailable: {e}")


@router.post("/submit")
async def submit_batch_job(request: BatchJobRequest):
    """Submit a batch processing job; use Celery if available, otherwise run synchronously."""
    try:
        BatchJobManager, process_batch_job = _lazy_workers()

        # Create job
        job_id = BatchJobManager.create_job(request.job_type, request.tasks)

        # Try Celery
        try:
            task = process_batch_job.delay(job_id)  # type: ignore[attr-defined]
            return {
                "job_id": job_id,
                "celery_task_id": task.id,
                "status": "queued",
                "message": "Batch job submitted successfully",
            }
        except Exception as e:
            # Fallback: process synchronously
            result = process_batch_job(job_id)
            return {
                "job_id": job_id,
                "status": "completed_sync",
                "message": f"Celery not available, processed synchronously ({e})",
                "result": result,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=BatchJobStatus)
async def get_batch_status(job_id: str):
    """Get batch job status."""
    try:
        BatchJobManager, _ = _lazy_workers()
        job_data = BatchJobManager.get_job(job_id)

        if not job_data:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return BatchJobStatus(
            job_id=job_data["job_id"],
            status=job_data["status"],
            total_tasks=job_data["total_tasks"],
            completed_tasks=job_data["completed_tasks"],
            failed_tasks=job_data["failed_tasks"],
            created_at=job_data["created_at"],
            updated_at=job_data["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
