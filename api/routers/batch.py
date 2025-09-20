# api/routers/batch.py
"""
Batch Processing Router
"""

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Tuple, Any, Optional
import uuid
from datetime import datetime
from schemas.batch import (
    BatchJobRequest,
    BatchJobResponse,
    BatchStatus,
    BatchJobList,
    TaskProgress,
)
from workers.tasks import (
    batch_caption_task,
    batch_vqa_task,
    batch_chat_task,
    get_task_status,
    cancel_task,
)
from core.batch.manager import BatchManager

router = APIRouter(prefix="/batch", tags=["batch"])
batch_manager = BatchManager()


@router.post("/submit", response_model=BatchJobResponse)
async def submit_batch_job(request: BatchJobRequest):
    """Submit a batch processing job"""
    try:
        job_id = str(uuid.uuid4())

        # Route to appropriate task based on job type
        if request.job_type == "caption":
            task = batch_caption_task.delay(
                job_id=job_id, image_paths=request.inputs, config=request.config
            )
        elif request.job_type == "vqa":
            task = batch_vqa_task.delay(
                job_id=job_id, inputs=request.inputs, config=request.config
            )
        elif request.job_type == "chat":
            task = batch_chat_task.delay(
                job_id=job_id, messages_batch=request.inputs, config=request.config
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported job type: {request.job_type}"
            )

        # Store job metadata
        await batch_manager.create_job(
            job_id=job_id,
            task_id=task.id,
            job_type=request.job_type,
            total_items=len(request.inputs),
            config=request.config,
        )

        return BatchJobResponse(
            job_id=job_id,
            task_id=task.id,
            status=BatchStatus.PENDING,
            created_at=datetime.utcnow(),
            total_items=len(request.inputs),
            processed_items=0,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=BatchJobResponse)
async def get_batch_status(job_id: str):
    """Get batch job status and progress"""
    try:
        job_info = await batch_manager.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get Celery task status
        task_status = get_task_status(job_info["task_id"])

        return BatchJobResponse(
            job_id=job_id,
            task_id=job_info["task_id"],
            status=task_status["status"],
            created_at=job_info["created_at"],
            completed_at=task_status.get("completed_at"),
            total_items=job_info["total_items"],
            processed_items=task_status.get("processed_items", 0),
            failed_items=task_status.get("failed_items", 0),
            results_path=task_status.get("results_path"),
            error_message=task_status.get("error"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=BatchJobList)
async def list_batch_jobs(
    status: Optional[BatchStatus] = None, limit: int = 20, offset: int = 0
):
    """List batch jobs with optional filtering"""
    try:
        jobs = await batch_manager.list_jobs(status=status, limit=limit, offset=offset)

        total = await batch_manager.count_jobs(status=status)

        return BatchJobList(jobs=jobs, total=total, limit=limit, offset=offset)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
