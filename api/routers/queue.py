"""Queue management router (adapted from sd-multimodal-platform).

Original: app/api/v1/queue.py
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from core.queue.queue_manager import (
    QueueManager,
    get_queue_manager,
    TaskStatus,
    TaskPriority,
)
from core.queue.queue_requests import (
    QueueSubmitRequest,
    QueueTaskResponse,
    QueueStatsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/queue", tags=["Queue"])


async def _get_qm() -> QueueManager:
    qm = await get_queue_manager()
    return qm


@router.post("/submit", response_model=QueueTaskResponse)
async def submit_task(req: QueueSubmitRequest):
    """Submit a new task to the queue."""
    qm = await _get_qm()
    priority = TaskPriority(req.priority)
    task_id = await qm.submit(req.task_type, req.payload, priority=priority)
    task = await qm.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=500, detail="Task not found after submission")
    return QueueTaskResponse(
        task_id=task.task_id,
        task_type=task.task_type,
        status=task.status.value,
        priority=task.priority.value,
        progress=task.progress,
        result=task.result,
        error=task.error,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@router.get("/task/{task_id}", response_model=QueueTaskResponse)
async def get_task_status(task_id: str):
    """Get the status of a specific task."""
    qm = await _get_qm()
    task = await qm.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return QueueTaskResponse(
        task_id=task.task_id,
        task_type=task.task_type,
        status=task.status.value,
        priority=task.priority.value,
        progress=task.progress,
        result=task.result,
        error=task.error,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a queued or running task."""
    qm = await _get_qm()
    task = await qm.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED):
        raise HTTPException(status_code=400, detail=f"Task {task_id} cannot be cancelled (status={task.status.value})")
    await qm.cancel(task_id)
    return {"task_id": task_id, "status": "cancelled"}


@router.get("/stats", response_model=QueueStatsResponse)
async def get_queue_stats():
    """Get queue statistics."""
    qm = await _get_qm()
    stats = await qm.get_stats()
    return QueueStatsResponse(**stats)


@router.get("/list")
async def list_tasks(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """List recent tasks, optionally filtered by status."""
    qm = await _get_qm()
    tasks = []
    for tid in sorted(qm._tasks, key=lambda t: qm._tasks[t].created_at, reverse=True)[:limit]:
        task = qm._tasks[tid]
        if status and task.status.value != status:
            continue
        tasks.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status.value,
            "priority": task.priority.value,
            "progress": task.progress,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
        })
    return {"tasks": tasks, "count": len(tasks)}
