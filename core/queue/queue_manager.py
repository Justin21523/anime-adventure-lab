"""Queue manager (adapted from sd-multimodal-platform for anime-adventure-lab).

Original: app/core/queue_manager.py
"""

from __future__ import annotations

import asyncio
import logging
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

try:
    import redis.asyncio as redis_async

    REDIS_AVAILABLE = True
except ImportError:
    redis_async = None  # type: ignore
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


PRIORITY_WEIGHTS = {
    TaskPriority.LOW: 0,
    TaskPriority.NORMAL: 1,
    TaskPriority.HIGH: 2,
    TaskPriority.CRITICAL: 3,
}


@dataclass
class TaskInfo:
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueueManager:
    """In-memory task queue with optional Redis backend."""

    def __init__(self, redis_url: Optional[str] = None):
        self._tasks: Dict[str, TaskInfo] = {}
        self._queue: List[str] = []  # ordered by priority
        self._redis = None
        if redis_url and REDIS_AVAILABLE and redis_async:
            try:
                self._redis = redis_async.from_url(redis_url)
                logger.info(f"QueueManager connected to Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory: {e}")
                self._redis = None

    async def submit(self, task_type: str, payload: Dict[str, Any], priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit a task and return task_id."""
        task_id = str(uuid.uuid4())
        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            metadata=payload,
        )
        self._tasks[task_id] = task
        self._queue.append(task_id)
        self._queue.sort(key=lambda tid: -PRIORITY_WEIGHTS.get(self._tasks[tid].priority, 0))
        task.status = TaskStatus.QUEUED
        task.updated_at = datetime.utcnow().isoformat()

        if self._redis:
            await self._redis.hset(f"task:{task_id}", mapping=task.to_dict())
            await self._redis.lpush("task_queue", task_id)

        logger.info(f"Task queued: {task_id} [{task_type}] priority={priority}")
        return task_id

    async def get_next(self) -> Optional[TaskInfo]:
        """Get the next highest-priority task."""
        if not self._queue:
            return None

        task_id = self._queue.pop(0)
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow().isoformat()
            task.updated_at = datetime.utcnow().isoformat()
        return task

    async def complete(self, task_id: str, result: Optional[Dict] = None):
        task = self._tasks.get(task_id)
        if not task:
            return
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow().isoformat()
        task.updated_at = datetime.utcnow().isoformat()
        task.result = result

    async def fail(self, task_id: str, error: str):
        task = self._tasks.get(task_id)
        if not task:
            return
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.utcnow().isoformat()
        task.updated_at = datetime.utcnow().isoformat()
        task.error = error

    async def cancel(self, task_id: str):
        task = self._tasks.get(task_id)
        if not task:
            return
        task.status = TaskStatus.CANCELLED
        task.updated_at = datetime.utcnow().isoformat()

    async def update_progress(self, task_id: str, progress: float):
        task = self._tasks.get(task_id)
        if not task:
            return
        task.progress = min(progress, 1.0)
        task.updated_at = datetime.utcnow().isoformat()

    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)

    async def get_stats(self) -> Dict[str, Any]:
        stats = {
            "total": len(self._tasks),
            "queued": sum(1 for t in self._tasks.values() if t.status == TaskStatus.QUEUED),
            "running": sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING),
            "completed": sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED),
            "cancelled": sum(1 for t in self._tasks.values() if t.status == TaskStatus.CANCELLED),
        }
        return stats

    async def close(self):
        if self._redis:
            await self._redis.close()


# Singleton
_default_queue = None


async def get_queue_manager(redis_url: Optional[str] = None) -> QueueManager:
    global _default_queue
    if _default_queue is None:
        _default_queue = QueueManager(redis_url=redis_url)
    return _default_queue
