# core/batch/queue.py
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from .manager import BatchManager
from ..monitoring.logger import structured_logger


class QueueManager:
    """Manages task queues and priorities"""

    def __init__(self, batch_manager: BatchManager):
        self.batch_manager = batch_manager
        self.queue_priorities = {
            "training": 1,  # Highest priority
            "vision": 2,
            "text": 3,
            "default": 4,  # Lowest priority
        }

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics for all queues"""
        try:
            from ...jobs.worker import celery_app

            inspect = celery_app.control.inspect()
            active_tasks = inspect.active() or {}
            scheduled_tasks = inspect.scheduled() or {}

            queue_stats = {}

            for queue_name in self.queue_priorities.keys():
                active_count = sum(
                    len(
                        [
                            task
                            for task in worker_tasks
                            if task.get("delivery_info", {}).get("routing_key")
                            == queue_name
                        ]
                    )
                    for worker_tasks in active_tasks.values()
                )

                scheduled_count = sum(
                    len(
                        [
                            task
                            for task in worker_tasks
                            if task.get("delivery_info", {}).get("routing_key")
                            == queue_name
                        ]
                    )
                    for worker_tasks in scheduled_tasks.values()
                )

                queue_stats[queue_name] = {
                    "active_tasks": active_count,
                    "scheduled_tasks": scheduled_count,
                    "priority": self.queue_priorities[queue_name],
                }

            return queue_stats

        except Exception as e:
            structured_logger.error(f"Failed to get queue stats: {e}")
            return {}

    async def estimate_queue_time(self, queue_name: str) -> Optional[int]:
        """Estimate waiting time in queue (seconds)"""
        try:
            stats = await self.get_queue_stats()
            queue_info = stats.get(queue_name, {})

            # Simple estimation: assume 30 seconds per task
            pending_tasks = queue_info.get("scheduled_tasks", 0)
            estimated_seconds = pending_tasks * 30

            return estimated_seconds

        except Exception as e:
            structured_logger.error(f"Failed to estimate queue time: {e}")
            return None
