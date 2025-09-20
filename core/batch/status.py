# core/batch/status.py
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime


class BatchStatus(str, Enum):
    """Batch job status enumeration"""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"


class TaskProgress:
    """Task progress tracking utility"""

    def __init__(self, task_id: str, total_items: int):
        self.task_id = task_id
        self.total_items = total_items
        self.processed_items = 0
        self.failed_items = 0
        self.start_time = datetime.utcnow()
        self.current_item = None
        self.estimated_completion = None

    def update(self, processed: int, failed: int = 0, current_item: str = None):
        """Update progress counters"""
        self.processed_items = processed
        self.failed_items = failed
        self.current_item = current_item

        # Estimate completion time
        if processed > 0:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            items_per_second = processed / elapsed
            remaining_items = self.total_items - processed
            estimated_seconds = (
                remaining_items / items_per_second if items_per_second > 0 else 0
            )
            self.estimated_completion = (
                datetime.utcnow().timestamp() + estimated_seconds
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "task_id": self.task_id,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
            "progress_percent": (
                (self.processed_items / self.total_items) * 100
                if self.total_items > 0
                else 0
            ),
            "current_item": self.current_item,
            "start_time": self.start_time.isoformat(),
            "estimated_completion": (
                datetime.fromtimestamp(self.estimated_completion).isoformat()
                if self.estimated_completion
                else None
            ),
        }
