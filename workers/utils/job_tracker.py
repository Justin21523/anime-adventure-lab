# workers/utils/job_tracker.py
"""
Job tracking utility for managing training jobs
Stores job state in Redis for persistence across restarts
"""

import json
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


class JobTracker:
    """Simple job tracker using Redis backend"""

    def __init__(self, redis_url: str = None):
        if redis_url is None:
            redis_url = "redis://localhost:6379/1"  # Use DB 1 for jobs

        self.redis_client = redis.from_url(redis_url)
        self.job_prefix = "job:"
        self.index_key = "job_index"

    def _job_key(self, job_id: str) -> str:
        """Get Redis key for job"""
        return f"{self.job_prefix}{job_id}"

    def create_job(
        self, job_id: str, task_id: str, job_type: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new job entry"""
        now = datetime.now().isoformat()

        job_data = {
            "job_id": job_id,
            "task_id": task_id,
            "job_type": job_type,
            "status": "pending",
            "config": config,
            "progress": None,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }

        # Store job data
        self.redis_client.setex(
            self._job_key(job_id),
            timedelta(days=30),  # TTL: 30 days
            json.dumps(job_data),
        )

        # Add to index
        self.redis_client.zadd(self.index_key, {job_id: datetime.now().timestamp()})

        return job_data

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        job_data = self.redis_client.get(self._job_key(job_id))
        if job_data:
            return json.loads(job_data)
        return None

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job data"""
        job_data = self.get_job(job_id)
        if not job_data:
            return False

        # Apply updates
        job_data.update(updates)
        job_data["updated_at"] = datetime.now().isoformat()

        # Store updated data
        self.redis_client.setex(
            self._job_key(job_id), timedelta(days=30), json.dumps(job_data)
        )

        return True

    def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filtering"""
        # Get job IDs from index (sorted by creation time, newest first)
        job_ids = self.redis_client.zrevrange(
            self.index_key, offset, offset + limit - 1
        )

        jobs = []
        for job_id in job_ids:
            job_data = self.get_job(job_id.decode())
            if job_data:
                # Apply filters
                if job_type and job_data["job_type"] != job_type:
                    continue
                if status and job_data["status"] != status:
                    continue

                jobs.append(job_data)

        return jobs

    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        # Remove from index
        self.redis_client.zrem(self.index_key, job_id)

        # Remove job data
        result = self.redis_client.delete(self._job_key(job_id))
        return result > 0

    def cleanup_old_jobs(self, days_old: int = 30):
        """Clean up jobs older than specified days"""
        cutoff_timestamp = (datetime.now() - timedelta(days=days_old)).timestamp()

        # Get old job IDs
        old_job_ids = self.redis_client.zrangebyscore(
            self.index_key, 0, cutoff_timestamp
        )

        # Delete old jobs
        for job_id in old_job_ids:
            self.delete_job(job_id.decode())

        return len(old_job_ids)

    def get_job_stats(self) -> Dict[str, int]:
        """Get job statistics"""
        all_job_ids = self.redis_client.zrange(self.index_key, 0, -1)

        stats = {
            "total": 0,
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
        }

        for job_id in all_job_ids:
            job_data = self.get_job(job_id.decode())
            if job_data:
                stats["total"] += 1
                status = job_data["status"]
                if status in stats:
                    stats[status] += 1

        return stats


# Global job tracker instance
_job_tracker = None


def get_job_tracker() -> JobTracker:
    """Get global job tracker instance"""
    global _job_tracker
    if _job_tracker is None:
        _job_tracker = JobTracker()
    return _job_tracker
