# core/batch/manager.py
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from ..monitoring.logger import structured_logger
from schemas.batch import BatchStatus


class BatchManager:
    """Manages batch job metadata and status"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            import os

            AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/tmp/ai_cache")
            cache_dir = Path(AI_CACHE_ROOT) / "outputs" / "multi-modal-lab"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / "batch_jobs.db"

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for job tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    job_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    total_items INTEGER NOT NULL,
                    processed_items INTEGER DEFAULT 0,
                    failed_items INTEGER DEFAULT 0,
                    config TEXT,
                    results_path TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """
            )

            # Create index for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON batch_jobs(status)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON batch_jobs(created_at)"
            )

    async def create_job(
        self,
        job_id: str,
        task_id: str,
        job_type: str,
        total_items: int,
        config: Dict[str, Any],
    ) -> None:
        """Create a new batch job record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO batch_jobs
                    (job_id, task_id, job_type, total_items, config)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (job_id, task_id, job_type, total_items, json.dumps(config)),
                )

            structured_logger.info(
                f"Created batch job {job_id} with {total_items} items"
            )

        except Exception as e:
            structured_logger.error(f"Failed to create job {job_id}: {e}")
            raise

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job information by job_id"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM batch_jobs WHERE job_id = ?
                """,
                    (job_id,),
                )

                row = cursor.fetchone()
                if row:
                    job_info = dict(row)
                    if job_info["config"]:
                        job_info["config"] = json.loads(job_info["config"])
                    return job_info
                return None

        except Exception as e:
            structured_logger.error(f"Failed to get job {job_id}: {e}")
            return None

    async def update_job_status(
        self,
        job_id: str,
        status: BatchStatus,
        processed_items: Optional[int] = None,
        failed_items: Optional[int] = None,
        results_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update job status and progress"""
        try:
            update_fields = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
            values = [status.value]

            if processed_items is not None:
                update_fields.append("processed_items = ?")
                values.append(processed_items)

            if failed_items is not None:
                update_fields.append("failed_items = ?")
                values.append(failed_items)

            if results_path is not None:
                update_fields.append("results_path = ?")
                values.append(results_path)

            if error_message is not None:
                update_fields.append("error_message = ?")
                values.append(error_message)

            if status in [
                BatchStatus.COMPLETED,
                BatchStatus.FAILED,
                BatchStatus.CANCELLED,
            ]:
                update_fields.append("completed_at = CURRENT_TIMESTAMP")

            values.append(job_id)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"""
                    UPDATE batch_jobs
                    SET {', '.join(update_fields)}
                    WHERE job_id = ?
                """,
                    values,
                )

            structured_logger.info(f"Updated job {job_id} status to {status.value}")

        except Exception as e:
            structured_logger.error(f"Failed to update job {job_id}: {e}")
            raise

    async def list_jobs(
        self, status: Optional[BatchStatus] = None, limit: int = 20, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List batch jobs with optional status filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                if status:
                    cursor = conn.execute(
                        """
                        SELECT * FROM batch_jobs
                        WHERE status = ?
                        ORDER BY created_at DESC
                        LIMIT ? OFFSET ?
                    """,
                        (status.value, limit, offset),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM batch_jobs
                        ORDER BY created_at DESC
                        LIMIT ? OFFSET ?
                    """,
                        (limit, offset),
                    )

                jobs = []
                for row in cursor.fetchall():
                    job_info = dict(row)
                    if job_info["config"]:
                        job_info["config"] = json.loads(job_info["config"])
                    jobs.append(job_info)

                return jobs

        except Exception as e:
            structured_logger.error(f"Failed to list jobs: {e}")
            return []

    async def count_jobs(self, status: Optional[BatchStatus] = None) -> int:
        """Count total jobs with optional status filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if status:
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*) FROM batch_jobs WHERE status = ?
                    """,
                        (status.value,),
                    )
                else:
                    cursor = conn.execute("SELECT COUNT(*) FROM batch_jobs")

                return cursor.fetchone()[0]

        except Exception as e:
            structured_logger.error(f"Failed to count jobs: {e}")
            return 0

    async def cleanup_old_jobs(self, days: int = 30) -> int:
        """Clean up old completed jobs (older than N days)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM batch_jobs
                    WHERE status IN ('COMPLETED', 'FAILED', 'CANCELLED')
                    AND created_at < datetime('now', '-{} days')
                """.format(
                        days
                    )
                )

                deleted_count = cursor.rowcount
                structured_logger.info(f"Cleaned up {deleted_count} old batch jobs")
                return deleted_count

        except Exception as e:
            structured_logger.error(f"Failed to cleanup old jobs: {e}")
            return 0
