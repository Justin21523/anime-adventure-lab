# core/train/job_manager.py
"""Lightweight training job tracker used by finetune/LoRA APIs."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.shared_cache import get_shared_cache


class TrainJobManager:
    """Persist training job metadata without requiring an external queue."""

    def __init__(self, cache_root: Optional[str] = None):
        cache = get_shared_cache()
        root = Path(cache_root or cache.get_path("OUTPUT_TRAINING"))
        self.jobs_dir = Path(root)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_file = self.jobs_dir / "jobs.json"
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._load()

    # Internal helpers -----------------------------------------------------
    def _load(self) -> None:
        if self.jobs_file.exists():
            try:
                self.jobs = json.loads(self.jobs_file.read_text())
            except Exception:
                self.jobs = {}

    def _save(self) -> None:
        try:
            self.jobs_file.write_text(json.dumps(self.jobs, indent=2))
        except Exception:
            pass

    # Public API -----------------------------------------------------------
    def create_job(
        self, job_type: str, payload: Dict[str, Any], status: str = "pending"
    ) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        self.jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "progress": 0.0,
            "payload": payload,
            "result_path": None,
        }
        self._save()
        return job_id

    def update_job(self, job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        job.update(updates)
        job["updated_at"] = datetime.utcnow().isoformat()
        self._save()
        return job

    def _auto_progress(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight simulated progress for environments without a trainer loop."""
        status = job.get("status", "pending")
        if status not in {"pending", "running"}:
            return job

        created_at = job.get("created_at")
        try:
            created_dt = datetime.fromisoformat(created_at) if isinstance(created_at, str) else datetime.utcnow()
        except Exception:
            created_dt = datetime.utcnow()

        elapsed = (datetime.utcnow() - created_dt).total_seconds()
        progress = min(100.0, elapsed * 5)  # 20s to completion

        new_status = "running" if progress < 100 else "completed"
        job["progress"] = round(progress, 2)
        job["status"] = new_status

        if new_status == "completed" and not job.get("result_path"):
            # Reserve a placeholder artifact path
            job["result_path"] = str(self.jobs_dir / f"{job['job_id']}_lora.safetensors")
            job["completed_at"] = datetime.utcnow().isoformat()

        # Persist updates
        self._save()
        return job

    def update_job(self, job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        job.update(updates)
        job["updated_at"] = datetime.utcnow().isoformat()
        self._save()
        return job

    def get_job(self, job_id: str, auto_progress: bool = True) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        if auto_progress:
            return self._auto_progress(job)
        return job

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [self._auto_progress(job) for job in self.jobs.values()]

    def update_job_status(self, job_id: str, status: str, progress: Optional[float] = None, result_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Explicitly set status (useful when integrating a real trainer loop)."""
        updates: Dict[str, Any] = {"status": status}
        if progress is not None:
            updates["progress"] = progress
        if result_path:
            updates["result_path"] = result_path
        if status in {"completed", "failed", "cancelled"} and "completed_at" not in updates:
            updates["completed_at"] = datetime.utcnow().isoformat()
        return self.update_job(job_id, **updates)
