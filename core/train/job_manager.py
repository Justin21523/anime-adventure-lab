# core/train/job_manager.py
"""Lightweight training job tracker used by finetune/LoRA APIs."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.shared_cache import get_shared_cache


def _ensure_writable_dir(path: Path, fallback_name: str) -> Path:
    """Return a writable directory, falling back to /tmp in restricted local envs."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        return path
    except Exception:
        fallback = Path("/tmp/ai_output/anime-adventure-lab") / fallback_name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


class TrainJobManager:
    """Persist training job metadata without requiring an external queue."""

    def __init__(self, cache_root: Optional[str] = None):
        cache = get_shared_cache()
        root = Path(cache_root or cache.get_path("OUTPUT_TRAINING"))
        self.jobs_dir = _ensure_writable_dir(Path(root), "training")
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
        self._load()
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
            "celery_task_id": None,
            "cancel_requested": False,
            "error": None,
        }
        self._save()
        return job_id

    def update_job(self, job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
        self._load()
        job = self.jobs.get(job_id)
        if not job:
            return None
        job.update(updates)
        job["updated_at"] = datetime.utcnow().isoformat()
        self._save()
        return job

    def _auto_progress(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight simulated progress for environments without a trainer loop."""
        payload = job.get("payload") or {}
        simulate = payload.get("simulate")
        if simulate is None:
            job_type = str(job.get("job_type") or "").lower()
            is_training_job = ("lora" in job_type) or ("finetune" in job_type) or ("train" in job_type)
            if is_training_job:
                simulate = os.getenv("TRAIN_SIMULATE", "1").lower() not in {"0", "false", "no"}
            else:
                simulate = False
        if not simulate:
            return job

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

    def get_job(self, job_id: str, auto_progress: bool = True) -> Optional[Dict[str, Any]]:
        self._load()
        job = self.jobs.get(job_id)
        if not job:
            return None
        if auto_progress:
            return self._auto_progress(job)
        return job

    def list_jobs(self) -> List[Dict[str, Any]]:
        self._load()
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

    def request_cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Mark a job as cancel requested (workers may poll this flag)."""
        return self.update_job(job_id, cancel_requested=True, status="cancelling")
