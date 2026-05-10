# workers/tasks/training.py
"""
Celery tasks for LoRA training
"""

import traceback
from pathlib import Path
from typing import Any, Dict

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from workers.celery_app import celery_app
from core.train.executor import TrainingExecutor

training_executor = TrainingExecutor()


@celery_app.task(bind=True, name="train_lora_task")
def train_lora_task(self, job_data: dict):
    """
    Background task for LoRA training

    Args:
        job_data: Dictionary containing:
            - job_id: Unique job identifier
            - config_path: Path to training config YAML
            - run_id: Optional custom run ID
            - character_name: Character name for preset registration
            - notes: Training notes
            - config: Loaded config dictionary
    """
    job_id = str(job_data.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("job_id is required")

    try:
        payload: Dict[str, Any] = dict(job_data.get("payload") or {})
        if not payload:
            # Backward compatible flattening
            payload = dict(job_data)

        payload.setdefault("job_type", job_data.get("job_type") or job_data.get("type") or "lora")
        training_executor.run_lora_job(job_id, payload)
        job_info = training_executor.job_manager.get_job(job_id, auto_progress=False)
        return job_info or {"status": "completed", "job_id": job_id}

    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        error_trace = traceback.format_exc()
        raise RuntimeError(f"Training failed: {error_msg}\n{error_trace}") from exc


@celery_app.task(name="cleanup_old_jobs")
def cleanup_old_jobs(days_old: int = 30):
    """Clean up old completed/failed jobs"""
    return f"cleanup_old_jobs is a no-op (file-based jobs): {days_old}"


@celery_app.task(name="generate_training_report")
def generate_training_report(job_id: str):
    """Generate detailed training report with visualizations"""
    raise NotImplementedError("Training report not implemented in file-based mode")
