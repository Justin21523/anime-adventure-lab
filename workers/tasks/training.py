# workers/tasks/training.py
"""
Celery tasks for LoRA training
"""

import os
import json
import time
import tempfile
import traceback
from pathlib import Path
from datetime import datetime

from celery import current_task
import yaml

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from workers.celery_app import celery_app
from workers.utils.job_tracker import JobTracker
try:
    from core.train.lora_trainer import train_lora_from_config, TrainingConfig
except Exception:
    train_lora_from_config = None  # type: ignore
    TrainingConfig = None  # type: ignore
from core.train.executor import TrainingExecutor

# Job tracker instance
job_tracker = JobTracker()
training_executor = TrainingExecutor()


def update_job_progress(job_id: str, step: int, total_steps: int, loss: float):
    """Update job progress"""
    progress = {
        "step": step,
        "total_steps": total_steps,
        "progress_percent": (step / total_steps) * 100,
        "current_loss": loss,
        "updated_at": datetime.now().isoformat(),
    }

    job_tracker.update_job(job_id, {"status": "running", "progress": progress})


@celery_app.task(bind=True)
def train_lora_async(self, config_data: dict):
    """Async LoRA training task"""
    try:
        # Update progress
        self.update_state(state="PROGRESS", meta={"step": 0, "total_steps": 1000})

        # Placeholder for actual training
        import time

        for step in range(0, 1000, 100):
            time.sleep(0.1)  # Simulate training
            self.update_state(
                state="PROGRESS", meta={"step": step, "total_steps": 1000}
            )

        return {"status": "completed", "model_path": "/path/to/trained/model"}

    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


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
    job_id = job_data["job_id"]
    config_path = job_data["config_path"]
    character_name = job_data.get("character_name")
    notes = job_data.get("notes")

    print(f"[Task] Starting LoRA training job {job_id}")

    try:
        # Update job status to running
        job_tracker.update_job(job_id, {"status": "running"})

        # 若缺少訓練依賴，切換模擬模式
        simulate = False
        if train_lora_from_config is None:
            simulate = True

        payload = {
            "base_model": job_data.get("base_model") or job_data["config"].get("base_model"),
            "output_name": job_data.get("run_id") or job_id,
            "config": job_data.get("config", {}),
            "notes": notes,
            "simulate": simulate,
        }

        # 執行訓練（或模擬）
        training_executor.run_lora_job(job_id, payload)

        # 回填進度/結果
        job_info = training_executor.job_manager.get_job(job_id, auto_progress=False)
        job_tracker.update_job(job_id, job_info or {"status": "completed"})

        return job_info or {"status": "completed", "job_id": job_id}

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        print(f"[Task] Training failed: {error_msg}")
        print(f"[Task] Error trace: {error_trace}")

        # Update job as failed
        job_tracker.update_job(
            job_id,
            {
                "status": "failed",
                "error": error_msg,
                "error_trace": error_trace,
            },
        )

        # Re-raise for Celery
        raise


@celery_app.task(name="cleanup_old_jobs")
def cleanup_old_jobs(days_old: int = 30):
    """Clean up old completed/failed jobs"""
    job_tracker.cleanup_old_jobs(days_old)
    return f"Cleaned up jobs older than {days_old} days"


@celery_app.task(name="generate_training_report")
def generate_training_report(job_id: str):
    """Generate detailed training report with visualizations"""
    job = job_tracker.get_job(job_id)
    if not job or job["status"] != "completed":
        raise ValueError("Job not found or not completed")

    output_dir = Path(job["result"]["output_dir"])

    # TODO: Generate detailed report with:
    # - Training loss curves
    # - Validation image grid
    # - Model comparison (before/after)
    # - Quality metrics (CLIP score, etc.)

    report_path = output_dir / "training_report.html"

    # Placeholder for now
    with open(report_path, "w") as f:
        f.write(
            f"""
        <html>
        <head><title>Training Report - {job_id}</title></head>
        <body>
        <h1>LoRA Training Report</h1>
        <p>Job ID: {job_id}</p>
        <p>Status: {job['status']}</p>
        <p>Output: {output_dir}</p>
        <!-- TODO: Add loss curves, validation images, metrics -->
        </body>
        </html>
        """
        )

    return str(report_path)
