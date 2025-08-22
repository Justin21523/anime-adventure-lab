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
from core.train.lora_trainer import train_lora_from_config, TrainingConfig

# Job tracker instance
job_tracker = JobTracker()


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

        # Create temporary config file with run_id if specified
        config = job_data["config"]
        if job_data.get("run_id"):
            config["output"]["run_id"] = job_data["run_id"]
        elif character_name:
            # Auto-generate run_id with character name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config["output"]["run_id"] = f"{timestamp}_{character_name}"

        # Create progress callback
        def progress_callback(step: int, total_steps: int, loss: float):
            update_job_progress(job_id, step, total_steps, loss)
            # Check if task was revoked
            if current_task.is_revoked():
                raise Exception("Training cancelled by user")

        # Save temporary config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name

        try:
            # Run training
            output_dir = train_lora_from_config(temp_config_path, progress_callback)

            # Auto-register as preset if character_name provided
            preset_id = None
            if character_name:
                preset_id = Path(output_dir).name
                preset_config = {
                    "style_id": preset_id,
                    "character_name": character_name,
                    "base_model": config["base_model"],
                    "lora_path": str(output_dir / "final" / "unet_lora"),
                    "lora_scale": 0.75,
                    "description": f"Auto-registered LoRA for {character_name}",
                    "notes": notes,
                    "registered_at": datetime.now().isoformat(),
                }

                # Save preset config
                presets_dir = Path("configs/presets")
                presets_dir.mkdir(parents=True, exist_ok=True)
                preset_file = presets_dir / f"{preset_id}.yaml"

                with open(preset_file, "w") as f:
                    yaml.dump(preset_config, f)

                print(f"[Task] Auto-registered preset: {preset_id}")

            # Prepare result
            result = {
                "output_dir": str(output_dir),
                "preset_id": preset_id,
                "character_name": character_name,
                "completed_at": datetime.now().isoformat(),
            }

            # Load training metrics if available
            metrics_file = output_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                result["metrics"] = metrics

            # Update job as completed
            job_tracker.update_job(
                job_id,
                {
                    "status": "completed",
                    "result": result,
                    "progress": {
                        "step": config["train_steps"],
                        "total_steps": config["train_steps"],
                        "progress_percent": 100.0,
                        "completed_at": datetime.now().isoformat(),
                    },
                },
            )

            print(f"[Task] Training completed successfully: {output_dir}")
            return result

        finally:
            # Clean up temporary config
            os.unlink(temp_config_path)

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
