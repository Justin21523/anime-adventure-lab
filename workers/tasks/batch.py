# workers/tasks/batch.py
import os
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from celery import current_task

from workers.celery_app import celery_app

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")
BATCH_DIR = f"{AI_CACHE_ROOT}/batch_jobs"
Path(BATCH_DIR).mkdir(parents=True, exist_ok=True)

# Simple in-memory job storage (could use Redis/DB in production)
job_storage = {}


class BatchJobManager:
    """Manage batch job lifecycle"""

    @staticmethod
    def create_job(job_type: str, tasks: List[Dict[str, Any]]) -> str:
        """Create a new batch job"""
        job_id = str(uuid.uuid4())

        job_data = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "total_tasks": len(tasks),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "tasks": tasks,
            "results": [],
            "errors": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        # Save to storage
        job_storage[job_id] = job_data

        # Save to disk
        job_file = Path(BATCH_DIR) / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)

        return job_id

    @staticmethod
    def get_job(job_id: str) -> Dict[str, Any]:
        """Get job status"""
        if job_id in job_storage:
            return job_storage[job_id]

        # Try loading from disk
        job_file = Path(BATCH_DIR) / f"{job_id}.json"
        if job_file.exists():
            with open(job_file, "r") as f:
                job_data = json.load(f)
                job_storage[job_id] = job_data
                return job_data

        return None

    @staticmethod
    def update_job(job_id: str, **updates):
        """Update job status"""
        if job_id not in job_storage:
            job_data = BatchJobManager.get_job(job_id)
            if not job_data:
                return False

        job_data = job_storage[job_id]
        job_data.update(updates)
        job_data["updated_at"] = datetime.utcnow().isoformat()

        # Save to disk
        job_file = Path(BATCH_DIR) / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)

        return True


@celery_app.task(bind=True)
def process_batch_job(job_id: str):
    """Process a batch job"""
    job_data = BatchJobManager.get_job(job_id)
    if not job_data:
        return {"error": f"Job {job_id} not found"}

    BatchJobManager.update_job(job_id, status="running")

    total_tasks = len(job_data["tasks"])
    completed = 0
    failed = 0
    results = []
    errors = []

    for i, task_data in enumerate(job_data["tasks"]):
        try:
            # Update progress
            current_task.update_state(
                state="PROGRESS", meta={"current": i + 1, "total": total_tasks}
            )

            # Process single task based on job type
            if job_data["job_type"] == "t2i":
                result = process_t2i_task(task_data)
            elif job_data["job_type"] == "caption":
                result = process_caption_task(task_data)
            elif job_data["job_type"] == "chat":
                result = process_chat_task(task_data)
            else:
                raise ValueError(f"Unknown job type: {job_data['job_type']}")

            results.append(result)
            completed += 1

        except Exception as e:
            error_msg = str(e)
            errors.append({"task_index": i, "error": error_msg, "task_data": task_data})
            failed += 1
            print(f"Task {i} failed: {error_msg}")

        # Update job progress
        BatchJobManager.update_job(
            job_id,
            completed_tasks=completed,
            failed_tasks=failed,
            results=results,
            errors=errors,
        )

    # Final status
    final_status = "completed" if failed == 0 else "completed_with_errors"
    BatchJobManager.update_job(job_id, status=final_status)

    return {
        "job_id": job_id,
        "status": final_status,
        "completed": completed,
        "failed": failed,
        "total": total_tasks,
    }


def process_t2i_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single T2I task"""
    from core.t2i import get_t2i_pipeline, save_image_to_cache
    import torch

    # Get pipeline
    pipeline = get_t2i_pipeline(
        task_data.get("model", "runwayml/stable-diffusion-v1-5")
    )

    # Generate image
    result = pipeline(
        prompt=task_data["prompt"],
        negative_prompt=task_data.get("negative_prompt", ""),
        width=task_data.get("width", 768),
        height=task_data.get("height", 768),
        num_inference_steps=task_data.get("steps", 25),
        guidance_scale=task_data.get("guidance_scale", 7.5),
        generator=torch.Generator().manual_seed(task_data.get("seed", 42)),
    )

    # Save image
    image_path, metadata_path = save_image_to_cache(result.images[0], task_data)

    return {
        "image_path": image_path,
        "metadata_path": metadata_path,
        "seed": task_data.get("seed", 42),
    }


def process_caption_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single caption task"""
    # Placeholder implementation
    return {
        "caption": f"Generated caption for {task_data.get('image_path', 'image')}",
        "confidence": 0.9,
    }


def process_chat_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single chat task"""
    # Placeholder implementation
    return {
        "message": f"Response to: {task_data.get('prompt', 'Hello')}",
        "model_used": "placeholder",
    }
