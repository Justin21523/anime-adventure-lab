# api/routers/finetune.py
"""
Fine-tuning Router
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, BackgroundTasks
from schemas.finetune import FinetuneRequest, FinetuneResponse, FinetuneStatusResponse
from core.train.job_manager import TrainJobManager
from core.train.registry import ModelRegistry
from core.train.executor import TrainingExecutor
from pathlib import Path
import time

logger = logging.getLogger(__name__)
router = APIRouter()
train_job_manager = TrainJobManager()
model_registry = ModelRegistry()
training_executor = TrainingExecutor()


def _simulate_training(job_id: str, job_type: str, payload: Dict[str, Any]):
    """委派至訓練執行器（可模擬或嘗試真實訓練，取決於 payload.simulate 或環境變量）。"""
    training_executor.run_lora_job(job_id, payload | {"job_type": job_type})


@router.post("/finetune/start", response_model=FinetuneResponse)
async def start_finetuning(request: FinetuneRequest, background_tasks: BackgroundTasks):
    """Start fine-tuning job"""
    try:
        job_id = train_job_manager.create_job(
            "finetune",
            {
                "model_name": request.model_name,
                "dataset_path": request.dataset_path,
                "output_name": request.output_name,
                "parameters": request.parameters.dict() if request.parameters else {},
                "simulate": True,
            },
            status="queued",
        )

        # Kick off lightweight simulated trainer
        background_tasks.add_task(
            _simulate_training,
            job_id,
            "finetune",
            {
                "model_name": request.model_name,
                "output_name": request.output_name,
                "dataset_path": request.dataset_path,
                "simulate": True,
            },
        )

        return FinetuneResponse(  # type: ignore
            job_id=job_id,
            status="queued",
            estimated_time_hours=2.5,
            parameters=request.parameters,
        )
    except Exception as e:
        raise HTTPException(500, f"Fine-tuning start failed: {str(e)}")


@router.get("/finetune/{job_id}/status", response_model=FinetuneStatusResponse)
async def get_finetuning_status(job_id: str):
    """Get fine-tuning job status"""
    try:
        job = train_job_manager.get_job(job_id, auto_progress=True)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")

        status = job.get("status", "pending")
        progress = float(job.get("progress", 0.0))
        # Provide minimal fields for compatibility
        return FinetuneStatusResponse(  # type: ignore
            job_id=job_id,
            status=status,
            progress_percentage=progress,
            current_epoch=int(progress // 10) if progress else 0,
            total_epochs=10,
            current_loss=None,
            model_path=job.get("result_path"),
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Status check failed: {str(e)}")


@router.post("/finetune/lora")
async def submit_lora_training(
    payload: Dict[str, Any] = Body(...), background_tasks: BackgroundTasks = None
):
    """
    Submit a LoRA training job. Uses lightweight on-disk tracking and returns a job_id.
    The actual training loop is not executed here; this endpoint wires into the job manager
    so callers can poll status via `/jobs/{job_id}`.
    """
    try:
        job_type = str(payload.get("job_type") or payload.get("type") or "lora").strip()
        simulate = payload.get("simulate")
        if simulate is None:
            simulate = True

        stored_payload = dict(payload)
        stored_payload["job_type"] = job_type
        stored_payload["simulate"] = bool(simulate)

        job_id = train_job_manager.create_job(job_type, stored_payload, status="queued")

        # Prefer Celery if available; fallback to FastAPI background task
        dispatched = False
        try:
            from workers.tasks.training import train_lora_task

            async_result = train_lora_task.delay(
                {"job_id": job_id, "job_type": job_type, "payload": stored_payload}
            )
            try:
                train_job_manager.update_job(job_id, celery_task_id=str(async_result.id))
            except Exception:
                pass
            dispatched = True
        except Exception as exc:  # noqa: BLE001
            logger.info("Celery dispatch skipped (%s), using background task", exc)

        if (not dispatched) and background_tasks is not None:
            background_tasks.add_task(_simulate_training, job_id, job_type, stored_payload)

        return {
            "job_id": job_id,
            "status": "queued",
            "received": True,
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to submit LoRA training job: %s", e)
        raise HTTPException(500, f"Failed to submit LoRA training job: {e}") from e

