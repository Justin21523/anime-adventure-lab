# api/routers/finetune.py
"""
Fine-tuning API endpoints for LoRA training
"""
from __future__ import annotations
import os
import uuid
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import (
    APIRouter,
    HTTPException,
    BackgroundTasks,
    UploadFile,
    File,
    Form,
    Depends,
)
from pydantic import BaseModel, Field
import yaml

from ..dependencies import get_cache  # <-- unified cache provider

router = APIRouter(prefix="/finetune", tags=["fine-tuning"])


class LoRATrainingRequest(BaseModel):
    """LoRA training request schema"""

    config_path: str = Field(..., description="Path to training config YAML file")
    run_id: Optional[str] = Field(
        None, description="Custom run ID (auto-generated if not provided)"
    )
    character_name: Optional[str] = Field(
        None, description="Character name for preset registration"
    )
    notes: Optional[str] = Field(None, description="Training notes")


class LoRATrainingResponse(BaseModel):
    """LoRA training response schema"""

    job_id: str
    status: str
    config_path: str
    estimated_duration_minutes: int
    created_at: str


class TrainingStatus(BaseModel):
    """Training status response schema"""

    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# lazy workers

_job_tracker = None  # singleton instance


def _lazy_training_modules():
    """Import training task and job tracker on demand (avoid import-time failures)."""
    try:
        from workers.tasks.training import train_lora_task  # type: ignore
        from workers.utils.job_tracker import JobTracker  # type: ignore

        return train_lora_task, JobTracker
    except Exception as e:  # ImportError or others
        raise HTTPException(
            status_code=503, detail=f"Training workers unavailable: {e}"
        )


def _get_job_tracker():
    global _job_tracker
    if _job_tracker is None:
        _, JobTracker = _lazy_training_modules()
        _job_tracker = JobTracker()
    return _job_tracker


@router.post("/lora", response_model=LoRATrainingResponse)
async def submit_lora_training(
    background_tasks: BackgroundTasks, request: LoRATrainingRequest
):
    """
    Submit LoRA training job

    The training will run in the background using Celery.
    Use the returned job_id to check status.
    """
    # Validate config file exists
    config_path = Path(request.config_path)
    if not config_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Config file not found: {request.config_path}"
        )

    # Load config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config file: {e}")

    # Validate dataset path
    dataset_root = Path(config["dataset"]["root"])
    if not dataset_root.exists():
        raise HTTPException(
            status_code=404, detail=f"Dataset not found: {dataset_root}"
        )

    # Estimate duration (rough heuristic)
    train_steps = config.get("train_steps", 4000)
    grad_accum = config.get("gradient_accumulation_steps", 8)
    effective_batches = max(train_steps / max(grad_accum, 1), 1)
    estimated_minutes = int(effective_batches * 1.2 / 60)  # ~1.2s / batch

    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "config_path": str(config_path),
        "run_id": request.run_id,
        "character_name": request.character_name,
        "notes": request.notes,
        "config": config,
        "estimated_duration_minutes": estimated_minutes,
    }

    # Enqueue via Celery (lazy import)
    train_lora_task, _ = _lazy_training_modules()
    task = train_lora_task.delay(job_data)

    # Track job
    jt = _get_job_tracker()
    jt.create_job(
        job_id=job_id, task_id=task.id, job_type="lora_training", config=job_data
    )

    return LoRATrainingResponse(
        job_id=job_id,
        status="pending",
        config_path=str(config_path),
        estimated_duration_minutes=estimated_minutes,
        created_at=datetime.now().isoformat(),
    )


@router.get("/jobs/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get training job status and progress."""
    jt = _get_job_tracker()
    job = jt.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return TrainingStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@router.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    jt = _get_job_tracker()
    job = jt.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ["pending", "running"]:
        raise HTTPException(
            status_code=400, detail=f"Cannot cancel job with status: {job['status']}"
        )

    # Revoke Celery task
    from workers.celery_app import celery_app  # type: ignore

    celery_app.control.revoke(job["task_id"], terminate=True)

    jt.update_job(job_id, {"status": "cancelled"})
    return {"message": "Job cancelled successfully"}


@router.get("/jobs")
async def list_training_jobs(
    status: Optional[str] = None, limit: int = 20, offset: int = 0
):
    """List training jobs with optional filtering."""
    jt = _get_job_tracker()
    jobs = jt.list_jobs(
        job_type="lora_training", status=status, limit=limit, offset=offset
    )
    return {"jobs": jobs, "total": len(jobs), "limit": limit, "offset": offset}


@router.post("/upload-dataset")
async def upload_dataset(
    files: list[UploadFile] = File(...),
    character_name: str = Form(...),
    instance_token: str = Form(default="<token>"),
    notes: Optional[str] = Form(default=None),
    cache=Depends(get_cache),
):
    """
    Upload dataset files for training.
    Expected:
      - images: .png/.jpg
      - captions: .txt (same basename as image)
      - splits: train.txt, val.txt (optional)
    """
    # Use unified cache path (DATASETS_RAW)
    dataset_root = Path(cache.get_path("DATASETS_RAW")) / f"anime-char-{character_name}"
    image_dir = dataset_root / "images"
    caption_dir = dataset_root / "captions"
    splits_dir = dataset_root / "splits"
    for d in (image_dir, caption_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    image_files: list[str] = []
    caption_files: list[str] = []
    split_files: list[str] = []

    for file in files:
        if not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower()
        content = await file.read()

        if suffix in {".png", ".jpg", ".jpeg"}:
            (image_dir / file.filename).write_bytes(content)
            image_files.append(file.filename)
        elif suffix == ".txt" and Path(file.filename).stem not in {
            "train",
            "val",
            "test",
        }:
            (caption_dir / file.filename).write_text(
                content.decode("utf-8"), encoding="utf-8"
            )
            caption_files.append(file.filename)
        elif Path(file.filename).name in {"train.txt", "val.txt", "test.txt"}:
            (splits_dir / file.filename).write_text(
                content.decode("utf-8"), encoding="utf-8"
            )
            split_files.append(Path(file.filename).name)

    # Auto-generate train.txt if missing
    if "train.txt" not in split_files and image_files:
        (splits_dir / "train.txt").write_text("\n".join(sorted(image_files)))

    metadata = {
        "character_name": character_name,
        "instance_token": instance_token,
        "notes": notes,
        "files": {
            "images": len(image_files),
            "captions": len(caption_files),
            "splits": split_files,
        },
        "created_at": datetime.now().isoformat(),
    }
    (dataset_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return {
        "message": "Dataset uploaded successfully",
        "dataset_path": str(dataset_root),
        "metadata": metadata,
    }


@router.get("/presets")
async def list_lora_presets(cache=Depends(get_cache)):
    """List available LoRA presets (scanned from MODELS_LORA)."""
    lora_root = Path(cache.get_path("MODELS_LORA"))
    if not lora_root.exists():
        return {"presets": []}

    presets = []
    for preset_dir in lora_root.iterdir():
        if not preset_dir.is_dir():
            continue
        # prefer /final, otherwise latest checkpoint
        final_dir = preset_dir / "final"
        if final_dir.exists():
            model_path = final_dir
        else:
            ckpts = sorted(preset_dir.glob("checkpoints/step_*"))
            if not ckpts:
                continue
            model_path = ckpts[-1]

        metadata = {}
        meta_file = preset_dir / "metadata.json"
        if meta_file.exists():
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))

        presets.append(
            {
                "id": preset_dir.name,
                "path": str(model_path),
                "metadata": metadata,
                "created_at": metadata.get("created_at", "unknown"),
            }
        )

    return {"presets": sorted(presets, key=lambda x: x["created_at"], reverse=True)}


@router.post("/presets/{preset_id}/register")
async def register_lora_preset(
    preset_id: str,
    character_name: str = Form(...),
    base_model: str = Form(...),
    lora_scale: float = Form(default=0.75),
    description: Optional[str] = Form(default=None),
    cache=Depends(get_cache),
):
    """Register a trained LoRA as a style preset for T2I generation."""
    lora_dir = Path(cache.get_path("MODELS_LORA")) / preset_id
    if not lora_dir.exists():
        raise HTTPException(status_code=404, detail="LoRA not found")

    final_dir = lora_dir / "final"
    if final_dir.exists():
        model_path = final_dir / "unet_lora"
    else:
        ckpts = sorted(lora_dir.glob("checkpoints/step_*/unet_lora"))
        if not ckpts:
            raise HTTPException(status_code=400, detail="No trained model found")
        model_path = ckpts[-1]

    preset_config = {
        "style_id": preset_id,
        "character_name": character_name,
        "base_model": base_model,
        "lora_path": str(model_path),
        "lora_scale": lora_scale,
        "description": description,
        "registered_at": datetime.now().isoformat(),
    }

    presets_dir = Path("configs/presets")
    presets_dir.mkdir(parents=True, exist_ok=True)
    preset_file = presets_dir / f"{preset_id}.yaml"
    preset_file.write_text(
        yaml.safe_dump(preset_config, sort_keys=False), encoding="utf-8"
    )

    return {
        "message": "LoRA preset registered successfully",
        "preset_id": preset_id,
        "config_path": str(preset_file),
        "config": preset_config,
    }
