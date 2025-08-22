# api/routers/finetune.py
"""
Fine-tuning API endpoints for LoRA training
"""

import os
import uuid
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel, Field
import yaml

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from workers.tasks.training import train_lora_task
from workers.utils.job_tracker import JobTracker

router = APIRouter(prefix="/finetune", tags=["fine-tuning"])

# Job tracker instance
job_tracker = JobTracker()


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

    # Load and validate config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config file: {str(e)}")

    # Validate dataset exists
    dataset_root = Path(config["dataset"]["root"])
    if not dataset_root.exists():
        raise HTTPException(
            status_code=404, detail=f"Dataset not found: {dataset_root}"
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Estimate training duration (rough estimate)
    train_steps = config.get("train_steps", 4000)
    batch_size = config.get("batch_size", 1)
    grad_accum = config.get("gradient_accumulation_steps", 8)

    # Rough estimate: ~1 second per effective batch on RTX 3090
    effective_batches = train_steps / grad_accum
    estimated_minutes = int(effective_batches * 1.2 / 60)  # 1.2 sec per batch

    # Prepare job data
    job_data = {
        "job_id": job_id,
        "config_path": str(config_path),
        "run_id": request.run_id,
        "character_name": request.character_name,
        "notes": request.notes,
        "config": config,
        "estimated_duration_minutes": estimated_minutes,
    }

    # Submit to Celery
    task = train_lora_task.delay(job_data)

    # Track job
    job_tracker.create_job(
        job_id=job_id,
        task_id=task.id,
        job_type="lora_training",
        config=job_data,
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
    """Get training job status and progress"""
    job = job_tracker.get_job(job_id)
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
    """Cancel a running training job"""
    job = job_tracker.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ["pending", "running"]:
        raise HTTPException(
            status_code=400, detail=f"Cannot cancel job with status: {job['status']}"
        )

    # Cancel Celery task
    from workers.celery_app import celery_app

    celery_app.control.revoke(job["task_id"], terminate=True)

    # Update job status
    job_tracker.update_job(job_id, {"status": "cancelled"})

    return {"message": "Job cancelled successfully"}


@router.get("/jobs")
async def list_training_jobs(
    status: Optional[str] = None, limit: int = 20, offset: int = 0
):
    """List training jobs with optional filtering"""
    jobs = job_tracker.list_jobs(
        job_type="lora_training", status=status, limit=limit, offset=offset
    )

    return {
        "jobs": jobs,
        "total": len(jobs),
        "limit": limit,
        "offset": offset,
    }


@router.post("/upload-dataset")
async def upload_dataset(
    files: list[UploadFile] = File(...),
    character_name: str = Form(...),
    instance_token: str = Form(default="<token>"),
    notes: Optional[str] = Form(default=None),
):
    """
    Upload dataset files for training

    Expected files:
    - images: .png/.jpg files
    - captions: .txt files with same basename as images
    - splits: train.txt, val.txt (optional)
    """
    # Create dataset directory
    ai_cache_root = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
    dataset_dir = Path(ai_cache_root) / "datasets" / f"anime-char-{character_name}"

    image_dir = dataset_dir / "images"
    caption_dir = dataset_dir / "captions"
    splits_dir = dataset_dir / "splits"

    for dir_path in [image_dir, caption_dir, splits_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Process uploaded files
    image_files = []
    caption_files = []
    split_files = []

    for file in files:
        if not file.filename:
            continue

        file_path = Path(file.filename)
        content = await file.read()

        if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            # Save image
            save_path = image_dir / file.filename
            with open(save_path, "wb") as f:
                f.write(content)
            image_files.append(file.filename)

        elif (
            file_path.suffix.lower() == ".txt"
            and file_path.stem != "train"
            and file_path.stem != "val"
        ):
            # Save caption
            save_path = caption_dir / file.filename
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content.decode("utf-8"))
            caption_files.append(file.filename)

        elif file_path.name in ["train.txt", "val.txt", "test.txt"]:
            # Save split file
            save_path = splits_dir / file.filename
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content.decode("utf-8"))
            split_files.append(file.filename)

    # Generate train.txt if not provided
    if "train.txt" not in split_files and image_files:
        train_list_path = splits_dir / "train.txt"
        with open(train_list_path, "w") as f:
            for img_file in sorted(image_files):
                f.write(f"{img_file}\n")
        split_files.append("train.txt")

    # Create metadata
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

    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "message": "Dataset uploaded successfully",
        "dataset_path": str(dataset_dir),
        "metadata": metadata,
    }


@router.get("/presets")
async def list_lora_presets():
    """List available LoRA presets"""
    ai_cache_root = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
    presets_dir = Path(ai_cache_root) / "models" / "lora"

    if not presets_dir.exists():
        return {"presets": []}

    presets = []
    for preset_dir in presets_dir.iterdir():
        if preset_dir.is_dir():
            # Look for final model or latest checkpoint
            final_dir = preset_dir / "final"
            if final_dir.exists():
                model_path = final_dir
            else:
                # Find latest checkpoint
                checkpoints = list(preset_dir.glob("checkpoints/step_*"))
                if checkpoints:
                    model_path = sorted(checkpoints)[-1]
                else:
                    continue

            # Load metadata if available
            metadata = {}
            metadata_file = preset_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

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
):
    """Register trained LoRA as a style preset for T2I generation"""
    ai_cache_root = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")

    # Validate LoRA exists
    lora_dir = Path(ai_cache_root) / "models" / "lora" / preset_id
    if not lora_dir.exists():
        raise HTTPException(status_code=404, detail="LoRA not found")

    # Find model path
    final_dir = lora_dir / "final"
    if final_dir.exists():
        model_path = final_dir / "unet_lora"
    else:
        checkpoints = list(lora_dir.glob("checkpoints/step_*/unet_lora"))
        if not checkpoints:
            raise HTTPException(status_code=400, detail="No trained model found")
        model_path = sorted(checkpoints)[-1]

    # Create preset config
    preset_config = {
        "style_id": preset_id,
        "character_name": character_name,
        "base_model": base_model,
        "lora_path": str(model_path),
        "lora_scale": lora_scale,
        "description": description,
        "registered_at": datetime.now().isoformat(),
    }

    # Save preset config
    presets_config_dir = Path("configs/presets")
    presets_config_dir.mkdir(parents=True, exist_ok=True)

    preset_file = presets_config_dir / f"{preset_id}.yaml"
    with open(preset_file, "w") as f:
        yaml.dump(preset_config, f)

    return {
        "message": "LoRA preset registered successfully",
        "preset_id": preset_id,
        "config_path": str(preset_file),
        "config": preset_config,
    }
