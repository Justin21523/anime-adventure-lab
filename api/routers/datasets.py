"""
Dataset Builder Router (World Studio)

Provides:
- Upload zip of images into a per-world dataset folder
- List datasets / items
- Edit captions & tags
- Build `metadata.jsonl` for SDXL LoRA training
- (Optional) auto-caption via VLM as a background job
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from core.datasets import get_dataset_manager
from core.shared_cache import get_shared_cache
from core.train.job_manager import TrainJobManager
from schemas.datasets import (
    DatasetBuildMetadataResponse,
    DatasetCaptionJobResponse,
    DatasetDetail,
    DatasetItemUpdateRequest,
    DatasetItemUpdateResponse,
    DatasetListResponse,
    DatasetSummary,
    DatasetUploadResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/datasets/file")
async def get_dataset_file(path: str):
    """Serve a dataset file under DATASETS_PROCESSED (safe, read-only)."""
    cache = get_shared_cache()
    root = Path(cache.get_path("DATASETS_PROCESSED")).resolve()
    rel = Path(str(path or "").lstrip("/"))
    if ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid path")

    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(target))


@router.get("/datasets/{world_id}", response_model=DatasetListResponse)
async def list_datasets(world_id: str):
    try:
        mgr = get_dataset_manager()
        datasets = mgr.list_datasets(world_id)
        return DatasetListResponse(
            world_id=mgr.validate_world_id(world_id),
            datasets=[DatasetSummary(**d) for d in datasets],
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/datasets/{world_id}/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(world_id: str, dataset_id: str, limit: int = Query(200, ge=1, le=2000)):
    try:
        mgr = get_dataset_manager()
        data = mgr.get_dataset(world_id, dataset_id, limit=int(limit))
        return DatasetDetail(**data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/datasets/{world_id}/upload_zip", response_model=DatasetUploadResponse)
async def upload_dataset_zip(
    world_id: str,
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix != ".zip":
        raise HTTPException(status_code=400, detail="Only .zip is supported for now")

    mgr = get_dataset_manager()
    safe_world_id = mgr.validate_world_id(world_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        dataset = mgr.create_from_zip(world_id=safe_world_id, zip_path=tmp_path, name=name)
        return DatasetUploadResponse(dataset=DatasetDetail(**dataset))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


@router.put("/datasets/{world_id}/{dataset_id}/items/{item_id}", response_model=DatasetItemUpdateResponse)
async def update_dataset_item(world_id: str, dataset_id: str, item_id: str, request: DatasetItemUpdateRequest):
    try:
        mgr = get_dataset_manager()
        item = mgr.update_item(
            world_id=world_id,
            dataset_id=dataset_id,
            item_id=item_id,
            caption=request.caption,
            tags=request.tags,
        )
        return DatasetItemUpdateResponse(item=item)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/datasets/{world_id}/{dataset_id}/build_metadata", response_model=DatasetBuildMetadataResponse)
async def build_dataset_metadata(world_id: str, dataset_id: str):
    try:
        mgr = get_dataset_manager()
        out = mgr.build_metadata_jsonl(world_id, dataset_id)
        return DatasetBuildMetadataResponse(
            metadata_jsonl_path=str(out["metadata_jsonl_path"]),
            total_images=int(out["total_images"]),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/datasets/{world_id}/{dataset_id}/auto_caption", response_model=DatasetCaptionJobResponse)
async def auto_caption_dataset(world_id: str, dataset_id: str):
    """
    Optional: run VLM captioning as a background job.

    If VLM is not available in the current environment, the job may fail; callers can still edit captions manually.
    """
    mgr = get_dataset_manager()
    safe_world_id = mgr.validate_world_id(world_id)
    try:
        # Verify dataset exists
        mgr.get_dataset(safe_world_id, dataset_id, limit=1)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    job_manager = TrainJobManager()
    job_id = job_manager.create_job(
        "dataset_caption",
        {"world_id": safe_world_id, "dataset_id": str(dataset_id)},
        status="queued",
    )

    try:
        from workers.tasks.datasets import dataset_caption_task

        async_result = dataset_caption_task.delay({"job_id": job_id, "payload": {"world_id": safe_world_id, "dataset_id": str(dataset_id)}})
        job_manager.update_job(job_id, celery_task_id=str(async_result.id))
    except Exception as exc:  # noqa: BLE001
        logger.info("Celery dispatch skipped (%s), auto-caption not started", exc)
        job_manager.update_job(job_id, status="failed", error="celery dispatch unavailable")
        raise HTTPException(status_code=500, detail="Celery not available for auto-caption") from exc

    return DatasetCaptionJobResponse(job_id=job_id)

