from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from workers.celery_app import celery_app

from core.datasets import get_dataset_manager
from core.train.job_manager import TrainJobManager


@celery_app.task(bind=True, name="dataset_caption_task")
def dataset_caption_task(self, job_data: dict):
    """Auto-caption dataset images and write captions/*.txt + metadata.jsonl."""
    job_id = str(job_data.get("job_id") or "").strip()
    payload: Dict[str, Any] = dict(job_data.get("payload") or {})
    if not payload:
        payload = dict(job_data)

    if not job_id:
        raise ValueError("job_id is required")

    mgr = get_dataset_manager()
    world_id = mgr.validate_world_id(str(payload.get("world_id") or "default"))
    dataset_id = str(payload.get("dataset_id") or "").strip()
    if not dataset_id:
        raise ValueError("dataset_id is required")

    job_manager = TrainJobManager()
    job_manager.update_job(job_id, status="running", progress=1.0)

    dataset = mgr.get_dataset(world_id, dataset_id, limit=5000)
    dataset_path = Path(str(dataset.get("dataset_path") or ""))
    images: List[Dict[str, Any]] = list(dataset.get("items") or [])
    total = len(images)
    if total <= 0:
        job_manager.update_job(job_id, status="failed", progress=0.0, error="no images found")
        return {"job_id": job_id, "status": "failed"}

    # Best-effort VLM captioning; fallback to filename stem when unavailable.
    engine = None
    try:
        from api.dependencies import get_vlm

        engine = get_vlm()
        if hasattr(engine, "load_caption_model"):
            engine.load_caption_model()
    except Exception:
        engine = None

    captions_dir = dataset_path / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    started = time.time()
    failures: List[Dict[str, Any]] = []

    for item in images:
        try:
            image_path = str(item.get("image_path") or "")
            if not image_path:
                continue

            img = Path(image_path)
            caption_text: Optional[str] = None
            if engine is not None and hasattr(engine, "caption"):
                out = engine.caption(image=str(img))
                if isinstance(out, dict) and out.get("caption") is not None:
                    caption_text = str(out.get("caption") or "").strip()

            if not caption_text:
                caption_text = img.stem.replace("_", " ").strip()

            (captions_dir / f"{img.stem}.txt").write_text(caption_text, encoding="utf-8")
            processed += 1

            pct = max(1.0, min(99.0, (processed / total) * 100.0))
            if processed % 5 == 0 or processed == total:
                job_manager.update_job(job_id, status="running", progress=round(pct, 2))
        except Exception as exc:  # noqa: BLE001
            failures.append({"item": item.get("item_id"), "error": str(exc)})
            continue

    # Build metadata.jsonl for SDXL trainer compatibility
    try:
        mgr.build_metadata_jsonl(world_id, dataset_id)
    except Exception:
        pass

    duration = time.time() - started
    job_manager.update_job(
        job_id,
        status="completed",
        progress=100.0,
        result={
            "success": True,
            "world_id": world_id,
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "processed": processed,
            "total": total,
            "failures": failures,
            "time_taken_seconds": duration,
        },
        completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    return job_manager.get_job(job_id, auto_progress=False) or {"job_id": job_id, "status": "completed"}

