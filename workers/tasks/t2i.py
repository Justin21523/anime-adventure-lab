from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from workers.celery_app import celery_app

from core.shared_cache import get_shared_cache
from core.t2i.engine import get_t2i_engine
from core.train.job_manager import TrainJobManager


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # In rare cases where an event loop exists (e.g., embedded runner), fallback.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _t2i_file_url(output_path: str) -> str:
    cache = get_shared_cache()
    root = Path(cache.get_path("OUTPUT_DIR")).resolve()
    try:
        rel = Path(str(output_path)).resolve().relative_to(root)
        return f"/api/v1/t2i/file?path={quote(str(rel))}"
    except Exception:
        return ""

def _job_report(
    job_manager: TrainJobManager,
    job_id: str,
    *,
    stage: str,
    progress: float,
    message: str,
    meta: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update job stage/progress (best-effort). Returns False if job is cancelled/terminal."""
    try:
        existing = job_manager.get_job(job_id, auto_progress=False) or {}
        status = str(existing.get("status") or "").lower()
        if status in {"completed", "failed", "cancelled"}:
            return False
        if bool(existing.get("cancel_requested")):
            return False

        p = float(progress or 0.0)
        if not (p == p):  # NaN
            p = 0.0
        p = max(0.0, min(100.0, p))

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        events = existing.get("stage_events")
        if not isinstance(events, list):
            events = []
        event: Dict[str, Any] = {"ts": now, "stage": str(stage), "progress": round(p, 2)}
        if message:
            event["message"] = str(message)
        if meta:
            event["meta"] = meta
        events.append(event)
        if len(events) > 50:
            events = events[-50:]

        updates: Dict[str, Any] = {
            "status": "running",
            "progress": round(p, 2),
            "stage": str(stage),
            "stage_message": str(message),
            "stage_updated_at": now,
            "stage_events": events,
        }
        if not existing.get("started_at"):
            updates["started_at"] = now
        job_manager.update_job(job_id, **updates)
        return True
    except Exception:
        return True


def _update_story_session_scene_image(
    *,
    session_id: str,
    turn: Optional[int],
    job_id: str,
    scene_image: Dict[str, Any],
    job_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort patch story session JSON with the generated scene image."""
    cache = get_shared_cache()
    base_dir = Path(cache.get_output_path("games")) / "story_sessions"
    session_file = base_dir / f"{session_id}.json"
    if not session_file.exists():
        session_file = Path("outputs") / "story_sessions" / f"{session_id}.json"
    if not session_file.exists():
        return

    # Best-effort optimistic update (avoid clobbering concurrent writes).
    for _ in range(3):
        try:
            before_mtime = session_file.stat().st_mtime
            data = json.loads(session_file.read_text(encoding="utf-8"))
            story_ctx = (
                ((data.get("current_state") or {}).get("story_context") or {})
                if isinstance(data.get("current_state"), dict)
                else {}
            )
            if not isinstance(story_ctx, dict):
                story_ctx = {}
            story_ctx["last_scene_image"] = scene_image
            story_ctx["last_scene_image_job_id"] = job_id
            if isinstance(data.get("current_state"), dict):
                data["current_state"]["story_context"] = story_ctx

            history = data.get("history")
            if isinstance(history, list) and history:
                updated = False
                for entry in reversed(history):
                    if not isinstance(entry, dict):
                        continue
                    if turn is not None and int(entry.get("turn", -1) or -1) != int(turn):
                        continue
                    if "player_input" not in entry:
                        continue
                    entry["scene_image"] = scene_image
                    entry["scene_image_job_id"] = job_id
                    try:
                        artifacts = entry.get("artifacts")
                        if not isinstance(artifacts, dict):
                            artifacts = {}
                        t2i_bucket = artifacts.get("t2i")
                        if not isinstance(t2i_bucket, dict):
                            t2i_bucket = {}
                        t2i_bucket["scene_image"] = scene_image
                        t2i_bucket["scene_image_job_id"] = job_id
                        if job_snapshot:
                            t2i_bucket["job"] = job_snapshot
                        artifacts["t2i"] = t2i_bucket
                        entry["artifacts"] = artifacts
                    except Exception:
                        pass
                    updated = True
                    break
                if not updated and isinstance(history[-1], dict):
                    history[-1]["scene_image"] = scene_image
                    history[-1]["scene_image_job_id"] = job_id
                    try:
                        artifacts = history[-1].get("artifacts")
                        if not isinstance(artifacts, dict):
                            artifacts = {}
                        t2i_bucket = artifacts.get("t2i")
                        if not isinstance(t2i_bucket, dict):
                            t2i_bucket = {}
                        t2i_bucket["scene_image"] = scene_image
                        t2i_bucket["scene_image_job_id"] = job_id
                        if job_snapshot:
                            t2i_bucket["job"] = job_snapshot
                        artifacts["t2i"] = t2i_bucket
                        history[-1]["artifacts"] = artifacts
                    except Exception:
                        pass

            if session_file.exists() and session_file.stat().st_mtime != before_mtime:
                continue

            session_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        except Exception:
            continue


@celery_app.task(bind=True, name="generate_image_async")
def generate_image_async(self, request_data: dict):
    """Async image generation task (legacy helper)."""
    engine = get_t2i_engine()
    if hasattr(engine, "mock_generation"):
        engine.mock_generation = bool(request_data.get("mock", False))

    payload = {
        "prompt": request_data["prompt"],
        "negative_prompt": request_data.get("negative_prompt", ""),
        "width": request_data.get("width", 768),
        "height": request_data.get("height", 768),
        "num_inference_steps": request_data.get("steps", 25),
        "guidance_scale": request_data.get("guidance_scale", 7.5),
        "seed": request_data.get("seed", 42),
        "session_id": request_data.get("session_id") or "general",
    }
    if request_data.get("model_id") or request_data.get("model"):
        payload["model_id"] = request_data.get("model_id") or request_data.get("model")

    result = _run_async(engine.txt2img(payload))
    metadata = (result or {}).get("metadata", {}) or {}
    paths = metadata.get("output_paths") or []

    return {
        "image_path": paths[0] if paths else "",
        "metadata": metadata,
        "seed": request_data.get("seed", 42),
    }


@celery_app.task(bind=True, name="story_scene_image_task")
def story_scene_image_task(self, job_data: dict):
    """Generate a story scene image and persist result into the story session history."""
    job_id = str(job_data.get("job_id") or "").strip()
    payload: Dict[str, Any] = dict(job_data.get("payload") or {})
    if not payload:
        payload = dict(job_data)

    if not job_id:
        raise ValueError("job_id is required")

    job_manager = TrainJobManager()
    existing = job_manager.get_job(job_id, auto_progress=False) or {}
    if str(existing.get("status") or "").lower() == "cancelled":
        return existing
    task_started = time.time()
    _job_report(job_manager, job_id, stage="load", progress=1.0, message="開始生成場景圖")

    session_id = str(payload.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("session_id is required")

    turn = payload.get("turn")
    try:
        turn = int(turn) if turn is not None else None
    except Exception:
        turn = None

    scene_context: Dict[str, Any] = dict(payload.get("scene_context") or {})
    world_id = str(scene_context.get("world_id", "default") or "default").strip() or "default"
    runtime_preset_id = str(scene_context.get("runtime_preset_id") or "").strip() or None

    # Prompt generation
    from core.t2i.story_prompt_generator import StoryPromptGenerator

    generator = StoryPromptGenerator()
    _job_report(job_manager, job_id, stage="prompt", progress=8.0, message="生成提示詞")
    prompt_data = _run_async(generator.generate_from_scene(scene_context))
    positive_prompt = prompt_data.positive
    negative_prompt = prompt_data.negative

    # Apply world visual style (LoRA / prefix / negatives)
    lora_configs: List[Dict[str, Any]] = []
    model_id: Optional[str] = None
    try:
        from core.worldpacks import get_worldpack_manager

        wpm = get_worldpack_manager()
        worldpack = wpm.get_worldpack(world_id)
        visual = getattr(worldpack, "visual", None)
        if visual:
            if getattr(visual, "prompt_prefix", "") and str(visual.prompt_prefix).strip():
                positive_prompt = ", ".join(
                    [str(visual.prompt_prefix).strip(), positive_prompt.strip()]
                )
            if getattr(visual, "negative_prompt", "") and str(visual.negative_prompt).strip():
                negative_prompt = ", ".join(
                    [negative_prompt.strip(), str(visual.negative_prompt).strip()]
                )
            if getattr(visual, "base_model", None):
                model_id = str(visual.base_model)
            if getattr(visual, "default_loras", None):
                lora_configs = [
                    {"lora_id": l.lora_id, "weight": float(getattr(l, "weight", 0.8))}
                    for l in visual.default_loras
                    if getattr(l, "lora_id", None)
                ]
    except Exception:
        pass

    # Apply runtime preset bounds (best-effort)
    _job_report(job_manager, job_id, stage="preset", progress=15.0, message="套用 runtime preset")
    width = int(payload.get("width", 768) or 768)
    height = int(payload.get("height", 768) or 768)
    steps = int(payload.get("steps", 25) or 25)
    guidance_scale = float(payload.get("guidance_scale", 7.0) or 7.0)
    preset_opt: Dict[str, Any] = {}

    try:
        if runtime_preset_id:
            from core.runtime.catalog import get_runtime_preset

            preset = get_runtime_preset(runtime_preset_id) or {}
            t2i = preset.get("t2i") if isinstance(preset.get("t2i"), dict) else {}

            max_w = int(t2i.get("max_width", width) or width)
            max_h = int(t2i.get("max_height", height) or height)
            max_steps = int(t2i.get("max_steps", steps) or steps)
            width = max(256, min(width, max_w))
            height = max(256, min(height, max_h))
            steps = max(1, min(steps, max_steps))

            if not model_id and t2i.get("model_id"):
                model_id = str(t2i.get("model_id"))

            preset_opt = {
                "enable_attention_slicing": bool(t2i.get("enable_attention_slicing", True)),
                "enable_vae_slicing": bool(t2i.get("enable_vae_slicing", True)),
                "enable_vae_tiling": bool(t2i.get("enable_vae_tiling", False)),
                "enable_cpu_offload": bool(t2i.get("enable_cpu_offload", False)),
                "enable_sequential_cpu_offload": bool(t2i.get("enable_sequential_cpu_offload", False)),
            }
    except Exception:
        pass

    # Generate image
    if not _job_report(job_manager, job_id, stage="generate", progress=25.0, message="生成圖像"):
        return job_manager.get_job(job_id, auto_progress=False) or {"job_id": job_id}
    engine = get_t2i_engine()
    if hasattr(engine, "mock_generation"):
        engine.mock_generation = bool(payload.get("mock", False))

    request_payload: Dict[str, Any] = {
        "prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "seed": payload.get("seed"),
        "session_id": session_id,
    }
    if preset_opt:
        request_payload.update(preset_opt)
    if model_id:
        request_payload["model_id"] = model_id
    if lora_configs:
        request_payload["lora_configs"] = lora_configs

    started = time.time()
    try:
        result = _run_async(engine.txt2img(request_payload))
    except Exception as exc:  # noqa: BLE001
        duration = time.time() - task_started
        job_manager.update_job(
            job_id,
            status="failed",
            progress=float((job_manager.get_job(job_id, auto_progress=False) or {}).get("progress") or 0.0),
            stage="failed",
            stage_message="生成失敗",
            error=str(exc),
            duration_seconds=duration,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        raise
    generation_duration = time.time() - started

    metadata = (result or {}).get("metadata", {}) or {}
    output_paths = metadata.get("output_paths") or []
    output_path = output_paths[0] if output_paths else ""

    scene_image = {
        "image_url": _t2i_file_url(output_path) if output_path else "",
        "prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "generation_time": float(metadata.get("generation_time", generation_duration) or generation_duration),
        "seed": metadata.get("seed") or payload.get("seed"),
        "width": int((metadata.get("parameters") or {}).get("width") or request_payload["width"]),
        "height": int((metadata.get("parameters") or {}).get("height") or request_payload["height"]),
    }

    current = job_manager.get_job(job_id, auto_progress=False) or {}
    if str(current.get("status") or "").lower() == "cancelled":
        return current

    _job_report(job_manager, job_id, stage="persist", progress=92.0, message="寫回 session artifacts")
    _job_report(job_manager, job_id, stage="done", progress=99.0, message="完成")
    job_manager.update_job(
        job_id,
        status="completed",
        progress=100.0,
        result_path=output_path or None,
        result={"scene_image": scene_image, "metadata": metadata},
        duration_seconds=time.time() - task_started,
        completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    snapshot: Dict[str, Any] = {}
    try:
        job = job_manager.get_job(job_id, auto_progress=False) or {}
        stage_events = job.get("stage_events")
        if not isinstance(stage_events, list):
            stage_events = []
        snapshot = {
            "job_id": job_id,
            "job_type": "scene_image",
            "status": job.get("status"),
            "stage": job.get("stage"),
            "stage_message": job.get("stage_message"),
            "progress": job.get("progress"),
            "started_at": job.get("started_at") or job.get("created_at"),
            "duration_seconds": job.get("duration_seconds"),
            "stage_events": [
                {
                    "ts": e.get("ts"),
                    "stage": e.get("stage"),
                    "progress": e.get("progress"),
                    "message": e.get("message"),
                    "meta": e.get("meta"),
                }
                for e in (stage_events or [])[-30:]
                if isinstance(e, dict)
            ],
        }
    except Exception:
        snapshot = {}

    _update_story_session_scene_image(
        session_id=session_id,
        turn=turn,
        job_id=job_id,
        scene_image=scene_image,
        job_snapshot=snapshot or None,
    )

    return {"job_id": job_id, "scene_image": scene_image, "output_path": output_path}


@celery_app.task(bind=True, name="story_character_portrait_task")
def story_character_portrait_task(self, job_data: dict):
    """Generate a character portrait and update the WorldPack."""
    job_id = str(job_data.get("job_id") or "").strip()
    payload: Dict[str, Any] = dict(job_data.get("payload") or {})
    if not payload:
        payload = dict(job_data)

    if not job_id:
        raise ValueError("job_id is required")

    job_manager = TrainJobManager()
    existing = job_manager.get_job(job_id, auto_progress=False) or {}
    if str(existing.get("status") or "").lower() == "cancelled":
        return existing
    
    task_started = time.time()
    _job_report(job_manager, job_id, stage="load", progress=1.0, message="開始生成角色立繪")

    character_name = str(payload.get("character_name") or "神秘角色")
    appearance_desc = str(payload.get("appearance_desc") or "anime style character")
    world_id = str(payload.get("world_id") or "default")
    character_id = str(payload.get("character_id") or "")
    visual_style = payload.get("visual_style")

    from core.story.t2i_integration import get_t2i_integration
    t2i_integration = get_t2i_integration()

    _job_report(job_manager, job_id, stage="generate", progress=20.0, message=f"正在生成 {character_name} 的立繪")
    
    # Run generation
    try:
        result = _run_async(t2i_integration.generate_character_portrait_image(
            character_name=character_name,
            appearance_desc=appearance_desc,
            world_id=world_id,
            visual_style=visual_style
        ))
    except Exception as exc:
        duration = time.time() - task_started
        job_manager.update_job(
            job_id,
            status="failed",
            error=f"Portrait generation failed: {exc}",
            duration_seconds=duration,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        raise

    if not result:
        _job_report(job_manager, job_id, stage="failed", progress=0.0, message="生成失敗")
        job_manager.update_job(job_id, status="failed", error="Generation returned no result")
        return {"job_id": job_id, "success": False}

    _job_report(job_manager, job_id, stage="update_world", progress=80.0, message="更新世界設定")

    # Update WorldPack character image_url
    image_url = getattr(result, "image_url", "")
    
    # --- NEW: Background Removal for Portraits ---
    if image_url and not image_url.startswith("data:"):
        try:
            from rembg import remove
            from PIL import Image
            import urllib.parse
            
            _job_report(job_manager, job_id, stage="rembg", progress=85.0, message="正在移除背景...")
            
            # Extract relative path from URL
            parsed_url = urllib.parse.urlparse(image_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            rel_path = query_params.get("path", [None])[0]
            
            if rel_path:
                cache = get_shared_cache()
                root = Path(cache.get_path("OUTPUT_DIR")).resolve()
                abs_path = (root / rel_path).resolve()
                
                if abs_path.exists():
                    logger.info(f"Removing background for {abs_path}")
                    input_img = Image.open(abs_path)
                    # Convert to RGBA if not already
                    if input_img.mode != "RGBA":
                        input_img = input_img.convert("RGBA")
                    
                    output_img = remove(input_img)
                    
                    # Save back to the same path (as PNG to support transparency)
                    # Note: we might want to change extension if it was .jpg
                    if abs_path.suffix.lower() != ".png":
                        new_abs_path = abs_path.with_suffix(".png")
                        output_img.save(new_abs_path, "PNG")
                        # Update URL to point to the new PNG
                        new_rel = new_abs_path.relative_to(root)
                        image_url = f"/api/v1/t2i/file?path={urllib.parse.quote(str(new_rel))}"
                    else:
                        output_img.save(abs_path, "PNG")
                        
                    logger.info(f"Background removed successfully: {image_url}")
        except Exception as rembg_exc:
            logger.error(f"Failed to remove background: {rembg_exc}")
            # Non-critical, continue with original image
    # --- END NEW ---

    if image_url and character_id:
        try:
            from core.worldpacks import get_worldpack_manager
            wpm = get_worldpack_manager()
            worldpack = wpm.get_worldpack(world_id)
            if worldpack:
                updated = False
                for char in worldpack.characters:
                    if char.character_id == character_id:
                        char.image_url = image_url
                        updated = True
                        break
                if updated:
                    wpm.update_worldpack(world_id, worldpack)
                    logger.info(f"Updated image_url for character {character_id} in world {world_id}")
        except Exception as exc:
            logger.error(f"Failed to update worldpack character: {exc}")

    _job_report(job_manager, job_id, stage="done", progress=100.0, message="完成")
    job_manager.update_job(
        job_id,
        status="completed",
        progress=100.0,
        result={"image_url": image_url},
        duration_seconds=time.time() - task_started,
        completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    return {"job_id": job_id, "image_url": image_url, "success": True}
