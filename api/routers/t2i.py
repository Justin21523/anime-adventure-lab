# api/routers/t2i.py
"""Text-to-Image Generation Router."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from core.shared_cache import get_shared_cache
from schemas.t2i import (
    T2IGenerateRequest,
    T2IGenerateResponse,
    T2IGeneratedImage,
    T2IParameters,
    T2IRequest,
    T2IResponse,
)
from api.dependencies import get_t2i

logger = logging.getLogger(__name__)
router = APIRouter()


async def _run_txt2img(engine, request: T2IRequest) -> T2IResponse:
    params: T2IParameters = request.parameters or T2IParameters()
    payload = {
        "prompt": request.prompt,
        "negative_prompt": params.negative_prompt or "",
        "width": params.width,
        "height": params.height,
        "num_inference_steps": params.steps,
        "guidance_scale": params.guidance_scale,
        "seed": params.seed,
        "model": params.model,
    }

    result = await engine.txt2img(payload)
    metadata = result.get("metadata", {}) or {}
    output_paths = metadata.get("output_paths") or []

    generation_info = {
        **metadata,
        "images_base64": result.get("images", []),
    }

    return T2IResponse(  # type: ignore
        success=True,
        image_path=output_paths[0] if output_paths else "",
        prompt=request.prompt,
        model_used=metadata.get("model_used") or params.model,
        parameters=params,
        generation_info=generation_info,
    )


@router.post("/t2i/generate", response_model=T2IGenerateResponse)
async def generate_image(
    request: T2IGenerateRequest,
    use_mock: bool = Query(False, description="Force mock generation (test only)"),
    engine=Depends(get_t2i),
):
    """Generate image from text prompt using the shared T2I engine."""
    try:
        # Allow optional runtime override for testing
        if hasattr(engine, "mock_generation"):
            engine.mock_generation = bool(use_mock)
        payload: Dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "batch_size": request.num_images,
            "session_id": request.session_id,
        }
        if request.model_id:
            payload["model_id"] = request.model_id

        if request.loras:
            payload["lora_configs"] = [
                {"lora_id": l.name, "weight": float(l.weight)} for l in request.loras if l.name
            ]

        result = await engine.txt2img(payload)
        metadata = result.get("metadata", {}) or {}
        images_base64: List[str] = list(result.get("images") or [])

        images: List[T2IGeneratedImage] = []
        for img_b64 in images_base64[: request.num_images]:
            images.append(
                T2IGeneratedImage(
                    image_url=f"data:image/png;base64,{img_b64}",
                    seed=request.seed,
                    prompt=request.prompt,
                    metadata={"model_used": metadata.get("model_used"), "output_paths": metadata.get("output_paths")},
                )
            )

        return T2IGenerateResponse(
            images=images,
            generation_time=float(metadata.get("generation_time", 0.0) or 0.0),
            model_used=str(metadata.get("model_used") or engine.current_model_id or ""),
        )
    except Exception as e:  # noqa: BLE001
        logger.error("T2I generation failed: %s", e)
        raise HTTPException(500, f"T2I generation failed: {str(e)}") from e


@router.post("/t2i/txt2img", response_model=T2IResponse)
async def text_to_image(request: T2IRequest, engine=Depends(get_t2i)):
    """Alias endpoint for T2I generation."""
    return await _run_txt2img(engine, request)  # type: ignore


@router.get("/t2i/status")
async def t2i_status(engine=Depends(get_t2i)):
    """Return current T2I engine status and stats."""
    return engine.get_status()


@router.get("/t2i/models")
async def list_t2i_models(engine=Depends(get_t2i)):
    """List configured diffusion models and load state."""
    try:
        return {
            "models": engine.model_config_manager.list_available_models(),  # type: ignore
            "current_model": engine.current_model_id,
            "mock_mode": getattr(engine, "mock_generation", False),
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to list T2I models: %s", e)
        raise HTTPException(500, f"Failed to list models: {str(e)}") from e


@router.get("/t2i/loras")
async def list_t2i_loras(engine=Depends(get_t2i)):
    """List available LoRA adapters discovered on disk."""
    try:
        loras = engine.lora_manager.list_available_loras()  # type: ignore
        return {"loras": loras, "total": len(loras)}
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to list LoRAs: %s", e)
        raise HTTPException(500, f"Failed to list LoRAs: {str(e)}") from e


@router.get("/t2i/controlnets")
async def list_controlnets(engine=Depends(get_t2i)):
    """List available ControlNet preprocessors."""
    try:
        return {"controlnets": engine.controlnet_manager.list_available_controlnets()}  # type: ignore
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to list ControlNets: %s", e)
        raise HTTPException(500, f"Failed to list ControlNets: {str(e)}") from e


@router.get("/t2i/file")
async def get_t2i_file(path: str):
    """Serve a generated file under OUTPUT_DIR (safe, read-only)."""
    cache = get_shared_cache()
    root = Path(cache.get_path("OUTPUT_DIR")).resolve()
    rel = Path(str(path or "").lstrip("/"))
    if ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid path")

    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(target))


@router.get("/t2i/history")
async def get_t2i_history(
    session_id: Optional[str] = Query(None, description="Optional story session id to filter"),
    limit: int = Query(30, ge=1, le=200),
):
    """List recent T2I generations by reading OUTPUT_DIR/t2i sidecar JSON metadata."""
    cache = get_shared_cache()
    root = Path(cache.get_path("OUTPUT_DIR")).resolve()
    base = (root / "t2i").resolve()
    if not base.exists():
        return {"history": [], "total": 0}

    if session_id:
        base = (base / str(session_id)).resolve()
        if not base.exists():
            return {"history": [], "total": 0}

    # Collect metadata files (newest first)
    metas = sorted(base.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    history: List[Dict[str, Any]] = []
    for meta_path in metas:
        if len(history) >= limit:
            break
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        image_path = meta_path.with_suffix(".png")
        if not image_path.exists():
            continue

        gen_params = data.get("generation_params") or {}
        prompt = str(gen_params.get("prompt") or "")
        negative = str(gen_params.get("negative_prompt") or "")
        width = int(gen_params.get("width") or 0) or int((gen_params.get("parameters") or {}).get("width") or 0) or 0
        height = int(gen_params.get("height") or 0) or int((gen_params.get("parameters") or {}).get("height") or 0) or 0
        steps = int(gen_params.get("num_inference_steps") or gen_params.get("steps") or 0) or 0
        cfg_scale = float(gen_params.get("guidance_scale") or 0.0) or 0.0
        seed = gen_params.get("seed")

        lora_configs = gen_params.get("lora_configs") or gen_params.get("loras") or []
        loras_out = []
        if isinstance(lora_configs, list):
            for l in lora_configs[:5]:
                if isinstance(l, dict):
                    name = str(l.get("lora_id") or l.get("name") or "").strip()
                    if not name:
                        continue
                    loras_out.append({"name": name, "weight": float(l.get("weight", 0.8) or 0.8)})

        # Build safe file URL (relative to OUTPUT_DIR)
        try:
            rel = image_path.resolve().relative_to(root)
            image_url = f"/api/v1/t2i/file?path={quote(str(rel))}"
        except Exception:
            image_url = ""

        history.append(
            {
                "id": meta_path.stem,
                "timestamp": data.get("generation_timestamp") or data.get("timestamp") or None,
                "prompt": prompt,
                "negative_prompt": negative or None,
                "images": [{"image_url": image_url, "seed": seed or 0, "prompt": prompt, "metadata": {}}],
                "settings": {
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "seed": seed,
                    "loras": loras_out or None,
                },
            }
        )

    return {"history": history, "total": len(history)}
