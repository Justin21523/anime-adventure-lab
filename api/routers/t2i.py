# api/routers/t2i.py
"""Text-to-Image Generation Router."""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from schemas.t2i import T2IRequest, T2IResponse, T2IParameters
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


@router.post("/t2i/generate", response_model=T2IResponse)
async def generate_image(
    request: T2IRequest,
    use_mock: bool = Query(False, description="Force mock generation (test only)"),
    engine=Depends(get_t2i),
):
    """Generate image from text prompt using the shared T2I engine."""
    try:
        # Allow optional runtime override for testing
        if hasattr(engine, "mock_generation"):
            engine.mock_generation = bool(use_mock)
        return await _run_txt2img(engine, request)
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
        return {"loras": engine.lora_manager.list_available_loras()}  # type: ignore
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
