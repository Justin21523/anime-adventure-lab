# api/routers/t2i.py
"""
Text-to-Image Generation Router
"""

import logging
from fastapi import APIRouter, HTTPException
from schemas.t2i import T2IRequest, T2IResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/txt2img", response_model=T2IResponse)
async def text_to_image(request: T2IRequest):
    """Generate image from text prompt"""
    try:
        # Mock T2I implementation
        return T2IResponse(  # type: ignore
            image_path="/tmp/generated_image.png",
            prompt=request.prompt,
            model_used="stable-diffusion-xl",
            parameters=request.parameters,
            generation_info={
                "steps": request.parameters.steps,  # type: ignore
                "guidance_scale": request.parameters.guidance_scale,  # type: ignore
                "seed": request.parameters.seed or 42,  # type: ignore
            },
        )

    except Exception as e:
        raise HTTPException(500, f"T2I generation failed: {str(e)}")
