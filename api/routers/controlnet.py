# api/routers/controlnet.py
"""
ControlNet Router
"""

import logging
from fastapi import APIRouter, HTTPException, File, UploadFile
from schemas.controlnet import ControlNetRequest, ControlNetResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/controlnet/generate", response_model=ControlNetResponse)
async def controlnet_generate(
    control_image: UploadFile = File(...),
    prompt: str = "a beautiful landscape",
    controlnet_type: str = "pose",
    conditioning_scale: float = 1.0,
):
    """Generate image with ControlNet conditioning"""
    try:
        # Mock ControlNet implementation
        return ControlNetResponse(
            image_path="/tmp/controlnet_output.png",
            prompt=prompt,
            controlnet_type=controlnet_type,
            parameters={  # type: ignore
                "controlnet_type": controlnet_type,
                "conditioning_scale": conditioning_scale,
                "guidance_scale": 7.5,
                "steps": 20,
            },
        )
    except Exception as e:
        raise HTTPException(500, f"ControlNet generation failed: {str(e)}")
