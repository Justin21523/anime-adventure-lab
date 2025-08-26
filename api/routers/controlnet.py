# api/routers/controlnet.py

from fastapi import APIRouter, HTTPException, File, UploadFile, Depends
from api.schemas import ControlNetRequest, ControlNetResponse
from ..dependencies import get_controlnet
import torch
import time
import base64
import io
from PIL import Image

router = APIRouter(tags=["controlnet"])


@router.post(
    "/controlnet/pose",
    response_model=ControlNetResponse,
    summary="Generate with ControlNet",
)
async def controlnet_pose_generation(
    request: ControlNetRequest, control=Depends(get_controlnet)
):
    """
    Generate an image conditioned on pose using the shared ControlNet engine.
    """
    try:
        start_time = time.time()

        # Load condition image
        if request.image_base64:
            image_data = base64.b64decode(request.image_base64)
            condition_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif request.image_path:
            condition_image = Image.open(request.image_path).convert("RGB")
        else:
            raise HTTPException(
                status_code=400, detail="Either image_path or image_base64 required"
            )

        # Apply controlnet (engine decides how to preprocess + run)
        result = control.apply(
            {"image": condition_image, "type": request.controlnet_type.value},
            strength=request.strength,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        return ControlNetResponse(
            image_path=result.get("image_path", ""),
            metadata_path=result.get("metadata_path", ""),
            condition_summary=f"{request.controlnet_type.value} applied with strength {request.strength}",
            seed=result.get("seed", 0),
            elapsed_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
