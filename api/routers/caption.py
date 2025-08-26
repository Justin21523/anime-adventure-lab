# api/routers/caption.py
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends
from PIL import Image
import base64, io, tempfile, os
from PIL import Image

from api.schemas import CaptionRequest, CaptionResponse
from ..dependencies import get_vlm

router = APIRouter(tags=["caption"])


def _load_image_from_inputs(
    image: Optional[UploadFile], image_base64: Optional[str]
) -> Image.Image:
    """Load a PIL image from either an uploaded file or base64 string."""
    if image:
        data = image.file.read()  # UploadFile.read() is sync-friendly here
        return Image.open(io.BytesIO(data)).convert("RGB")
    if image_base64:
        data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    raise HTTPException(
        status_code=400, detail="Either `image` or `image_base64` is required"
    )


@router.post(
    "/caption", response_model=CaptionResponse, summary="Generate an image caption"
)
async def image_caption(
    image: UploadFile = File(None),
    image_base64: Optional[str] = None,
    max_length: int = 50,
    num_beams: int = 3,
    vlm=Depends(get_vlm),
):
    """
    Generate caption using the shared VLM engine.
    We save a temporary image file and pass its path to the engine for compatibility.
    """
    try:
        pil = _load_image_from_inputs(image, image_base64)

        # Save to a temporary file (engine expects a path); auto-clean after call
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Your real VLM may accept additional kwargs (e.g., max_length, num_beams)
            text = vlm.caption(tmp_path)  # keep signature simple for now
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        return CaptionResponse(caption=text, confidence=0.9, model_used="vlm")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
