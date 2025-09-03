# api/routers/caption.py
from typing import List, Dict, Any, Optional
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

from PIL import Image
import base64, io, tempfile, os
from PIL import Image

from api.schemas import CaptionRequest, CaptionResponse
from core.vlm.engine import get_vlm_engine
from core.exceptions import VLMError, ImageProcessingError
from schemas.caption import CaptionRequest, CaptionResponse

from ..dependencies import get_vlm

router = APIRouter(tags=["caption"])

logger = logging.getLogger(__name__)


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


@router.post("/caption", response_model=CaptionResponse)
async def generate_caption(
    image: UploadFile = File(..., description="Image file to caption"),
    max_length: int = Form(50, description="Maximum caption length"),
    num_beams: int = Form(3, description="Number of beams for generation"),
    language: str = Form("en", description="Caption language (en/zh)"),
):
    """
    Generate image caption using BLIP-2

    - **image**: Image file (JPG, PNG, WebP)
    - **max_length**: Maximum caption length (10-200)
    - **num_beams**: Beam search size (1-5)
    - **language**: Output language (en/zh)
    """

    # Validate inputs
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type. Must be an image.")

    if not (10 <= max_length <= 200):
        raise HTTPException(400, "max_length must be between 10 and 200")

    if not (1 <= num_beams <= 5):
        raise HTTPException(400, "num_beams must be between 1 and 5")

    try:
        # Read image data
        image_data = await image.read()

        # Get VLM engine and generate caption
        vlm_engine = get_vlm_engine()
        result = vlm_engine.caption(
            image=image_data, max_length=max_length, num_beams=num_beams
        )

        # Translate to Chinese if requested
        caption_text = result["caption"]
        if language == "zh":
            # Simple translation - in production, use proper translation service
            # For now, prepend Chinese context
            caption_text = f"這張圖片顯示：{caption_text}"

        return CaptionResponse(
            caption=caption_text,
            confidence=result["confidence"],
            model_used=result["model_used"],
            language=language,
            metadata={
                "original_filename": image.filename,
                "file_size_kb": len(image_data) // 1024,
                "parameters": result["parameters"],
            },
        )

    except ImageProcessingError as e:
        logger.warning(f"Image processing error: {e}")
        raise HTTPException(400, f"Image processing failed: {e.message}")

    except VLMError as e:
        logger.error(f"Caption generation error: {e}")
        raise HTTPException(500, f"Caption generation failed: {e.message}")

    except Exception as e:
        logger.error(f"Unexpected error in caption endpoint: {e}", exc_info=True)
        raise HTTPException(500, "Internal server error occurred")


@router.post("/caption/batch", response_model=List[CaptionResponse])
async def batch_caption(
    images: List[UploadFile] = File(..., description="Multiple image files"),
    max_length: int = Form(50),
    num_beams: int = Form(3),
    language: str = Form("en"),
):
    """
    Generate captions for multiple images in batch

    - **images**: List of image files (max 10 files)
    - Other parameters same as single caption
    """

    # Validate batch size
    if len(images) > 10:
        raise HTTPException(400, "Maximum 10 images per batch")

    if not images:
        raise HTTPException(400, "At least one image required")

    results = []
    vlm_engine = get_vlm_engine()

    for i, image in enumerate(images):
        try:
            # Validate file type
            if not image.content_type or not image.content_type.startswith("image/"):
                results.append(
                    CaptionResponse(
                        caption=f"Error: Invalid file type for image {i+1}",
                        confidence=0.0,
                        model_used="",
                        language=language,
                        metadata={"error": "invalid_file_type", "index": i},
                    )
                )
                continue

            # Process image
            image_data = await image.read()
            result = vlm_engine.caption(
                image=image_data, max_length=max_length, num_beams=num_beams
            )

            # Format response
            caption_text = result["caption"]
            if language == "zh":
                caption_text = f"這張圖片顯示：{caption_text}"

            results.append(
                CaptionResponse(
                    caption=caption_text,
                    confidence=result["confidence"],
                    model_used=result["model_used"],
                    language=language,
                    metadata={
                        "index": i,
                        "original_filename": image.filename,
                        "file_size_kb": len(image_data) // 1024,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error processing image {i+1}: {e}")
            results.append(
                CaptionResponse(
                    caption=f"Error processing image {i+1}: {str(e)}",
                    confidence=0.0,
                    model_used="",
                    language=language,
                    metadata={"error": str(e), "index": i},
                )
            )

    return results


@router.get("/caption/models")
async def list_caption_models():
    """List available caption models"""
    return {
        "available_models": [
            {
                "name": "Salesforce/blip2-opt-2.7b",
                "description": "BLIP-2 with OPT-2.7B language model",
                "languages": ["en"],
                "recommended": True,
            },
            {
                "name": "Salesforce/blip2-flan-t5-xl",
                "description": "BLIP-2 with Flan-T5-XL language model",
                "languages": ["en", "zh"],
                "recommended": False,
            },
        ],
        "current_model": get_vlm_engine().config.model.caption_model,
    }
