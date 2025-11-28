# api/routers/caption.py
"""
Image Caption Router
BLIP-2 based image description generation
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends, Body
from fastapi.responses import JSONResponse

from api.dependencies import get_vlm
from core.exceptions import VLMError, ImageProcessingError
from schemas.caption import (
    CaptionRequest,
    CaptionResponse,
    CaptionParameters,
    BatchCaptionResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/caption", response_model=CaptionResponse)
async def generate_caption(
    image: UploadFile = File(..., description="Image file"),
    max_length: int = Form(50, description="Maximum caption length"),
    num_beams: int = Form(3, description="Number of beams for generation"),
    temperature: float = Form(0.7, description="Generation temperature"),
    language: str = Form("en", description="Caption language (en/zh)"),
    vlm_engine=Depends(get_vlm),
):
    """Generate caption for an image"""
    try:
        # Validate image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(400, "Invalid image file type")

        # Read image data
        image_data = await image.read()

        # Get VLM engine and process
        result = vlm_engine.caption(
            image=image_data,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
        )

        # Apply language formatting
        caption_text = result["caption"]
        if language == "zh":
            if not any(char for char in caption_text if "\u4e00" <= char <= "\u9fff"):
                caption_text = f"這張圖片顯示：{caption_text}"

        # Format response
        parameters = CaptionParameters(  # type: ignore
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            language=language,
        )

        return CaptionResponse(  # type: ignore
            caption=caption_text,
            confidence=result["confidence"],
            model_used=result["model_used"],
            language=language,
            parameters=parameters,
            image_info=result["image_info"],
        )

    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(500, f"Caption generation failed: {str(e)}")


# Alias 路由以符合 E2E 測試期望 (/vlm/caption)
@router.post("/vlm/caption", response_model=CaptionResponse)
async def generate_caption_vlm_alias(
    image: UploadFile = File(...),
    max_length: int = Form(50),
    num_beams: int = Form(3),
    temperature: float = Form(0.7),
    language: str = Form("en"),
    vlm_engine=Depends(get_vlm),
):
    return await generate_caption(
        image=image,
        max_length=max_length,
        num_beams=num_beams,
        temperature=temperature,
        language=language,
        vlm_engine=vlm_engine,
    )


@router.post("/vlm/check_consistency")
async def check_image_consistency(
    payload: Dict[str, Any] = Body(None),
    image_path: Optional[str] = Form(None),
    expected_tags: Optional[List[str]] = Form(None),
    world_id: Optional[str] = Form(None),
    vlm_engine=Depends(get_vlm),
):
    """
    簡易一致性檢查：使用 caption 生成文字後計算標籤匹配率。
    回傳 score (0~1) 以及命中的標籤。
    """
    try:
        # 若透過 JSON 傳遞，從 payload 讀取
        payload = payload or {}
        img_path = image_path or payload.get("image_path")
        tags = expected_tags or payload.get("expected_tags") or []
        world_id = world_id or payload.get("world_id")

        if not img_path:
            raise HTTPException(400, "image_path is required")

        caption_result = vlm_engine.caption(img_path, max_length=60, num_beams=3, temperature=0.7)
        caption_text = caption_result.get("caption", "")

        matched = [tag for tag in tags if str(tag).lower() in caption_text.lower()]
        score = len(matched) / max(len(tags), 1)

        return {
            "image_path": img_path,
            "expected_tags": tags,
            "matched_tags": matched,
            "consistency_score": round(score, 3),
            "caption": caption_text,
            "model_used": caption_result.get("model_used"),
            "world_id": world_id,
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Consistency check failed: %s", e)
        raise HTTPException(500, f"Consistency check failed: {e}") from e


@router.post("/caption/batch", response_model=BatchCaptionResponse)
async def batch_generate_captions(
    images: List[UploadFile] = File(..., description="Multiple image files"),
    max_length: int = Form(50),
    num_beams: int = Form(3),
    temperature: float = Form(0.7),
    language: str = Form("en"),
    vlm_engine=Depends(get_vlm),
):
    """Generate captions for multiple images in batch"""
    try:
        # Validate batch size
        if len(images) > 10:
            raise HTTPException(400, "Maximum 10 images per batch")

        if not images:
            raise HTTPException(400, "At least one image required")

        # Process each image
        results = []
        successful_count = 0
        failed_count = 0

        for i, image in enumerate(images):
            try:
                # Validate file type
                if not image.content_type or not image.content_type.startswith(
                    "image/"
                ):
                    results.append(
                        {
                            "batch_index": i,
                            "caption": f"Error: Invalid file type for image {i+1}",
                            "confidence": 0.0,
                            "error": "invalid_file_type",
                        }
                    )
                    failed_count += 1
                    continue

                # Process image
                image_data = await image.read()
                result = vlm_engine.caption(
                    image=image_data,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                )

                # Apply language formatting
                caption_text = result["caption"]
                if language == "zh":
                    if not any(
                        char for char in caption_text if "\u4e00" <= char <= "\u9fff"
                    ):
                        caption_text = f"這張圖片顯示：{caption_text}"

                # Format result
                parameters = CaptionParameters(  # type: ignore
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    language=language,
                )

                formatted_result = CaptionResponse(  # type: ignore
                    caption=caption_text,
                    confidence=result["confidence"],
                    model_used=result["model_used"],
                    language=language,
                    parameters=parameters,
                    image_info=result["image_info"],
                )

                # Add batch index
                result_dict = formatted_result.dict()
                result_dict["batch_index"] = i
                results.append(result_dict)
                successful_count += 1

            except Exception as e:
                logger.error(f"Caption generation failed for image {i}: {e}")
                results.append(
                    {
                        "batch_index": i,
                        "caption": f"Error processing image {i+1}: {str(e)}",
                        "confidence": 0.0,
                        "error": str(e),
                    }
                )
                failed_count += 1

        return BatchCaptionResponse(  # type: ignore
            results=results,
            total_items=len(results),
            successful_items=successful_count,
            failed_items=failed_count,
            success_rate=successful_count / len(results) if results else 0,
            parameters=CaptionParameters(  # type: ignore
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                language=language,
            ),
        )

    except Exception as e:
        logger.error(f"Batch caption generation failed: {e}")
        raise HTTPException(500, f"Batch caption generation failed: {str(e)}")


@router.get("/caption/models")
async def get_available_caption_models():
    """Get list of available caption models"""
    try:
        vlm_engine = get_vlm_engine()
        status = vlm_engine.get_status()

        return {
            "current_model": status.get("caption_model"),
            "model_loaded": status.get("caption_model_loaded", False),
            "available_models": [
                "Salesforce/blip2-opt-2.7b",
                "Salesforce/blip2-opt-6.7b",
                "Salesforce/blip2-flan-t5-xl",
                "Salesforce/blip-image-captioning-large",
            ],
            "recommended": "Salesforce/blip2-opt-2.7b",
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get caption models: {str(e)}")
