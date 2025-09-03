# api/routers/vqa.py
"""
Visual Question Answering router (DI-based).
Saves image to a temp file and calls vlm.vqa(path, question) if available.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from api.schemas import VQARequest, VQAResponse
import base64, io, tempfile, os
from PIL import Image

from ..dependencies import get_vlm
from core.vlm.engine import get_vlm_engine
from core.exceptions import VLMError, ValidationError
from schemas.vqa import VQARequest, VQAResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vqa"])


def _load_image_from_request(req: VQARequest) -> Image.Image:
    if req.image_base64:
        data = base64.b64decode(req.image_base64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    if req.image_path:
        return Image.open(req.image_path).convert("RGB")
    raise HTTPException(
        status_code=400, detail="Either image_path or image_base64 required"
    )


@router.post("/vqa", response_model=VQAResponse)
async def visual_question_answering(
    image: UploadFile = File(..., description="Image file for questioning"),
    question: str = Form(..., description="Question about the image"),
    max_length: int = Form(100, description="Maximum answer length"),
    language: str = Form("auto", description="Answer language (auto/en/zh)"),
):
    """
    Visual Question Answering using LLaVA/Qwen-VL

    - **image**: Image file (JPG, PNG, WebP)
    - **question**: Question about the image content
    - **max_length**: Maximum answer length (20-300)
    - **language**: Answer language (auto detect or specify)
    """

    # Validate inputs
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type. Must be an image.")

    if not question or len(question.strip()) < 3:
        raise HTTPException(400, "Question must be at least 3 characters long")

    if not (20 <= max_length <= 300):
        raise HTTPException(400, "max_length must be between 20 and 300")

    try:
        # Read image data
        image_data = await image.read()

        # Process question for Chinese context
        processed_question = question.strip()
        if language == "zh" or (language == "auto" and _contains_chinese(question)):
            # Add Chinese context to improve Chinese response quality
            if not processed_question.endswith(("？", "?", "。", ".")):
                processed_question += "？"

        # Get VLM engine and generate answer
        vlm_engine = get_vlm_engine()
        result = vlm_engine.vqa(
            image=image_data, question=processed_question, max_length=max_length
        )

        # Post-process answer based on language
        answer_text = result["answer"]
        detected_language = _detect_language(answer_text)

        if language == "zh" and detected_language == "en":
            # Simple translation hint for English answers
            answer_text = f"{answer_text}（英文回答）"

        return VQAResponse(
            question=question,
            answer=answer_text,
            confidence=result["confidence"],
            model_used=result["model_used"],
            language=detected_language,
            metadata={
                "original_filename": image.filename,
                "file_size_kb": len(image_data) // 1024,
                "question_length": len(question),
                "answer_length": len(answer_text),
                "parameters": result["parameters"],
            },
        )

    except VLMError as e:
        logger.error(f"VQA error: {e}")
        raise HTTPException(500, f"VQA failed: {e.message}")

    except ValidationError as e:
        logger.warning(f"VQA validation error: {e}")
        raise HTTPException(400, f"Validation failed: {e.message}")

    except Exception as e:
        logger.error(f"Unexpected error in VQA endpoint: {e}", exc_info=True)
        raise HTTPException(500, "Internal server error occurred")


@router.post("/vqa/conversation")
async def vqa_conversation(
    image: UploadFile = File(...),
    messages: str = Form(..., description="JSON array of conversation messages"),
    max_length: int = Form(150),
):
    """
    Multi-turn VQA conversation

    - **image**: Image file for the entire conversation
    - **messages**: JSON array of messages [{"role": "user", "content": "question"}]
    - **max_length**: Maximum response length
    """

    import json

    try:
        # Parse conversation messages
        try:
            message_list = json.loads(messages)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid JSON format for messages")

        if not isinstance(message_list, list) or len(message_list) == 0:
            raise HTTPException(400, "Messages must be a non-empty array")

        # Get the latest user question
        latest_question = None
        for msg in reversed(message_list):
            if msg.get("role") == "user":
                latest_question = msg.get("content", "")
                break

        if not latest_question:
            raise HTTPException(400, "No user question found in messages")

        # Build context from conversation history
        conversation_context = []
        for msg in message_list[:-1]:  # Exclude latest question
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                conversation_context.append(f"{role.title()}: {content}")

        # Enhanced question with context
        if conversation_context:
            context_str = "\n".join(conversation_context)
            enhanced_question = f"Previous conversation:\n{context_str}\n\nCurrent question: {latest_question}"
        else:
            enhanced_question = latest_question

        # Read image and process VQA
        image_data = await image.read()
        vlm_engine = get_vlm_engine()

        result = vlm_engine.vqa(
            image=image_data, question=enhanced_question, max_length=max_length
        )

        return {
            "question": latest_question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "model_used": result["model_used"],
            "conversation_turns": len(message_list),
            "metadata": {
                "has_context": len(conversation_context) > 0,
                "context_length": (
                    len("\n".join(conversation_context)) if conversation_context else 0
                ),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"VQA conversation error: {e}", exc_info=True)
        raise HTTPException(500, f"VQA conversation failed: {str(e)}")


def _contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters"""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


def _detect_language(text: str) -> str:
    """Simple language detection"""
    chinese_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    total_chars = len([c for c in text if c.isalnum()])

    if total_chars == 0:
        return "unknown"

    chinese_ratio = chinese_count / total_chars
    return "zh" if chinese_ratio > 0.3 else "en"
