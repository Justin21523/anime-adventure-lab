# api/routers/vqa.py
"""
Visual Question Answering Router
LLaVA/Qwen-VL based image Q&A
"""

import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import List, Dict, Any, Optional

from core.vlm.engine import get_vlm_engine
from core.exceptions import VLMError, ValidationError
from schemas.vqa import VQARequest, VQAResponse, VQAParameters, BatchVQAResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/vqa", response_model=VQAResponse)
async def visual_question_answering(
    image: UploadFile = File(..., description="Image file"),
    question: str = Form(..., description="Question about the image"),
    max_length: int = Form(100, description="Maximum answer length"),
    temperature: float = Form(0.7, description="Generation temperature"),
    language: str = Form("auto", description="Response language (auto/en/zh)"),
):
    """Answer a question about an image"""
    try:
        # Validate image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(400, "Invalid image file type")

        # Validate question
        if not question.strip():
            raise HTTPException(400, "Question cannot be empty")

        # Read image data
        image_data = await image.read()

        # Get VLM engine and process
        vlm_engine = get_vlm_engine()
        result = vlm_engine.vqa(
            image=image_data,
            question=question,
            max_length=max_length,
            temperature=temperature,
        )

        # Format response
        parameters = VQAParameters(  # type: ignore
            max_length=max_length, temperature=temperature, language=language
        )

        return VQAResponse(  # type: ignore
            question=result["question"],
            answer=result["answer"],
            confidence=result["confidence"],
            model_used=result["model_used"],
            language_detected=result["language_detected"],
            parameters=parameters,
            image_info=result["image_info"],
        )

    except Exception as e:
        logger.error(f"VQA failed: {e}")
        raise HTTPException(500, f"Visual question answering failed: {str(e)}")


@router.post("/vqa/batch", response_model=BatchVQAResponse)
async def batch_visual_question_answering(
    files: List[UploadFile] = File(..., description="Image files"),
    questions: List[str] = Form(..., description="Questions for each image"),
    max_length: int = Form(100),
    temperature: float = Form(0.7),
    language: str = Form("auto"),
):
    """Answer questions about multiple images in batch"""
    try:
        # Validate inputs
        if len(files) != len(questions):
            raise HTTPException(400, "Number of images must match number of questions")

        if len(files) > 10:
            raise HTTPException(400, "Maximum 10 image-question pairs per batch")

        if not files:
            raise HTTPException(400, "At least one image-question pair required")

        # Process each pair
        results = []
        vlm_engine = get_vlm_engine()
        successful_count = 0
        failed_count = 0

        for i, (image_file, question) in enumerate(zip(files, questions)):
            try:
                # Validate file type
                if (
                    not image_file.content_type
                    or not image_file.content_type.startswith("image/")
                ):
                    results.append(
                        {
                            "batch_index": i,
                            "question": question,
                            "answer": f"Error: Invalid file type for image {i+1}",
                            "confidence": 0.0,
                            "error": "invalid_file_type",
                        }
                    )
                    failed_count += 1
                    continue

                # Validate question
                if not question.strip():
                    results.append(
                        {
                            "batch_index": i,
                            "question": question,
                            "answer": f"Error: Empty question for image {i+1}",
                            "confidence": 0.0,
                            "error": "empty_question",
                        }
                    )
                    failed_count += 1
                    continue

                # Process image
                image_data = await image_file.read()
                result = vlm_engine.vqa(
                    image=image_data,
                    question=question,
                    max_length=max_length,
                    temperature=temperature,
                )

                # Format result
                parameters = VQAParameters(  # type: ignore
                    max_length=max_length, temperature=temperature, language=language
                )

                formatted_result = VQAResponse(  # type: ignore
                    question=result["question"],
                    answer=result["answer"],
                    confidence=result["confidence"],
                    model_used=result["model_used"],
                    language_detected=result["language_detected"],
                    parameters=parameters,
                    image_info=result["image_info"],
                )

                # Add batch index
                result_dict = formatted_result.dict()
                result_dict["batch_index"] = i
                results.append(result_dict)
                successful_count += 1

            except Exception as e:
                logger.error(f"VQA failed for image {i}: {e}")
                results.append(
                    {
                        "batch_index": i,
                        "question": question,
                        "answer": f"Error processing image {i+1}: {str(e)}",
                        "confidence": 0.0,
                        "error": str(e),
                    }
                )
                failed_count += 1

        return BatchVQAResponse(  # type: ignore
            results=results,
            total_items=len(results),
            successful_items=successful_count,
            failed_items=failed_count,
            success_rate=successful_count / len(results) if results else 0,
            parameters=VQAParameters(  # type: ignore
                max_length=max_length, temperature=temperature, language=language
            ),
        )

    except Exception as e:
        logger.error(f"Batch VQA failed: {e}")
        raise HTTPException(500, f"Batch visual question answering failed: {str(e)}")


@router.get("/vqa/models")
async def get_available_vqa_models():
    """Get list of available VQA models"""
    try:
        vlm_engine = get_vlm_engine()
        status = vlm_engine.get_status()

        return {
            "current_model": status.get("vqa_model"),
            "model_loaded": status.get("vqa_model_loaded", False),
            "available_models": [
                "llava-hf/llava-1.5-7b-hf",
                "llava-hf/llava-1.5-13b-hf",
                "Qwen/Qwen-VL-Chat",
                "Salesforce/blip2-opt-2.7b",
            ],
            "recommended": "llava-hf/llava-1.5-7b-hf",
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get VQA models: {str(e)}")


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
