# api/routers/safety.py
"""
Safety Filter Router
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

from core.safety import get_content_filter, get_rate_limiter, get_input_validator
from core.exceptions import SafetyError, ValidationError, RateLimitError
from schemas.safety import SafetyCheckRequest, SafetyCheckResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/safety", tags=["safety"])


# Request/Response models
class TextSafetyRequest(BaseModel):
    text: str
    check_toxicity: bool = True
    check_blocked_terms: bool = True
    check_sensitive_data: bool = True


class TextSafetyResponse(BaseModel):
    is_safe: bool
    violations: List[Dict[str, Any]]
    confidence: float
    filtered_text: str


class ImageSafetyRequest(BaseModel):
    image_data: str  # Base64 encoded image
    check_nsfw: bool = True
    check_faces: bool = False


class ImageSafetyResponse(BaseModel):
    is_safe: bool
    violations: List[Dict[str, Any]]
    confidence: float
    nsfw_score: float


class RateLimitStatusResponse(BaseModel):
    allowed: bool
    remaining: Dict[str, int]
    reset_times: Dict[str, float]
    client_id: str


class ValidationRequest(BaseModel):
    data: Dict[str, Any]
    validation_type: str  # "generation", "chat", "lora", "batch"


class ValidationResponse(BaseModel):
    is_valid: bool
    validated_data: Dict[str, Any]
    errors: List[str]


class TextBatchRequest(BaseModel):
    texts: List[str]
    check_toxicity: bool = True
    check_blocked_terms: bool = True
    check_sensitive_data: bool = True


class TextBatchResponse(BaseModel):
    total: int
    unsafe: int
    results: List[TextSafetyResponse]


@router.post("/check", response_model=SafetyCheckResponse)
async def check_content_safety(request: SafetyCheckRequest):
    """Check content for safety violations"""
    try:
        cf = get_content_filter()
        text_result = cf.check_text_safety(request.content)
        safe = text_result["is_safe"]
        risk_level = "low" if safe else "high" if len(text_result["violations"]) > 1 else "medium"

        # Optional image check if provided
        image_issues = []
        if request.image_base64:
            image_result = cf.check_image_safety(cf.image_processor.load_image_from_base64(request.image_base64))  # type: ignore
            if not image_result["is_safe"]:
                image_issues.append("NSFW/unsafe image detected")
                safe = False
                risk_level = "high"

        detected = text_result["violations"] + image_issues
        recommendations = []
        if not safe:
            recommendations.append("Remove or rewrite flagged content")
            recommendations.append("Follow safety and content guidelines")

        return SafetyCheckResponse(  # type: ignore
            safe=safe,
            risk_level=risk_level,
            detected_issues=detected,
            confidence=text_result.get("confidence", 0.8),
            recommendations=recommendations,
        )

    except Exception as e:
        raise HTTPException(500, f"Safety check failed: {str(e)}")


@router.post("/text/check", response_model=TextSafetyResponse)
async def check_text_safety(request: TextSafetyRequest):
    """Check text content for safety violations"""
    try:
        cf = get_content_filter()
        result = cf.check_text_safety(request.text)

        return TextSafetyResponse(
            is_safe=result["is_safe"],
            violations=result["violations"],
            confidence=result["confidence"],
            filtered_text=result["filtered_text"],
        )

    except Exception as e:
        logger.error(f"Text safety check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Safety check failed: {e}")


@router.post("/image/check", response_model=ImageSafetyResponse)
async def check_image_safety(request: ImageSafetyRequest):
    """Check image content for safety violations"""
    try:
        # Load image from base64
        import base64
        from io import BytesIO
        from PIL import Image

        image_data = base64.b64decode(request.image_data)
        image = Image.open(BytesIO(image_data))

        cf = get_content_filter()
        result = cf.check_image_safety(image)

        return ImageSafetyResponse(
            is_safe=result["is_safe"],
            violations=result["violations"],
            confidence=result["confidence"],
            nsfw_score=result["nsfw_score"],
        )

    except Exception as e:
        logger.error(f"Image safety check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Safety check failed: {e}")


@router.get("/rate-limit/status", response_model=RateLimitStatusResponse)
async def get_rate_limit_status(request: Request):
    """Get current rate limit status for client"""
    try:
        rate_limiter = get_rate_limiter()
        client_id = rate_limiter.get_client_id(request)

        # Check rate limits for common endpoints
        endpoints = ["/api/v1/txt2img", "/api/v1/chat", "/api/v1/caption"]
        status_info = {"allowed": True, "remaining": {}, "reset_times": {}}

        for endpoint in endpoints:
            rate_limit = rate_limiter.get_rate_limit_for_endpoint(endpoint)
            allowed, info = await rate_limiter.check_rate_limit(request, endpoint)

            if "remaining" in info:
                status_info["remaining"][endpoint] = info["remaining"]
            if "reset_time" in info:
                status_info["reset_times"][endpoint] = info["reset_time"]

            if not allowed:
                status_info["allowed"] = False

        return RateLimitStatusResponse(
            allowed=status_info["allowed"],
            remaining=status_info["remaining"],
            reset_times=status_info["reset_times"],
            client_id=client_id,
        )

    except Exception as e:
        logger.error(f"Rate limit status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rate limit check failed: {e}")


@router.post("/validate", response_model=ValidationResponse)
async def validate_input(request: ValidationRequest):
    """Validate input data according to specified type"""
    try:
        validator = get_input_validator()

        errors = []
        validated_data = {}

        if request.validation_type == "generation":
            validated_data = validator.validate_generation_params(request.data)
        elif request.validation_type == "chat":
            validated_data = validator.validate_chat_params(request.data)
        elif request.validation_type == "lora":
            validated_data = validator.validate_lora_params(request.data)
        elif request.validation_type == "batch":
            if "batch_data" in request.data:
                validated_data["batch_data"] = validator.validate_batch_request(
                    request.data["batch_data"]
                )
        else:
            raise ValidationError(f"Unknown validation type: {request.validation_type}")

        return ValidationResponse(
            is_valid=True, validated_data=validated_data, errors=errors
        )

    except (ValidationError, SafetyError) as e:
        return ValidationResponse(is_valid=False, validated_data={}, errors=[str(e)])
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")


@router.post("/text/batch", response_model=TextBatchResponse)
async def check_text_batch(request: TextBatchRequest):
    """Batch text safety checking (for agents/LLM outputs)."""
    try:
        results: List[TextSafetyResponse] = []
        unsafe = 0
        cf = get_content_filter()

        for text in request.texts:
            res = cf.check_text_safety(text)
            result_obj = TextSafetyResponse(
                is_safe=res["is_safe"],
                violations=res["violations"],
                confidence=res["confidence"],
                filtered_text=res["filtered_text"],
            )
            results.append(result_obj)
            if not res["is_safe"]:
                unsafe += 1

        return TextBatchResponse(total=len(request.texts), unsafe=unsafe, results=results)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Batch text safety failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch safety failed: {e}")


@router.get("/status")
async def safety_status():
    """Return safety system readiness (models, blocked terms count)."""
    try:
        cf = get_content_filter()
        return {
            "blocked_terms": len(cf.blocked_terms),
            "text_classifier_loaded": cf.text_classifier is not None,
            "nsfw_classifier_loaded": cf.nsfw_classifier is not None,
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"Safety status failed: {e}")
        raise HTTPException(status_code=500, detail=f"Safety status failed: {e}")
