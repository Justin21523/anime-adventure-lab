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


@router.post("/check", response_model=SafetyCheckResponse)
async def check_content_safety(request: SafetyCheckRequest):
    """Check content for safety violations"""
    try:
        # Mock safety check
        blocked_terms = ["violent", "nsfw", "hate"]
        detected_issues = []

        content_lower = request.content.lower()
        for term in blocked_terms:
            if term in content_lower:
                detected_issues.append(f"Contains blocked term: {term}")

        safe = len(detected_issues) == 0
        risk_level = "low" if safe else "high" if len(detected_issues) > 1 else "medium"

        recommendations = []
        if not safe:
            recommendations.append("Remove flagged content")
            recommendations.append("Review content guidelines")

        return SafetyCheckResponse(  # type: ignore
            safe=safe,
            risk_level=risk_level,
            detected_issues=detected_issues,
            confidence=0.95,
            recommendations=recommendations,
        )

    except Exception as e:
        raise HTTPException(500, f"Safety check failed: {str(e)}")


@router.post("/text/check", response_model=TextSafetyResponse)
async def check_text_safety(request: TextSafetyRequest):
    """Check text content for safety violations"""
    try:
        content_filter = get_content_filter()
        result = content_filter.check_text_safety(request.text)

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
        content_filter = get_content_filter()

        # Load image from base64
        import base64
        from io import BytesIO
        from PIL import Image

        image_data = base64.b64decode(request.image_data)
        image = Image.open(BytesIO(image_data))

        result = content_filter.check_image_safety(image)

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
