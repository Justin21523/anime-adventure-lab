# api/routers/safety.py
"""
Safety Filter Router
"""

import logging
from fastapi import APIRouter, HTTPException
from schemas.safety import SafetyCheckRequest, SafetyCheckResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/safety/check", response_model=SafetyCheckResponse)
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
