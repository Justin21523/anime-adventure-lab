# api/routers/safety.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import tempfile
import shutil
from PIL import Image

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from core.safety.detector import SafetyEngine
from core.safety.license import LicenseManager, LicenseInfo
from core.safety.watermark import AttributionManager, ComplianceLogger
from core.shared_cache import get_cache_root

router = APIRouter(prefix="/safety", tags=["safety"])


# Pydantic models
class LicenseInfoRequest(BaseModel):
    license_type: str = Field(..., description="License type (CC0, CC-BY, MIT, etc.)")
    attribution_required: bool = Field(
        True, description="Whether attribution is required"
    )
    commercial_use: bool = Field(True, description="Whether commercial use is allowed")
    derivative_works: bool = Field(
        True, description="Whether derivative works are allowed"
    )
    share_alike: bool = Field(False, description="Whether share-alike is required")
    source_url: Optional[str] = Field(None, description="Source URL if applicable")
    author: Optional[str] = Field(None, description="Author/creator name")
    license_text: Optional[str] = Field(None, description="Custom license text")
    restrictions: List[str] = Field(
        default_factory=list, description="Additional restrictions"
    )


class SafetyCheckRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to check")
    check_nsfw: bool = Field(True, description="Enable NSFW detection")
    check_injection: bool = Field(True, description="Enable prompt injection detection")


class WatermarkRequest(BaseModel):
    add_visible: bool = Field(True, description="Add visible watermark")
    position: str = Field("bottom_right", description="Watermark position")
    opacity: float = Field(0.6, description="Watermark opacity (0.0-1.0)")


class SafetyCheckResponse(BaseModel):
    is_safe: bool
    prompt_check: Dict[str, Any]
    image_check: Optional[Dict[str, Any]] = None
    actions_taken: List[str]
    warnings: List[str]


class UploadResponse(BaseModel):
    file_id: str
    safety_check: Dict[str, Any]
    license_check: Dict[str, Any]
    metadata_path: str


# Dependency injection
def get_safety_engine() -> SafetyEngine:
    """Get SafetyEngine instance"""
    return SafetyEngine()


def get_license_manager() -> LicenseManager:
    """Get LicenseManager instance"""
    cache_root = get_cache_root()
    return LicenseManager(cache_root)


def get_attribution_manager() -> AttributionManager:
    """Get AttributionManager instance"""
    cache_root = get_cache_root()
    return AttributionManager(cache_root)


def get_compliance_logger() -> ComplianceLogger:
    """Get ComplianceLogger instance"""
    cache_root = get_cache_root()
    return ComplianceLogger(cache_root)


@router.post("/check/prompt", response_model=SafetyCheckResponse)
async def check_prompt_safety(
    request: SafetyCheckRequest,
    safety_engine: SafetyEngine = Depends(get_safety_engine),
):
    """Check prompt safety for injection and harmful content"""

    try:
        # Check prompt safety
        prompt_result = safety_engine.check_prompt_safety(request.prompt)

        is_safe = prompt_result.get("is_safe", True)
        actions_taken = []
        warnings = prompt_result.get("warnings", [])

        if not is_safe:
            actions_taken.append("prompt_blocked")

        return SafetyCheckResponse(
            is_safe=is_safe,
            prompt_check=prompt_result,
            actions_taken=actions_taken,
            warnings=warnings,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Safety check failed: {str(e)}")


@router.post("/upload", response_model=UploadResponse)
async def upload_with_safety_check(
    file: UploadFile = File(...),
    license_info: str = Form(..., description="JSON string of license information"),
    uploader_id: str = Form(..., description="Uploader identifier"),
    auto_blur_faces: bool = Form(
        False, description="Automatically blur detected faces"
    ),
    license_manager: LicenseManager = Depends(get_license_manager),
    safety_engine: SafetyEngine = Depends(get_safety_engine),
    compliance_logger: ComplianceLogger = Depends(get_compliance_logger),
):
    """Upload file with comprehensive safety and license checks"""

    import json

    try:
        # Parse license information
        license_data = json.loads(license_info)
        license_obj = LicenseInfo(**license_data)

        # Create temporary file
        cache_root = get_cache_root()
        upload_dir = Path(cache_root) / "uploads" / "temp"
        upload_dir.mkdir(parents=True, exist_ok=True)

        temp_file_path = upload_dir / f"temp_{file.filename}"

        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform safety checks for images
        safety_result = {"is_safe": True}
        if file.content_type and file.content_type.startswith("image/"):
            try:
                image = Image.open(temp_file_path)
                safety_result = safety_engine.check_image_safety(image)

                # Handle face blurring if requested and faces detected
                if auto_blur_faces and "processed_image" in safety_result:
                    processed_image = safety_result["processed_image"]
                    processed_image.save(temp_file_path)

            except Exception as e:
                safety_result = {
                    "is_safe": False,
                    "error": f"Image processing failed: {str(e)}",
                }

        # Register upload with license manager
        upload_metadata = license_manager.register_upload(
            str(temp_file_path), license_obj, uploader_id, safety_result
        )

        # Move file to permanent location if safe
        if safety_result.get("is_safe", False):
            permanent_dir = Path(cache_root) / "datasets" / "raw"
            permanent_dir.mkdir(parents=True, exist_ok=True)
            permanent_path = (
                permanent_dir / f"{upload_metadata.file_id}_{file.filename}"
            )
            shutil.move(str(temp_file_path), str(permanent_path))
        else:
            # Remove unsafe file
            temp_file_path.unlink(missing_ok=True)

            # Log safety violation
            compliance_logger.log_safety_violation(
                "unsafe_upload",
                {"filename": file.filename, "content_type": file.content_type},
                "file_rejected",
            )

            raise HTTPException(
                status_code=400,
                detail=f"Upload rejected due to safety concerns: {safety_result}",
            )

        # Validate license
        validation_result = license_manager.validator.validate_license(license_obj)

        # Log upload
        compliance_logger.log_upload(
            upload_metadata.file_id,
            {"license_info": license_data, "uploader_id": uploader_id},
            safety_result,
        )

        return UploadResponse(
            file_id=upload_metadata.file_id,
            safety_check=safety_result,
            license_check=validation_result,
            metadata_path=str(permanent_path.with_suffix(".json")),
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid license information JSON")
    except Exception as e:
        # Clean up temp file if it exists
        if "temp_file_path" in locals():
            Path(temp_file_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/license/{file_id}")
async def get_license_info(
    file_id: str, license_manager: LicenseManager = Depends(get_license_manager)
):
    """Get license information for a file"""

    metadata = license_manager.get_metadata(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "file_id": file_id,
        "license_info": metadata.license_info,
        "attribution_text": license_manager.generate_attribution_text(file_id),
    }


@router.post("/license/check-compliance")
async def check_usage_compliance(
    file_id: str,
    intended_use: str,
    license_manager: LicenseManager = Depends(get_license_manager),
):
    """Check if intended use complies with file's license"""

    compliance_result = license_manager.check_usage_compliance(file_id, intended_use)

    return {
        "file_id": file_id,
        "intended_use": intended_use,
        "compliance_result": compliance_result,
    }


@router.get("/audit/summary")
async def get_audit_summary(
    days: int = 7, compliance_logger: ComplianceLogger = Depends(get_compliance_logger)
):
    """Get compliance audit summary for the last N days"""

    summary = compliance_logger.get_audit_summary(days)
    return summary


@router.get("/licenses/list")
async def list_license_types():
    """List supported license types and their properties"""

    from core.safety.license import LicenseValidator

    validator = LicenseValidator()

    return {
        "supported_licenses": validator.KNOWN_LICENSES,
        "description": "Supported license types with their properties and requirements",
    }


@router.get("/files/by-license/{license_type}")
async def list_files_by_license(
    license_type: str, license_manager: LicenseManager = Depends(get_license_manager)
):
    """List all files with specific license type"""

    files = license_manager.list_files_by_license(license_type)

    return {
        "license_type": license_type,
        "file_count": len(files),
        "files": [
            {
                "file_id": f.file_id,
                "filename": f.original_filename,
                "upload_timestamp": f.upload_timestamp,
                "attribution_text": license_manager.generate_attribution_text(
                    f.file_id
                ),
            }
            for f in files
        ],
    }


@router.post("/watermark/add")
async def add_watermark_to_image(
    file: UploadFile = File(...),
    watermark_config: str = Form(..., description="JSON watermark configuration"),
    attribution_manager: AttributionManager = Depends(get_attribution_manager),
):
    """Add watermark to an uploaded image"""

    import json

    try:
        # Parse watermark configuration
        config = json.loads(watermark_config)

        # Load image
        image = Image.open(file.file)

        # Create dummy generation params for watermark
        generation_params = {
            "prompt": config.get("prompt", "Manual watermark addition"),
            "model_id": "manual",
            "timestamp": "manual_processing",
        }

        # Process image with watermark
        processed_image, metadata = attribution_manager.process_generated_image(
            image,
            generation_params,
            add_visible_watermark=config.get("add_visible", True),
            watermark_position=config.get("position", "bottom_right"),
        )

        # Save to temporary location
        cache_root = get_cache_root()
        output_dir = Path(cache_root) / "outputs" / "watermarked"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"watermarked_{file.filename}"
        saved_path = attribution_manager.save_with_attribution(
            processed_image, str(output_path), metadata
        )

        return {
            "output_path": saved_path,
            "metadata": metadata,
            "watermark_added": True,
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid watermark configuration JSON"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Watermark addition failed: {str(e)}"
        )


@router.get("/health")
async def safety_health_check():
    """Health check for safety systems"""

    try:
        # Test safety engine initialization
        safety_engine = get_safety_engine()

        # Test license manager
        license_manager = get_license_manager()

        # Test attribution manager
        attribution_manager = get_attribution_manager()

        return {
            "status": "healthy",
            "components": {
                "safety_engine": "operational",
                "license_manager": "operational",
                "attribution_manager": "operational",
                "compliance_logger": "operational",
            },
            "models_loaded": {
                "nsfw_detector": hasattr(safety_engine.nsfw_detector, "model"),
                "face_detector": hasattr(safety_engine.face_blurrer, "face_cascade"),
            },
        }

    except Exception as e:
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )
