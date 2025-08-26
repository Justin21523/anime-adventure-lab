# api/routers/safety.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os
import json
from pathlib import Path
import tempfile
import shutil
from PIL import Image


from ..dependencies import (
    get_safety_engine,
    get_license_manager,
    get_attribution_manager,
    get_compliance_logger,
    get_cache,  # if you need raw paths
)

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


@router.post("/check/prompt", response_model=SafetyCheckResponse)
async def check_prompt_safety(
    request: SafetyCheckRequest, safety=Depends(get_safety_engine)
):
    """Check prompt safety for injection and harmful content."""
    try:
        prompt_result = safety.check_prompt_safety(request.prompt)
        is_safe = bool(prompt_result.get("is_safe", True))
        warnings = prompt_result.get("warnings", [])
        actions = ["prompt_blocked"] if not is_safe else []
        return SafetyCheckResponse(
            is_safe=is_safe,
            prompt_check=prompt_result,
            actions_taken=actions,
            warnings=warnings,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Safety check failed: {e}")


@router.post("/upload", response_model=UploadResponse)
async def upload_with_safety_check(
    file: UploadFile = File(...),
    license_info: str = Form(..., description="JSON string of license information"),
    uploader_id: str = Form(...),
    auto_blur_faces: bool = Form(False),
    license_manager=Depends(get_license_manager),
    safety=Depends(get_safety_engine),
    compliance=Depends(get_compliance_logger),
    cache=Depends(get_cache),
):
    """Upload file with safety and license checks."""
    try:
        license_data = json.loads(license_info)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid license information JSON")

    try:
        cache_root = cache.cache_root
        temp_dir = Path(cache_root) / "uploads" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"temp_{file.filename}"

        with open(temp_file, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # Image safety checks
        safety_result = {"is_safe": True}
        if file.content_type and file.content_type.startswith("image/"):
            try:
                image = Image.open(temp_file)
                safety_result = safety.check_image_safety(image)
                if auto_blur_faces and "processed_image" in safety_result:
                    processed = safety_result["processed_image"]
                    processed.save(temp_file)
            except Exception as e:
                safety_result = {
                    "is_safe": False,
                    "error": f"Image processing failed: {e}",
                }

        # Register upload & move if safe
        upload_meta = license_manager.register_upload(
            str(temp_file), license_data, uploader_id, safety_result
        )
        if safety_result.get("is_safe", False):
            final_dir = Path(cache_root) / "datasets" / "raw"
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / f"{upload_meta.file_id}_{file.filename}"
            shutil.move(str(temp_file), str(final_path))
        else:
            temp_file.unlink(missing_ok=True)
            compliance.log_safety_violation(
                "unsafe_upload",
                {"filename": file.filename, "content_type": file.content_type},
                "file_rejected",
            )
            raise HTTPException(
                status_code=400, detail=f"Upload rejected: {safety_result}"
            )

        validation = license_manager.validator.validate_license(license_data)
        compliance.log_upload(
            upload_meta.file_id,
            {"uploader_id": uploader_id, "license_info": license_data},
            safety_result,
        )

        return UploadResponse(
            file_id=upload_meta.file_id,
            safety_check=safety_result,
            license_check=validation,
            metadata_path=str(final_path.with_suffix(".json")),
        )
    except HTTPException:
        raise
    except Exception as e:
        # best-effort temp cleanup
        try:
            temp_file.unlink(missing_ok=True)  # type: ignore
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


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
async def safety_health(
    safety=Depends(get_safety_engine),
    license_manager=Depends(get_license_manager),
    attribution=Depends(get_attribution_manager),
    compliance=Depends(get_compliance_logger),
):
    """Check status of safety components."""
    try:
        return {
            "status": "healthy",
            "components": {
                "safety_engine": "operational",
                "license_manager": "operational",
                "attribution_manager": "operational",
                "compliance_logger": "operational",
            },
        }
    except Exception as e:
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )
