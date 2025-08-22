# api/routers/vlm.py
"""
VLM API Router - FastAPI endpoints for vision-language model operations
Handles image captioning, analysis, and consistency checking
"""

import asyncio
import json
import uuid
import torch
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import io
import logging

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from core.vlm.captioner import VLMCaptioner
from core.vlm.consistency import VLMConsistencyChecker, ConsistencyReport
from core.vlm.tagger import WD14Tagger
from core.rag.engine import ChineseRAGEngine
from core.config import get_config
from api.dependencies import get_rag_engine, get_vlm_captioner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vlm", tags=["vlm"])


# Request/Response Models
class CaptionRequest(BaseModel):
    model_type: Optional[str] = Field(default="blip2", description="VLM model to use")
    prompt: Optional[str] = Field(
        default=None, description="Custom prompt for captioning"
    )
    language: Optional[str] = Field(default="en", description="Response language")


class CaptionResponse(BaseModel):
    caption: str
    model_type: str
    processing_time: float
    image_info: Dict[str, Any]
    timestamp: datetime


class AnalysisRequest(BaseModel):
    model_type: Optional[str] = Field(default="blip2", description="VLM model to use")
    include_tags: bool = Field(default=True, description="Include wd14 tags")
    check_consistency: bool = Field(
        default=False, description="Check against world data"
    )
    world_id: Optional[str] = Field(
        default=None, description="World ID for consistency check"
    )
    character_id: Optional[str] = Field(
        default=None, description="Character ID for validation"
    )
    scene_id: Optional[str] = Field(default=None, description="Scene ID for validation")


class AnalysisResponse(BaseModel):
    analysis: Dict[str, Any]
    tags: Optional[List[str]] = None
    consistency_report: Optional[Dict[str, Any]] = None
    model_type: str
    processing_time: float
    timestamp: datetime


class BatchCaptionRequest(BaseModel):
    model_type: Optional[str] = Field(default="blip2", description="VLM model to use")
    prompt: Optional[str] = Field(
        default=None, description="Custom prompt for all images"
    )
    output_format: str = Field(
        default="json", description="Output format: json, csv, txt"
    )


class WritebackRequest(BaseModel):
    world_id: str
    scope: str = Field(
        default="scene_notes",
        description="Memory scope: scene_notes, character_memory, world_lore",
    )
    caption: str
    image_context: Dict[str, Any] = Field(default_factory=dict)
    auto_tag: bool = Field(default=True, description="Automatically generate tags")


# Helper Functions
async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """Load PIL Image from uploaded file"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


def get_vlm_config() -> Dict[str, Any]:
    """Get VLM configuration"""
    config = get_config()
    return config.get(
        "vlm",
        {
            "default_model": "blip2",
            "device": "auto",
            "low_vram": True,
            "models": {
                "blip2": "Salesforce/blip2-opt-2.7b",
                "llava": "liuhaotian/llava-v1.6-mistral-7b",
            },
        },
    )


# API Endpoints
@router.post("/caption", response_model=CaptionResponse)
async def caption_image(
    file: UploadFile = File(...),
    request: CaptionRequest = Depends(),
    captioner: VLMCaptioner = Depends(get_vlm_captioner),
):
    """Generate caption for uploaded image"""
    start_time = datetime.now()

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Load image
        image = await load_image_from_upload(file)

        # Generate caption
        caption = captioner.caption(
            image=image, model_type=request.model_type, prompt=request.prompt
        )

        # Translate if needed
        if request.language == "zh" and request.model_type == "blip2":
            # Simple translation for Chinese (could integrate proper translation service)
            caption = f"[ZH] {caption}"  # Placeholder for translation

        processing_time = (datetime.now() - start_time).total_seconds()

        return CaptionResponse(
            caption=caption,
            model_type=request.model_type,
            processing_time=processing_time,
            image_info={
                "size": image.size,
                "mode": image.mode,
                "filename": file.filename,
            },
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Caption generation failed: {str(e)}"
        )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    request: AnalysisRequest = Depends(),
    captioner: VLMCaptioner = Depends(get_vlm_captioner),
    rag_engine: ChineseRAGEngine = Depends(get_rag_engine),
):
    """Perform comprehensive image analysis"""
    start_time = datetime.now()

    try:
        # Load image
        image = await load_image_from_upload(file)

        # Perform VLM analysis
        analysis = captioner.analyze(image=image, model_type=request.model_type)

        tags = None
        consistency_report = None

        # Generate tags if requested
        if request.include_tags:
            try:
                tagger = WD14Tagger()
                tags = tagger.tag_image(image)
            except Exception as e:
                logger.warning(f"Tagging failed: {e}")
                tags = []

        # Check consistency if requested
        if request.check_consistency and request.world_id:
            try:
                # Get world data from RAG
                world_data = await get_world_data(rag_engine, request.world_id)

                # Create consistency checker
                config = get_vlm_config()
                checker = VLMConsistencyChecker(world_data, config)

                # Check consistency
                image_context = {
                    "character_id": request.character_id,
                    "scene_id": request.scene_id,
                }

                report = checker.check_consistency(
                    caption=analysis.get("general", ""), image_context=image_context
                )

                consistency_report = {
                    "overall_score": report.overall_score,
                    "issues": [
                        {
                            "category": issue.category,
                            "severity": issue.severity,
                            "description": issue.description,
                            "expected": issue.expected,
                            "found": issue.found,
                            "confidence": issue.confidence,
                        }
                        for issue in report.issues
                    ],
                    "suggestions": report.suggestions,
                    "validated_elements": report.validated_elements,
                }

            except Exception as e:
                logger.warning(f"Consistency check failed: {e}")
                consistency_report = {"error": str(e)}

        processing_time = (datetime.now() - start_time).total_seconds()

        return AnalysisResponse(
            analysis=analysis,
            tags=tags,
            consistency_report=consistency_report,
            model_type=request.model_type,
            processing_time=processing_time,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch_caption")
async def batch_caption_images(
    files: List[UploadFile] = File(...),
    request: BatchCaptionRequest = Depends(),
    captioner: VLMCaptioner = Depends(get_vlm_captioner),
):
    """Generate captions for multiple images"""
    start_time = datetime.now()

    try:
        if len(files) > 20:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many files (max 20)")

        results = []

        for i, file in enumerate(files):
            try:
                # Load image
                image = await load_image_from_upload(file)

                # Generate caption
                caption = captioner.caption(
                    image=image, model_type=request.model_type, prompt=request.prompt
                )

                results.append(
                    {
                        "index": i,
                        "filename": file.filename,
                        "caption": caption,
                        "success": True,
                        "error": None,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "index": i,
                        "filename": file.filename,
                        "caption": None,
                        "success": False,
                        "error": str(e),
                    }
                )

        processing_time = (datetime.now() - start_time).total_seconds()

        # Format output
        if request.output_format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["filename", "caption", "success", "error"])

            for result in results:
                writer.writerow(
                    [
                        result["filename"],
                        result["caption"] or "",
                        result["success"],
                        result["error"] or "",
                    ]
                )

            return JSONResponse(
                {
                    "format": "csv",
                    "data": output.getvalue(),
                    "processing_time": processing_time,
                    "total_images": len(files),
                    "successful": sum(1 for r in results if r["success"]),
                }
            )

        else:  # JSON format
            return JSONResponse(
                {
                    "format": "json",
                    "results": results,
                    "processing_time": processing_time,
                    "total_images": len(files),
                    "successful": sum(1 for r in results if r["success"]),
                }
            )

    except Exception as e:
        logger.error(f"Batch captioning failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {str(e)}"
        )


@router.post("/writeback")
async def writeback_to_rag(
    request: WritebackRequest, rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """Write VLM-generated content back to RAG memory"""
    try:
        # Prepare memory content
        memory_content = {
            "type": "vlm_caption",
            "content": request.caption,
            "context": request.image_context,
            "timestamp": datetime.now().isoformat(),
            "source": "vlm_api",
        }

        # Add tags if requested
        if request.auto_tag:
            try:
                tagger = WD14Tagger()
                # Note: Would need image here for tagging, simplified for API
                memory_content["auto_tags"] = []
            except Exception as e:
                logger.warning(f"Auto-tagging failed: {e}")

        # Write to RAG memory
        doc_id = f"vlm_{uuid.uuid4().hex[:8]}"

        await rag_engine.write_memory(
            world_id=request.world_id,
            scope=request.scope,
            content=json.dumps(memory_content, ensure_ascii=False),
            doc_id=doc_id,
            metadata={
                "type": "vlm_generated",
                "scope": request.scope,
                "timestamp": datetime.now().isoformat(),
            },
        )

        return JSONResponse(
            {
                "success": True,
                "doc_id": doc_id,
                "world_id": request.world_id,
                "scope": request.scope,
                "message": "Content written to RAG memory successfully",
            }
        )

    except Exception as e:
        logger.error(f"RAG writeback failed: {e}")
        raise HTTPException(status_code=500, detail=f"Writeback failed: {str(e)}")


@router.get("/models")
async def list_available_models():
    """List available VLM models"""
    config = get_vlm_config()

    models = []
    for model_type, model_name in config.get("models", {}).items():
        models.append(
            {
                "type": model_type,
                "name": model_name,
                "default": model_type == config.get("default_model", "blip2"),
                "capabilities": {
                    "captioning": True,
                    "analysis": True,
                    "multilingual": model_type in ["blip2", "llava"],
                },
            }
        )

    return JSONResponse(
        {"models": models, "default_model": config.get("default_model", "blip2")}
    )


@router.get("/health")
async def vlm_health_check(captioner: VLMCaptioner = Depends(get_vlm_captioner)):
    """Health check for VLM services"""
    try:
        # Try to load default model
        default_model = captioner.config.get("default_model", "blip2")
        model = captioner.load_model(default_model)

        return JSONResponse(
            {
                "status": "healthy",
                "default_model": default_model,
                "models_loaded": list(captioner.model_cache.keys()),
                "gpu_available": (
                    torch.cuda.is_available() if "torch" in globals() else False
                ),
            }
        )

    except Exception as e:
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)


# Helper function for getting world data
async def get_world_data(rag_engine: ChineseRAGEngine, world_id: str) -> Dict[str, Any]:
    """Retrieve world data from RAG for consistency checking"""
    try:
        # Query for character and scene data
        character_results = await rag_engine.retrieve(
            query="characters personalities appearances",
            world_id=world_id,
            scopes=["world_lore"],
            top_k=10,
        )

        scene_results = await rag_engine.retrieve(
            query="scenes locations environments settings",
            world_id=world_id,
            scopes=["world_lore"],
            top_k=10,
        )

        # Parse results into structured data
        # This is a simplified version - in practice would need better parsing
        world_data = {"characters": {}, "scenes": {}, "lore": []}

        for result in character_results + scene_results:
            world_data["lore"].append(
                {
                    "content": result.get("content", ""),
                    "source": result.get("source", ""),
                }
            )

        return world_data

    except Exception as e:
        logger.warning(f"Failed to retrieve world data: {e}")
        return {"characters": {}, "scenes": {}, "lore": []}
