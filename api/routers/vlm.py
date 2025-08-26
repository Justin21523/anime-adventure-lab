# api/routers/vlm.py
"""
VLM API Router - FastAPI endpoints for vision-language model operations
Handles image captioning, analysis, and consistency checking
"""
from __future__ import annotations
import io, tempfile, os, logging, inspect
import json
import uuid
import torch
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from ..dependencies import get_vlm, get_rag, get_vlm_settings

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
async def _load_image(file: UploadFile) -> Image.Image:
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def _save_temp_image(pil: Image.Image) -> str:
    """Save to a temp file when engine expects a path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil.save(tmp.name)
        return tmp.name


# API Endpoints
@router.post("/caption", response_model=CaptionResponse)
async def caption_image(
    file: UploadFile = File(...),
    request: CaptionRequest = Depends(),
    vlm=Depends(get_vlm),
):
    """Generate caption using shared VLM engine."""
    start = datetime.now()
    try:
        img = await _load_image(file)

        # Prefer engine.caption(image=...), fallback to engine.caption(path)
        try:
            caption = vlm.caption(
                image=img, model_type=request.model_type, prompt=request.prompt
            )
        except Exception:
            path = _save_temp_image(img)
            try:
                caption = vlm.caption(
                    path, model_type=request.model_type, prompt=request.prompt
                )
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass

        if request.language == "zh":
            caption = f"[ZH] {caption}"  # placeholder translation

        dt = (datetime.now() - start).total_seconds()
        return CaptionResponse(
            caption=caption,
            model_type=request.model_type or "default",
            processing_time=dt,
            image_info={"size": img.size, "mode": img.mode, "filename": file.filename},
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Caption failed: {e}")
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {e}")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    request: AnalysisRequest = Depends(),
    vlm=Depends(get_vlm),
    rag=Depends(get_rag),
):
    """Comprehensive image analysis via VLM; optional consistency against RAG."""
    start = datetime.now()
    try:
        img = await _load_image(file)

        # analysis(image=...) fallback to path
        try:
            analysis = vlm.analyze(image=img, model_type=request.model_type)
        except Exception:
            path = _save_temp_image(img)
            try:
                analysis = vlm.analyze(path, model_type=request.model_type)
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass

        tags = None
        try:
            if request.include_tags:
                from core.vlm.tagger import WD14Tagger  # optional

                tags = WD14Tagger().tag_image(img)
        except Exception as e:
            logger.warning(f"Tagger unavailable: {e}")
            tags = []

        consistency_report = None
        if request.check_consistency and request.world_id:
            try:
                # Best-effort world data via RAG
                get_stats = getattr(rag, "retrieve", None)
                world = []
                if callable(get_stats):
                    world_hits = rag.retrieve(
                        query="characters scenes lore",
                        world_id=request.world_id,
                        top_k=10,
                        alpha=0.7,
                    )
                    world = [
                        {
                            "content": getattr(h, "text", ""),
                            "score": getattr(h, "score", 0.0),
                        }
                        for h in world_hits
                    ]
                consistency_report = {"world_snippets": world[:10]}
            except Exception as e:
                logger.warning(f"Consistency check failed: {e}")
                consistency_report = {"error": str(e)}

        dt = (datetime.now() - start).total_seconds()
        return AnalysisResponse(
            analysis=analysis or {},
            tags=tags,
            consistency_report=consistency_report,
            model_type=request.model_type or "default",
            processing_time=dt,
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analyze failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@router.post("/batch_caption")
async def batch_caption_images(
    files: List[UploadFile] = File(...),
    request: BatchCaptionRequest = Depends(),
    vlm=Depends(get_vlm),
):
    """Caption multiple images; graceful failures per item."""
    start = datetime.now()
    try:
        if len(files) > 20:
            raise HTTPException(status_code=400, detail="Too many files (max 20)")

        results = []
        for i, f in enumerate(files):
            try:
                img = await _load_image(f)
                try:
                    cap = vlm.caption(
                        image=img, model_type=request.model_type, prompt=request.prompt
                    )
                except Exception:
                    p = _save_temp_image(img)
                    try:
                        cap = vlm.caption(
                            p, model_type=request.model_type, prompt=request.prompt
                        )
                    finally:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                results.append(
                    {
                        "index": i,
                        "filename": f.filename,
                        "caption": cap,
                        "success": True,
                        "error": None,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "index": i,
                        "filename": f.filename,
                        "caption": None,
                        "success": False,
                        "error": str(e),
                    }
                )

        dt = (datetime.now() - start).total_seconds()
        if request.output_format == "csv":
            import csv, io as _io

            out = _io.StringIO()
            w = csv.writer(out)
            w.writerow(["filename", "caption", "success", "error"])
            for r in results:
                w.writerow(
                    [r["filename"], r["caption"] or "", r["success"], r["error"] or ""]
                )
            return JSONResponse(
                {
                    "format": "csv",
                    "data": out.getvalue(),
                    "processing_time": dt,
                    "total_images": len(files),
                    "successful": sum(1 for r in results if r["success"]),
                }
            )
        return JSONResponse(
            {
                "format": "json",
                "results": results,
                "processing_time": dt,
                "total_images": len(files),
                "successful": sum(1 for r in results if r["success"]),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")


@router.post("/writeback")
async def writeback_to_rag(request: WritebackRequest, rag=Depends(get_rag)):
    """Write VLM-generated caption back to RAG memory (best-effort)."""
    try:
        doc_id = f"vlm_{uuid.uuid4().hex[:8]}"
        payload = {
            "type": "vlm_caption",
            "content": request.caption,
            "context": request.image_context,
            "timestamp": datetime.now().isoformat(),
            "scope": request.scope,
        }
        # Prefer write_memory(...); fallback to add_document(...)
        writer = getattr(rag, "write_memory", None)
        if callable(writer):
            if inspect.iscoroutinefunction(writer):
                await writer(
                    world_id=request.world_id,
                    scope=request.scope,
                    content=json.dumps(payload, ensure_ascii=False),
                    doc_id=doc_id,
                    metadata={"source": "vlm"},
                )
            else:
                writer(
                    world_id=request.world_id,
                    scope=request.scope,
                    content=json.dumps(payload, ensure_ascii=False),
                    doc_id=doc_id,
                    metadata={"source": "vlm"},
                )
        else:
            adder = getattr(rag, "add_document", None)
            if callable(adder):
                adder(
                    json.dumps(payload, ensure_ascii=False),
                    {"doc_id": doc_id, "world_id": request.world_id},
                )
            else:
                raise HTTPException(
                    status_code=501,
                    detail="RAG memory write is not supported by current engine",
                )

        return {
            "success": True,
            "doc_id": doc_id,
            "world_id": request.world_id,
            "scope": request.scope,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Writeback failed: {e}")


@router.get("/models")
async def list_available_models(
    vlm=Depends(get_vlm), vlm_cfg=Depends(get_vlm_settings)
):
    """List available VLM models based on app settings."""
    try:
        models_cfg = getattr(vlm_cfg, "models", {}) if vlm_cfg else {}
        default_name = (
            getattr(vlm_cfg, "default_model", "blip2") if vlm_cfg else "blip2"
        )
        models = [
            {
                "type": k,
                "name": v,
                "default": (k == default_name),
                "capabilities": {
                    "captioning": True,
                    "analysis": True,
                    "multilingual": k in ["blip2", "llava"],
                },
            }
            for k, v in (models_cfg.items() if isinstance(models_cfg, dict) else [])
        ]
        return {"models": models, "default_model": default_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def vlm_health_check(vlm=Depends(get_vlm)):
    """Quick VLM health check."""
    try:
        # If engine exposes a quick probe, use it; else just return ok
        probe = getattr(vlm, "health", None)
        if callable(probe):
            return probe()
        return {
            "status": "healthy",
            "models_loaded": list(getattr(vlm, "model_cache", {}).keys()),
        }
    except Exception as e:
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)
