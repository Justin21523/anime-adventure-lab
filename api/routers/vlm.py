# api/routers/vlm.py
"""
VLM Management Router
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from core.vlm.engine import get_vlm_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vlm", tags=["VLM"])


@router.get("/status")
async def get_vlm_status():
    """Get VLM engine status"""
    try:
        vlm_engine = get_vlm_engine()
        return vlm_engine.get_status()
    except Exception as e:
        raise HTTPException(500, f"Failed to get VLM status: {str(e)}")


@router.post("/load/{model_type}")
async def load_vlm_model(model_type: str):
    """Load specific VLM model"""
    try:
        vlm_engine = get_vlm_engine()
        if model_type == "caption":
            vlm_engine.load_caption_model()
            return {"message": "Caption model loaded successfully"}
        elif model_type == "vqa":
            vlm_engine.load_vqa_model()
            return {"message": "VQA model loaded successfully"}
        else:
            raise HTTPException(400, f"Unknown model type: {model_type}")
    except Exception as e:
        raise HTTPException(500, f"Failed to load VLM model: {str(e)}")


@router.post("/unload")
async def unload_vlm_models():
    """Unload all VLM models"""
    try:
        vlm_engine = get_vlm_engine()
        vlm_engine.unload_models()
        return {"message": "All VLM models unloaded successfully"}
    except Exception as e:
        raise HTTPException(500, f"Failed to unload VLM models: {str(e)}")


@router.post("/analyze_to_rag")
async def analyze_image_to_rag(
    file: UploadFile = File(...),
    world_id: str = Form("default"),
    prompt: Optional[str] = Form(None)
):
    """
    Use VLM to analyze an image and return structured Markdown for RAG ingestion.
    Useful for creating role settings from character art.
    """
    try:
        vlm_engine = get_vlm_engine()
        
        # Save uploaded file temporarily
        suffix = Path(file.filename or "image.png").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Default analysis prompt if none provided
            analysis_prompt = prompt or (
                "請詳細描述這張圖片中的角色或場景特徵。如果是角色，請描述其性別、年齡感、髮色、瞳色、服裝風格、攜帶的武器或配件。"
                "請務必以 Markdown 格式輸出，並包含 '# 角色描述' 或 '# 場景描述' 標題。"
            )
            
            # 1. Generate description via VLM
            description = await vlm_engine.vqa(
                image_path=tmp_path,
                prompt=analysis_prompt
            )
            
            # Return for preview (frontend will call /rag/ingest_structured next)
            return {
                "success": True,
                "suggested_title": file.filename or "分析結果",
                "markdown_content": description,
                "world_id": world_id
            }
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        logger.error(f"VLM Analysis to RAG failed: {e}")
        raise HTTPException(500, f"VLM Analysis error: {str(e)}")
