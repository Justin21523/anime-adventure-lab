# api/routers/models.py
"""
Model Management Router - Handles scanning and selection of local models.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel

from core.config import get_config
from core.shared_cache import get_shared_cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Models"])

class LocalModelInfo(BaseModel):
    name: str
    path: str
    type: str  # llm, vlm, t2i, embedding
    size_gb: float
    description: Optional[str] = None

class ModelSettingsUpdateRequest(BaseModel):
    chat_model: Optional[str] = None
    vqa_model: Optional[str] = None
    caption_model: Optional[str] = None
    sd_model: Optional[str] = None
    embedding_model: Optional[str] = None

@router.get("/local", response_model=List[LocalModelInfo])
async def list_local_models(
    type_filter: Optional[str] = Query(None, description="Filter by type (llm|vlm|t2i|embedding)")
):
    """Scan AI_MODELS_ROOT for available local models with exhaustive deep scanning."""
    try:
        # 1. Resolve models root
        models_root_str = os.getenv("AI_MODELS_ROOT", "/warehouse/ai_models")
        models_root = Path(models_root_str).resolve()
        
        logger.info(f"--- START EXHAUSTIVE MODEL SCAN ---")
        logger.info(f"Models Root: {models_root}")
        
        if not models_root.exists():
            logger.error(f"Models root directory NOT FOUND: {models_root}")
            # Try fallback to common WSL/Native C drive path
            fallback = Path("/mnt/c/ai_models").resolve()
            if fallback.exists():
                logger.info(f"Using fallback C drive models root: {fallback}")
                models_root = fallback
            else:
                logger.error("No valid models root found. Returning empty list.")
                return []

        # Target subdirectories as requested
        # Ensure diffusion is also under diffusion/ if that's the pattern, 
        # but user said /mnt/c/ai_models/diffusion/
        targets = {
            "llm": models_root / "language" / "llm",
            "vlm": models_root / "language" / "vlm",
            "t2i": models_root / "diffusion",
        }

        models = []
        
        def scan_target(target_dir: Path, model_type: str):
            if not target_dir.exists():
                logger.warning(f"Target sub-directory NOT FOUND: {target_dir}")
                return
            
            logger.info(f"Scanning {model_type} in {target_dir}...")
            
            for root, dirs, files in os.walk(target_dir, followlinks=True):
                root_path = Path(root)
                
                # A: Detect HF/Diffusers directory units
                if (root_path / "config.json").exists() or (root_path / "model_index.json").exists():
                    try:
                        try:
                            name = str(root_path.relative_to(models_root))
                        except ValueError:
                            name = root_path.name
                            
                        # Efficient size calculation (max 100 files)
                        size_bytes = 0
                        f_count = 0
                        for f in root_path.glob('*'):
                            if f.is_file():
                                size_bytes += f.stat().st_size
                                f_count += 1
                                if f_count > 100: break
                        
                        models.append(LocalModelInfo(
                            name=name,
                            path=str(root_path),
                            type=model_type,
                            size_gb=round(size_bytes / (1024**3), 2)
                        ))
                        logger.info(f"Detected model directory: {name}")
                        dirs[:] = [] # Stop recursion
                        continue
                    except Exception as e:
                        logger.error(f"Error processing model dir {root_path}: {e}")

                # B: Detect single file models
                valid_exts = (".safetensors", ".gguf", ".ckpt", ".bin", ".pth")
                for file in files:
                    if any(file.endswith(ext) for ext in valid_exts):
                        file_path = root_path / file
                        try:
                            size_bytes = file_path.stat().st_size
                            if size_bytes > 0.05 * (1024**3): # Min 50MB
                                try:
                                    name = str(file_path.relative_to(models_root))
                                except ValueError:
                                    name = file
                                    
                                models.append(LocalModelInfo(
                                    name=name,
                                    path=str(file_path),
                                    type=model_type,
                                    size_gb=round(size_bytes / (1024**3), 2)
                                ))
                                logger.info(f"Detected model file: {name}")
                        except Exception: pass

        if type_filter and type_filter in targets:
            scan_target(targets[type_filter], type_filter)
        else:
            for m_type, path in targets.items():
                scan_target(path, m_type)

        unique_models = []
        seen_paths = set()
        for m in models:
            if m.path not in seen_paths:
                unique_models.append(m)
                seen_paths.add(m.path)

        logger.info(f"--- END EXHAUSTIVE SCAN (Found {len(unique_models)}) ---")
        return unique_models
    except Exception as e:
        logger.error(f"FATAL Model scan failed: {e}", exc_info=True)
        raise HTTPException(500, f"Model scan failed: {str(e)}")


@router.get("/debug/paths")
async def debug_model_paths():
    """Debug endpoint to see what the container sees in the models directory."""
    try:
        models_root_str = os.getenv("AI_MODELS_ROOT", "/warehouse/ai_models")
        root = Path(models_root_str).resolve()
        
        structure = {}
        if root.exists():
            for entry in os.scandir(root):
                if entry.is_dir():
                    subdirs = []
                    try:
                        for sub in os.scandir(entry.path):
                            if sub.is_dir():
                                subdirs.append(sub.name)
                    except Exception: pass
                    structure[entry.name] = subdirs
                else:
                    structure[entry.name] = "file"
        
        return {
            "root_path": str(root),
            "exists": root.exists(),
            "env_var": models_root_str,
            "structure": structure
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/config")
async def get_current_model_config():
    """Get currently active model configuration."""
    config = get_config()
    return {
        "chat_model": config.model.chat_model,
        "vqa_model": config.model.vqa_model,
        "caption_model": config.model.caption_model,
        "sd_model": config.model.default_sd_model,
        "embedding_model": config.model.embedding_model,
    }

@router.put("/config")
async def update_model_config(settings: ModelSettingsUpdateRequest):
    """Update active model configuration and trigger reload."""
    config = get_config()
    changed = False
    
    updates = settings.model_dump(exclude_none=True)
    if not updates:
        return {"success": True, "message": "No changes requested"}

    # Update in-memory config
    if settings.chat_model:
        config.model.chat_model = settings.chat_model
        changed = True
    if settings.vqa_model:
        config.model.vqa_model = settings.vqa_model
        changed = True
    if settings.caption_model:
        config.model.caption_model = settings.caption_model
        changed = True
    if settings.sd_model:
        config.model.default_sd_model = settings.sd_model
        changed = True
    if settings.embedding_model:
        config.model.embedding_model = settings.embedding_model
        changed = True

    if changed:
        logger.info(f"Model configuration updated: {updates}")
        
    return {
        "success": True, 
        "updated": updates,
        "message": "Configuration updated. Models will be reloaded on next use."
    }
