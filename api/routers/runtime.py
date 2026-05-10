# api/routers/runtime.py
"""
Runtime preset catalog router.

提供前端用來選擇「硬體/品質 preset」（例如 RTX 5080 16GB）：
- 預設 LLM (Qwen 7B) 的載入策略提示
- 預設 SDXL 的生成參數/VRAM 優化建議
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from core.runtime.catalog import load_runtime_preset_catalog
from schemas.runtime import (
    RuntimePreset,
    RuntimePresetCatalogResponse,
    RuntimePresetLLM,
    RuntimePresetT2I,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/runtime/presets", response_model=RuntimePresetCatalogResponse)
async def list_runtime_presets():
    """List runtime presets for UI selection (world/session)."""
    try:
        catalog = load_runtime_preset_catalog()
        default_preset_id = str(catalog.get("default_preset_id") or "").strip()
        presets_out = []
        for item in catalog.get("presets") or []:
            if not isinstance(item, dict):
                continue
            preset_id = str(item.get("preset_id") or "").strip()
            if not preset_id:
                continue
            presets_out.append(
                RuntimePreset(
                    preset_id=preset_id,
                    name=str(item.get("name") or preset_id),
                    description=str(item.get("description") or ""),
                    llm=RuntimePresetLLM(**(item.get("llm") or {})),
                    t2i=RuntimePresetT2I(**(item.get("t2i") or {})),
                )
            )

        if not default_preset_id:
            default_preset_id = presets_out[0].preset_id if presets_out else "rtx_5080_16gb"

        return RuntimePresetCatalogResponse(
            success=True,
            default_preset_id=default_preset_id,
            presets=presets_out,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to list runtime presets: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list runtime presets: {str(exc)}") from exc

