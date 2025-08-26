# api/routers/lora.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import psutil
import torch
import time
from api.schemas import LoRAListResponse, LoRALoadRequest, LoRAInfo
from ..dependencies import get_lora  # <- use DI

router = APIRouter(prefix="/lora", tags=["lora"])


@router.get(
    "/list", response_model=LoRAListResponse, summary="List available LoRA models"
)
async def list_loras(lora=Depends(get_lora)):
    try:
        items = lora.list()  # expected to return a list of dicts or ids
        infos: List[LoRAInfo] = []
        for it in items:
            if isinstance(it, str):
                infos.append(
                    LoRAInfo(
                        id=it,
                        name=it,
                        model_type="unknown",
                        rank=0,
                        path="",
                        loaded=False,
                    )
                )
            else:
                infos.append(
                    LoRAInfo(
                        id=it.get("id", it.get("name", "")),
                        name=it.get("name", it.get("id", "")),
                        model_type=it.get("model_type", "unknown"),
                        rank=it.get("rank", 0),
                        path=it.get("path", ""),
                        loaded=it.get("loaded", False),
                    )
                )
        return LoRAListResponse(loras=infos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load", summary="Load a LoRA model")
async def load_lora(req: LoRALoadRequest, lora=Depends(get_lora)):
    try:
        ok = lora.load(
            req.lora_id
        )  # optional: pass req.weight if your manager supports it
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"LoRA '{req.lora_id}' not found or failed to load",
            )
        return {
            "message": f"LoRA {req.lora_id} loaded",
            "weight": req.weight,
            "status": "success",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
