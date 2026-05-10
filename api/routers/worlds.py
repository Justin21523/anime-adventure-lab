# api/routers/worlds.py
"""
World Studio (WorldPacks) Router

WorldPack 用來把「世界設定 + 角色/NPC 模板 + 視覺風格(LoRA)」整合到 Story 主流程。
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from core.worldpacks import get_worldpack_manager
from schemas.world import (
    WorldAgentSuggestRequest,
    WorldAgentSuggestResponse,
    WorldCreateRequest,
    WorldPack,
    WorldSummary,
    WorldUpdateRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _worldpack_summary(pack: WorldPack) -> str:
    try:
        return (
            f"- world_id: {pack.world_id}\n"
            f"- name: {pack.name}\n"
            f"- description: {pack.description}\n"
            f"- setting: {pack.setting}\n"
            f"- difficulty: {pack.difficulty}\n"
            f"- characters: {len(pack.characters)}\n"
            f"- player_templates: {len(pack.player_templates)}\n"
            f"- default_loras: {len(pack.visual.default_loras)}"
        )
    except Exception:
        return f"- world_id: {getattr(pack, 'world_id', '?')}"


def _safe_list_loras() -> List[Dict[str, Any]]:
    try:
        from core.shared_cache import get_shared_cache
        from core.t2i.lora_manager import LoRAManager

        cache = get_shared_cache()
        mgr = LoRAManager(cache.get_path("CACHE_DIR"))
        return mgr.list_available_loras()
    except Exception as exc:  # noqa: BLE001
        logger.debug("List LoRAs skipped: %s", exc)
        return []


def _safe_rag_snippets(world_id: str, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """Best-effort RAG snippets without forcing embedding model load."""
    try:
        from core.rag import get_rag_engine

        rag_engine = get_rag_engine()
        target = str(world_id or "default").strip() or "default"
        q = str(query or "").strip().lower()
        tokens = [t for t in re.split(r"\s+", q) if t][:5]

        candidates: List[tuple[int, Any]] = []
        for doc in rag_engine.documents.values():
            metadata = doc.metadata or {}
            if str(metadata.get("world_id", "default")).strip() != target:
                continue
            content = str(doc.content or "")
            content_l = content.lower()
            score = 0
            if q:
                score += content_l.count(q)
            for t in tokens:
                score += content_l.count(t)
            candidates.append((score, doc))

        candidates.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, doc in candidates[: max(0, int(top_k or 0))]:
            out.append(
                {
                    "doc_id": getattr(doc, "doc_id", ""),
                    "score": score,
                    "content": str(getattr(doc, "content", "") or "")[:900],
                    "metadata": getattr(doc, "metadata", {}) or {},
                }
            )
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("RAG snippets skipped: %s", exc)
        return []


def _merge_item(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(existing or {})
    for k, v in (incoming or {}).items():
        if v is None:
            continue
        if isinstance(v, str):
            if not v.strip():
                continue
            out[k] = v
            continue
        out[k] = v
    return out


def _upsert_by_id(
    existing_items: List[Dict[str, Any]],
    incoming_items: List[Dict[str, Any]],
    *,
    key: str,
) -> List[Dict[str, Any]]:
    existing = [dict(x) for x in (existing_items or []) if isinstance(x, dict)]
    incoming = [dict(x) for x in (incoming_items or []) if isinstance(x, dict)]
    index: Dict[str, int] = {}
    for i, item in enumerate(existing):
        item_id = str(item.get(key) or "").strip()
        if item_id and item_id not in index:
            index[item_id] = i

    for inc in incoming:
        inc_id = str(inc.get(key) or "").strip()
        if not inc_id:
            continue
        if inc_id in index:
            existing[index[inc_id]] = _merge_item(existing[index[inc_id]], inc)
        else:
            index[inc_id] = len(existing)
            existing.append(inc)

    return existing


def _apply_worldpack_patch(base: WorldPack, patch: Dict[str, Any]) -> WorldPack:
    base_dict = base.model_dump()
    patch = dict(patch or {})

    # Keep world_id immutable
    patch.pop("world_id", None)
    candidate = dict(base_dict)

    for k in ["name", "description", "setting", "difficulty"]:
        v = patch.get(k)
        if isinstance(v, str) and v.strip():
            candidate[k] = v.strip()

    if isinstance(patch.get("world_flags"), dict):
        candidate_flags = dict(candidate.get("world_flags") or {})
        for fk, fv in patch["world_flags"].items():
            kk = str(fk or "").strip()
            if kk:
                candidate_flags[kk] = bool(fv)
        candidate["world_flags"] = candidate_flags

    if isinstance(patch.get("visual"), dict):
        visual = dict(candidate.get("visual") or {})
        v_patch = patch["visual"]
        if isinstance(v_patch.get("prompt_prefix"), str):
            visual["prompt_prefix"] = v_patch["prompt_prefix"]
        if isinstance(v_patch.get("negative_prompt"), str):
            visual["negative_prompt"] = v_patch["negative_prompt"]
        if v_patch.get("base_model") is None or isinstance(v_patch.get("base_model"), str):
            visual["base_model"] = v_patch.get("base_model")
        if isinstance(v_patch.get("default_loras"), list):
            visual["default_loras"] = [
                {
                    "lora_id": str(x.get("lora_id") or "").strip(),
                    "weight": float(x.get("weight", 0.8) or 0.8),
                }
                for x in v_patch.get("default_loras", [])
                if isinstance(x, dict) and str(x.get("lora_id") or "").strip()
            ]
        candidate["visual"] = visual

    if isinstance(patch.get("rag_profile"), dict):
        rag_profile = dict(candidate.get("rag_profile") or {})
        r_patch = patch["rag_profile"]
        if "enable_rerank" in r_patch:
            rag_profile["enable_rerank"] = bool(r_patch.get("enable_rerank"))
        candidate["rag_profile"] = rag_profile

    if isinstance(patch.get("player_templates"), list):
        candidate["player_templates"] = _upsert_by_id(
            list(candidate.get("player_templates") or []),
            patch["player_templates"],
            key="template_id",
        )

    if isinstance(patch.get("characters"), list):
        candidate["characters"] = _upsert_by_id(
            list(candidate.get("characters") or []),
            patch["characters"],
            key="character_id",
        )

    return WorldPack(**candidate)


@router.get("/worlds", response_model=List[WorldSummary])
async def list_worlds():
    """List available worlds for selector UI."""
    manager = get_worldpack_manager()
    return manager.list_worldpacks()


@router.post("/worlds", response_model=WorldPack)
async def create_world(request: WorldCreateRequest):
    """Create a new worldpack."""
    manager = get_worldpack_manager()
    try:
        return manager.create_worldpack(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Create world failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create world")


@router.get("/worlds/{world_id}", response_model=WorldPack)
async def get_world(world_id: str):
    """Get full worldpack detail."""
    manager = get_worldpack_manager()
    pack = manager.get_worldpack(world_id)
    if pack is None:
        raise HTTPException(status_code=404, detail=f"World not found: {world_id}")
    return pack


@router.put("/worlds/{world_id}", response_model=WorldPack)
async def update_world(world_id: str, request: WorldUpdateRequest):
    """Update a worldpack (full replace)."""
    manager = get_worldpack_manager()
    try:
        return manager.update_worldpack(world_id, request.world)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Update world failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update world")


@router.delete("/worlds/{world_id}")
async def delete_world(world_id: str):
    """Delete a worldpack."""
    manager = get_worldpack_manager()
    try:
        manager.delete_worldpack(world_id)
        return {"success": True}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Delete world failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete world")


@router.post("/worlds/{world_id}/agents/suggest", response_model=WorldAgentSuggestResponse)
async def suggest_worldpack_update(world_id: str, request: WorldAgentSuggestRequest):
    """
    Multi-agent World Studio endpoint:
    - 讀取目前 worldpack
    - 取出 world_id 對應的 RAG snippets（best-effort）
    - 以多個子代理產生 patch，回傳預覽或直接套用保存
    """
    manager = get_worldpack_manager()
    pack = manager.get_worldpack(world_id)
    if pack is None:
        raise HTTPException(status_code=404, detail=f"World not found: {world_id}")

    rag_snippets = _safe_rag_snippets(world_id, request.instruction, top_k=request.rag_top_k)
    loras = _safe_list_loras() if request.include_visual else []

    try:
        from core.agents.world_studio_orchestrator import WorldStudioOrchestrator

        max_llm_calls = int(os.getenv("WORLD_STUDIO_MAX_LLM_CALLS", "5") or 5)
        orchestrator = WorldStudioOrchestrator(max_llm_calls=max_llm_calls)

        result = orchestrator.suggest_worldpack_patch(
            world_id=world_id,
            instruction=request.instruction,
            worldpack_summary=_worldpack_summary(pack),
            rag_snippets=rag_snippets,
            available_loras=loras,
            include_visual=bool(request.include_visual),
            max_new_characters=int(request.max_new_characters or 0),
            max_new_player_templates=int(request.max_new_player_templates or 0),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("WorldStudioOrchestrator failed: %s", exc, exc_info=True)
        return WorldAgentSuggestResponse(
            success=False,
            applied=False,
            world_id=world_id,
            patch={},
            worldpack=pack,
            contributors=[],
            errors=[str(exc)],
        )

    patch = result.get("patch") or {}
    contributors = list(result.get("contributors") or [])
    errors = list(result.get("errors") or [])

    try:
        candidate = _apply_worldpack_patch(pack, patch)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"patch 無法套用到 worldpack：{exc}")
        return WorldAgentSuggestResponse(
            success=False,
            applied=False,
            world_id=world_id,
            patch=patch,
            worldpack=pack,
            contributors=contributors,
            errors=errors,
        )

    if not request.apply:
        return WorldAgentSuggestResponse(
            success=True,
            applied=False,
            world_id=world_id,
            patch=patch,
            worldpack=candidate,
            contributors=contributors,
            errors=errors,
        )

    try:
        saved = manager.update_worldpack(world_id, candidate)
        return WorldAgentSuggestResponse(
            success=True,
            applied=True,
            world_id=world_id,
            patch=patch,
            worldpack=saved,
            contributors=contributors,
            errors=errors,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"保存 worldpack 失敗：{exc}")
        return WorldAgentSuggestResponse(
            success=False,
            applied=False,
            world_id=world_id,
            patch=patch,
            worldpack=pack,
            contributors=contributors,
            errors=errors,
        )
