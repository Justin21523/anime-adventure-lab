# api/routers/story.py
"""
Story Engine Router (separate from game.py)
Provides story-specific APIs with LLM/agent/RAG integration.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse

from core.story.engine import get_story_engine
from core.train.job_manager import TrainJobManager
from core.train.job_context import get_job_context, job_context
from core.agents import (
    StoryContext as AgentStoryContext,
    get_story_agent_manager,
)
from core.exceptions import GameError, SessionNotFoundError, InvalidChoiceError
from schemas.game import GamePersonaInfo
from schemas.story import (
    SceneImage,
    StoryAgentActionRequest,
    StoryAgentActionResponse,
    StoryAgentProfilePatchRequest,
    StoryAgentProfileResponse,
    StoryAgentProfileUpdateRequest,
    StoryChoicePreview,
    StoryContextSnapshot,
    StoryExportResponse,
    StoryImportRequest,
    StoryImportResponse,
    StoryKnowledgeSearchRequest,
    StoryKnowledgeSearchResponse,
    StoryMetricsResponse,
    StorySessionCreateRequest,
    StorySessionDetail,
    StorySessionInfo,
    StoryTurnRequest,
    StoryTurnJobResponse,
    StoryTurnResponse,
    StoryTurnHistoryEntry,
    StoryWorldSyncRequest,
    StoryWorldSyncResponse,
    StoryWorldWritebackSuggestRequest,
    StoryWorldWritebackSuggestResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()
story_turn_job_manager = TrainJobManager()


def _run_jobs_sync_fallback() -> bool:
    """Default to sync jobs for local/portfolio demo; real worker deploys can opt out."""
    raw = os.getenv("JOBS_SYNC_FALLBACK", os.getenv("API_SYNC_JOBS", "1"))
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


# Helpers ---------------------------------------------------------------------
def _job_report(stage: str, progress: float, message: Optional[str] = None, **meta: Any) -> None:
    """Best-effort progress reporting for background story_turn jobs."""
    try:
        ctx = get_job_context()
        if not ctx or str(ctx.job_type or "").strip() != "story_turn":
            return

        job_id = str(ctx.job_id or "").strip()
        if not job_id:
            return

        existing = story_turn_job_manager.get_job(job_id, auto_progress=False) or {}
        status = str(existing.get("status") or "").lower()
        if status in {"completed", "failed", "cancelled"}:
            return
        if bool(existing.get("cancel_requested")):
            return

        try:
            p = float(progress)
        except Exception:
            p = 0.0
        if p != p:  # NaN
            p = 0.0
        p = max(0.0, min(100.0, p))

        now = datetime.utcnow().isoformat()

        events = existing.get("stage_events")
        if not isinstance(events, list):
            events = []
        event: Dict[str, Any] = {
            "ts": now,
            "stage": str(stage),
            "progress": round(p, 2),
        }
        if message:
            event["message"] = str(message)
        if meta:
            event["meta"] = meta
        events.append(event)
        if len(events) > 50:
            events = events[-50:]

        updates: Dict[str, Any] = {
            "status": "running",
            "progress": round(p, 2),
            "stage": str(stage),
            "stage_message": str(message) if message else None,
            "stage_updated_at": now,
            "stage_events": events,
        }
        if not existing.get("started_at"):
            updates["started_at"] = now
        story_turn_job_manager.update_job(job_id, **updates)
    except Exception:
        return


def _deep_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge dicts (best-effort).

    - Only merges nested dict values recursively.
    - For non-dict values, patch overwrites base.
    - Does not mutate inputs.
    """
    if not isinstance(base, dict):
        base = {}
    if not isinstance(patch, dict):
        return dict(base)

    merged: Dict[str, Any] = dict(base)
    for key, patch_value in patch.items():
        if isinstance(patch_value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged.get(key, {}), patch_value)  # type: ignore[arg-type]
        else:
            merged[key] = patch_value
    return merged


def _build_scene_context(context_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Build stable scene_context payload for scene-image generation jobs."""
    return {
        "location": context_snapshot.get("current_scene", {}).get("name", "unknown place"),
        "time": context_snapshot.get("time_of_day", "daytime"),
        "atmosphere": context_snapshot.get("atmosphere", "neutral"),
        "characters": [
            char.get("name", "")
            for char in context_snapshot.get("present_characters", [])
        ],
        "world_id": context_snapshot.get("world_id", "default"),
        "runtime_preset_id": context_snapshot.get("runtime_preset_id"),
        "scene_transition": context_snapshot.get("scene_transition", False),
        "is_major_event": context_snapshot.get("is_major_event", False),
        "weather": context_snapshot.get("weather", ""),
    }


def _enqueue_scene_image_job(
    *,
    session,
    narrative_text: str,
    context_snapshot: Dict[str, Any],
    force: bool = False,
) -> Optional[str]:
    """Dispatch a background scene image job (Celery vision queue)."""
    try:
        from core.train.job_manager import TrainJobManager
        from workers.tasks.t2i import story_scene_image_task

        job_manager = TrainJobManager()

        turn = None
        try:
            if getattr(session, "history", None) and isinstance(session.history[-1], dict):
                turn = int(session.history[-1].get("turn", 0) or 0)
        except Exception:
            turn = None

        width = 768
        height = 768
        steps = 25
        guidance_scale = 7.0

        try:
            preset_id = str(context_snapshot.get("runtime_preset_id") or "").strip()
            if preset_id:
                from core.runtime.catalog import get_runtime_preset

                preset = get_runtime_preset(preset_id) or {}
                t2i = preset.get("t2i") if isinstance(preset.get("t2i"), dict) else {}
                width = int(t2i.get("default_width", width) or width)
                height = int(t2i.get("default_height", height) or height)
                steps = int(t2i.get("default_steps", steps) or steps)
                guidance_scale = float(t2i.get("default_guidance_scale", guidance_scale) or guidance_scale)

                max_w = int(t2i.get("max_width", width) or width)
                max_h = int(t2i.get("max_height", height) or height)
                max_steps = int(t2i.get("max_steps", steps) or steps)
                width = max(256, min(width, max_w))
                height = max(256, min(height, max_h))
                steps = max(1, min(steps, max_steps))
        except Exception:
            pass

        payload = {
            "session_id": getattr(session, "session_id", ""),
            "turn": turn,
            "force": bool(force),
            "scene_context": _build_scene_context(context_snapshot),
            "narrative_text": narrative_text,
            # Use conservative defaults; can be overridden later via UI.
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
        }

        job_id = job_manager.create_job("scene_image", payload, status="queued")
        async_result = story_scene_image_task.delay({"job_id": job_id, "payload": payload})
        try:
            job_manager.update_job(job_id, celery_task_id=str(async_result.id))
        except Exception:
            pass
        return job_id
    except Exception as exc:  # noqa: BLE001
        logger.info("Scene image job dispatch skipped: %s", exc)
        return None


def _enqueue_character_portrait_job(
    *,
    character_id: str,
    character_name: str,
    appearance_desc: str,
    world_id: str,
    visual_style: Optional[str] = None,
) -> Optional[str]:
    """Dispatch a background character portrait generation job."""
    try:
        from core.train.job_manager import TrainJobManager
        from workers.tasks.t2i import story_character_portrait_task

        job_manager = TrainJobManager()
        payload = {
            "character_id": character_id,
            "character_name": character_name,
            "appearance_desc": appearance_desc,
            "world_id": world_id,
            "visual_style": visual_style,
        }

        job_id = job_manager.create_job("character_portrait", payload, status="queued")
        async_result = story_character_portrait_task.delay({"job_id": job_id, "payload": payload})
        try:
            job_manager.update_job(job_id, celery_task_id=str(async_result.id))
        except Exception:
            pass
        return job_id
    except Exception as exc:  # noqa: BLE001
        logger.info("Character portrait job dispatch skipped: %s", exc)
        return None


def _safe_rag_search(
    query: str,
    top_k: int = 3,
    world_id: Optional[str] = None,
    *,
    enable_rerank: Optional[bool] = None,
    rerank_top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Try to retrieve RAG context; degrade gracefully offline."""
    if not query:
        return []
    try:
        from core.rag import get_rag_engine

        rag_engine = get_rag_engine()
        results = rag_engine.search(
            query,
            top_k=top_k,
            world_id=str(world_id).strip() if world_id else None,
            enable_rerank=enable_rerank,
            rerank_top_k=rerank_top_k,
        )
        simplified = []
        for res in results:
            metadata = res.document.metadata or {}
            simplified.append(
                {
                    "content": res.document.content,
                    "score": float(getattr(res, "score", 0.0)),
                    "semantic_score": getattr(res, "semantic_score", None),
                    "bm25_score": getattr(res, "bm25_score", None),
                    "combined_score": getattr(res, "combined_score", None),
                    "rerank_score": getattr(res, "rerank_score", None),
                    "metadata": metadata,
                }
            )
        return simplified
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG search skipped: %s", exc)
        return []


def _world_has_rag_documents(world_id: str) -> bool:
    """Best-effort check: does this world_id have any indexed RAG chunks?"""
    try:
        from core.rag import get_rag_engine

        rag_engine = get_rag_engine()
        target = str(world_id or "default").strip() or "default"
        for doc in rag_engine.documents.values():
            metadata = doc.metadata or {}
            if str(metadata.get("world_id", "default")).strip() == target:
                return True
        return False
    except Exception as exc:  # noqa: BLE001
        logger.debug("RAG availability check skipped: %s", exc)
        return False


def _world_enable_rerank(world_id: str) -> bool:
    """Best-effort world default: should reranker be enabled for this world?"""
    try:
        from core.worldpacks import get_worldpack_manager

        wpm = get_worldpack_manager()
        pack = wpm.get_worldpack(str(world_id or "default").strip() or "default")
        if not pack:
            return False
        rag_profile = getattr(pack, "rag_profile", None)
        return bool(getattr(rag_profile, "enable_rerank", False)) if rag_profile else False
    except Exception:
        return False


def _compact_knowledge_used(
    knowledge_used: List[Dict[str, Any]] | None,
    *,
    max_items: int = 5,
    max_chars: int = 420,
) -> List[Dict[str, Any]]:
    """Reduce RAG snippets for safe persistence in session.history."""
    compact: List[Dict[str, Any]] = []
    for k in (knowledge_used or [])[: max(0, int(max_items or 0))]:
        if not isinstance(k, dict):
            continue
        content = str(k.get("content") or "").strip()
        if max_chars and len(content) > int(max_chars):
            content = content[: int(max_chars)] + "…"
        item: Dict[str, Any] = {
            "content": content,
            "score": float(k.get("score", 0.0) or 0.0),
            "metadata": k.get("metadata") or {},
        }
        for key in ["semantic_score", "bm25_score", "combined_score", "rerank_score"]:
            if key in k:
                try:
                    item[key] = float(k.get(key)) if k.get(key) is not None else None
                except Exception:
                    item[key] = k.get(key)
        compact.append(item)
    return compact


def _load_session_agent_profile(session) -> Dict[str, Any]:
    """Best-effort load of session.story_context.agent_profile (as plain dict)."""
    try:
        story_ctx = getattr(getattr(session, "current_state", None), "story_context", {}) or {}
        if isinstance(story_ctx, dict):
            raw = story_ctx.get("agent_profile") or {}
        else:
            raw = {}
    except Exception:
        raw = {}

    if isinstance(raw, dict):
        return raw
    try:
        return dict(raw)  # type: ignore[arg-type]
    except Exception:
        return {}


async def _retrieve_story_context(
    session_id: str,
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """Retrieve story context using memory manager"""
    try:
        from core.story.memory_manager import get_memory_manager

        memory_manager = get_memory_manager(session_id)
        context = await memory_manager.retrieve_relevant_context(
            query=query,
            max_results=top_k,
            include_short_term=True
        )
        return context
    except Exception as exc:  # noqa: BLE001
        logger.warning("Story context retrieval skipped: %s", exc)
        return {"short_term": [], "summaries": [], "rag_results": []}


def _filter_text(text: str) -> str:
    """Run safety filter to avoid unsafe prompts."""
    try:
        from core.safety import get_content_filter

        cf = get_content_filter()
        result = cf.check_text_safety(text)
        return result.get("filtered_text", text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Safety filter fallback: %s", exc)
        return text


async def _generate_scene_image(
    narrative_text: str,
    context_snapshot: Dict[str, Any],
    force: bool = False
) -> Optional[SceneImage]:
    """Generate scene image if triggered by story context."""
    try:
        from core.story.t2i_integration import get_t2i_integration

        t2i_integration = get_t2i_integration()

        # Build scene context from story context
        scene_context = _build_scene_context(context_snapshot)

        # Generate scene image
        result = await t2i_integration.generate_scene_image(
            scene_context=scene_context,
            narrative_text=narrative_text,
            force=force
        )

        if result:
            return SceneImage(
                image_url=result.image_url,
                prompt=result.prompt,
                negative_prompt=result.negative_prompt,
                generation_time=result.generation_time,
                seed=result.seed,
                width=result.width,
                height=result.height
            )
        return None

    except Exception as exc:  # noqa: BLE001
        logger.warning("Scene image generation skipped: %s", exc)
        return None


def _build_agent_context(session) -> AgentStoryContext:
    """Convert engine session into agent context."""
    history = []
    for item in getattr(session, "history", [])[-5:]:
        history.append(
            {
                "action": item.get("player_input"),
                "result": item.get("response"),
                "choice_id": item.get("choice_id"),
            }
        )

    available_actions = []
    for c in getattr(session.current_state, "available_choices", []) or []:
        available_actions.append(c.get("text") or c.get("choice_id", ""))

    return AgentStoryContext(
        story_id=session.session_id,
        character_name=session.player_name,
        current_scene=session.current_state.scene_id,
        character_state={
            "stats": session.stats.to_dict(),
            "flags": session.current_state.flags,
        },
        story_history=history,
        available_actions=available_actions,
        narrative_style=session.current_state.story_context.get(
            "setting", "adventure"
        ),
    )


async def _agent_assist(
    session,
    player_input: str,
    scenario_type: Optional[str] = None,
    scenario_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Invoke story agent to get guidance; fallback to stub."""
    manager = get_story_agent_manager()
    story_context = _build_agent_context(session)
    scenario_data = scenario_data or {}

    try:
        if scenario_type:
            # Multi-agent scenario
            result = await manager.process_complex_story_scenario(
                story_context=story_context,
                scenario_type=scenario_type,
                scenario_data=scenario_data,
            )
            return {
                "success": result.get("success", False),
                "agent_steps": result.get("scenario_result", {}).get("steps_run"),
                "tools_used": result.get("scenario_result", {}).get("tools_used"),
                "narrative": (
                    result.get("scenario_result", {})
                    .get("final_output", "")
                    .strip()
                ),
            }
        # Single agent action
        agent = manager.get_story_agent("narrative")
        agent_result = await agent.process_story_action(  # type: ignore[call-arg]
            story_context=story_context,
            player_action=player_input,
            action_parameters=scenario_data,
        )
        if agent_result.get("success"):
            story_payload = agent_result.get("story_response", {})
            return {
                "success": True,
                "narrative": story_payload.get("narrative", ""),
                "available_actions": story_payload.get("available_actions", []),
                "consequences": story_payload.get("consequences", []),
                "agent_steps": agent_result.get("agent_steps"),
                "tools_used": agent_result.get("tools_used"),
            }
        return {"success": False, "narrative": agent_result.get("error", "")}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent assist fallback: %s", exc)
        return {
            "success": False,
            "narrative": f"You {player_input}. (agent fallback)",
            "available_actions": ["continue", "wait", "look around"],
            "consequences": [],
            "fallback_used": True,
        }


# Routes ---------------------------------------------------------------------
@router.post("/story/session", response_model=StoryTurnResponse)
async def create_story_session(request: StorySessionCreateRequest):
    """Create a new story session with optional RAG and agent assist."""
    try:
        story_engine = get_story_engine()
        session = story_engine.create_session(
            player_name=request.player_name,
            persona_id=request.persona_id,
            setting=request.setting,
            difficulty=request.difficulty,
            world_id=request.world_id,
            player_template_id=request.player_template_id,
            runtime_preset_id=getattr(request, "runtime_preset_id", None),
        )
        effective_world_id = getattr(session, "world_id", None) or request.world_id or "default"
        rag_available = _world_has_rag_documents(effective_world_id)
        world_rerank_default = _world_enable_rerank(effective_world_id)

        requested_mode = getattr(request, "rag_mode", None)
        if requested_mode in {"auto", "on", "off"}:
            if requested_mode == "auto":
                rag_auto = True
                effective_enrich_with_rag = bool(rag_available)
            elif requested_mode == "on":
                rag_auto = False
                effective_enrich_with_rag = True
            else:  # off
                rag_auto = False
                effective_enrich_with_rag = False
        else:
            rag_auto = request.enrich_with_rag is None
            effective_enrich_with_rag = (
                bool(request.enrich_with_rag)
                if request.enrich_with_rag is not None
                else bool(rag_available)
            )
        # Persist preference so StoryAgentLayer / orchestrator can use it for agent-side RAG decisions
        try:
            session.current_state.story_context["rag_auto"] = bool(rag_auto)
            session.current_state.story_context["rag_available"] = bool(rag_available)
            session.current_state.story_context["enrich_with_rag"] = bool(effective_enrich_with_rag)
            if request.rag_query is not None:
                session.current_state.story_context["rag_query"] = request.rag_query
        except Exception:  # noqa: BLE001
            pass

        requested_rerank_mode = getattr(request, "rerank_mode", None)
        if requested_rerank_mode not in {"auto", "on", "off"}:
            requested_rerank_mode = "auto"

        effective_enable_rerank = (
            bool(world_rerank_default)
            if requested_rerank_mode == "auto"
            else bool(requested_rerank_mode == "on")
        )
        try:
            session.current_state.story_context["rerank_mode"] = requested_rerank_mode
        except Exception:
            pass

        # Optional RAG context
        knowledge_used: List[Dict[str, Any]] = []
        raw_opening_input = (
            request.initial_prompt.strip()
            if request.initial_prompt and request.initial_prompt.strip()
            else f"開始故事，玩家 {request.player_name}。"
        )
        opening_input = raw_opening_input
        if effective_enrich_with_rag:
            knowledge_used = _safe_rag_search(
                request.rag_query or request.setting,
                top_k=3,
                world_id=effective_world_id,
                enable_rerank=effective_enable_rerank,
            )
            if knowledge_used:
                context_texts = [k["content"] for k in knowledge_used[:2]]
                opening_input += "\n背景資訊：" + " ".join(context_texts)

        opening_input = _filter_text(opening_input)

        # Optional agent overlay
        agent_overlay: Optional[Dict[str, Any]] = None
        if request.use_agent:
            agent_overlay = await _agent_assist(session, opening_input)
            if agent_overlay.get("narrative"):
                opening_input += f"\n[agent_hint] {agent_overlay['narrative']}"

        enriched_opening_input = opening_input

        turn_result = await story_engine.process_turn(
            session_id=session.session_id,
            player_input=opening_input,
            choice_id=None,
        )

        # Get context snapshot for scene image
        context_snapshot = story_engine.get_session_context(session.session_id) or {}
        try:
            story_ctx = getattr(session.current_state, "story_context", {}) or {}
            if isinstance(story_ctx, dict):
                context_snapshot["runtime_preset_id"] = story_ctx.get("runtime_preset_id")
        except Exception:
            pass
        context_snapshot.update({
            "world_id": effective_world_id,
            "setting": request.setting,
            "difficulty": request.difficulty,
            "persona_id": request.persona_id,
        })

        # Generate opening scene image
        scene_image = None
        scene_image_job_id = None
        if request.include_image:
            scene_image_job_id = _enqueue_scene_image_job(
                session=session,
                narrative_text=turn_result["narrative"],
                context_snapshot=context_snapshot,
                force=False,
            )
            if not scene_image_job_id:
                scene_image = await _generate_scene_image(
                    narrative_text=turn_result["narrative"],
                    context_snapshot=context_snapshot,
                    force=False,
                )

        # Check for missing character portraits and trigger generation
        try:
            present_characters = context_snapshot.get("present_characters", [])
            from core.worldpacks import get_worldpack_manager

            wpm = get_worldpack_manager()
            worldpack = wpm.get_worldpack(effective_world_id)
            if worldpack:
                for char_data in present_characters:
                    char_id = char_data.get("character_id")
                    if not char_id:
                        continue

                    # Find character in worldpack
                    matching_char = next(
                        (c for c in worldpack.characters if c.character_id == char_id),
                        None,
                    )
                    if matching_char and not matching_char.image_url:
                        appearance = (
                            matching_char.background_story
                            or f"Anime style character {matching_char.name}"
                        )
                        _enqueue_character_portrait_job(
                            character_id=char_id,
                            character_name=matching_char.name,
                            appearance_desc=appearance,
                            world_id=effective_world_id,
                            visual_style=getattr(worldpack.visual, "prompt_prefix", ""),
                        )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Character portrait trigger failed: %s", exc)

        # Persist UI-friendly extras on session

            if scene_image:
                session.current_state.story_context["last_scene_image"] = scene_image.model_dump()
            if scene_image_job_id:
                session.current_state.story_context["last_scene_image_job_id"] = scene_image_job_id
            if agent_overlay:
                session.current_state.story_context["last_agent_overlay"] = agent_overlay
            if turn_result.get("agent_actions"):
                session.current_state.story_context["last_agent_actions"] = turn_result.get("agent_actions")

            # Persist per-turn artifacts into history (Turn Inspector)
            try:
                if getattr(session, "history", None) and isinstance(session.history[-1], dict):
                    last = session.history[-1]
                    last["player_input"] = raw_opening_input
                    if raw_opening_input != enriched_opening_input:
                        last["enriched_player_input"] = enriched_opening_input

                    effective_rag_mode = (
                        request.rag_mode
                        if getattr(request, "rag_mode", None) in {"auto", "on", "off"}
                        else ("auto" if bool(rag_auto) else ("on" if bool(effective_enrich_with_rag) else "off"))
                    )
                    last["rag_mode"] = effective_rag_mode
                    last["rag_query"] = request.rag_query or request.setting
                    last["rerank_mode"] = requested_rerank_mode
                    compact_hits = _compact_knowledge_used(knowledge_used) if knowledge_used else []
                    last["knowledge_used"] = compact_hits

                    if agent_overlay:
                        last["agent_overlay"] = agent_overlay
                    last["agent_used"] = bool(agent_overlay) or bool(last.get("agent_actions"))

                    if scene_image:
                        last["scene_image"] = scene_image.model_dump()
                    if scene_image_job_id:
                        last["scene_image_job_id"] = scene_image_job_id

                    # Normalized artifacts for Turn Inspector (do not clobber engine/worker buckets)
                    try:
                        world_bucket: Dict[str, Any] = {"world_id": effective_world_id}
                        applied_worldpack_updated_at = None
                        try:
                            story_ctx = getattr(session.current_state, "story_context", {}) or {}
                            if isinstance(story_ctx, dict):
                                applied_worldpack_updated_at = story_ctx.get("worldpack_updated_at")
                        except Exception:
                            applied_worldpack_updated_at = None
                        if applied_worldpack_updated_at is not None:
                            world_bucket["applied_worldpack_updated_at"] = applied_worldpack_updated_at

                        current_worldpack_updated_at = None
                        try:
                            from core.worldpacks import get_worldpack_manager

                            wpm = get_worldpack_manager()
                            pack = wpm.get_worldpack(str(effective_world_id or "default").strip() or "default")
                            current_worldpack_updated_at = getattr(pack, "updated_at", None) if pack else None
                        except Exception:
                            current_worldpack_updated_at = None
                        if current_worldpack_updated_at is not None:
                            world_bucket["worldpack_updated_at_current"] = current_worldpack_updated_at

                        if applied_worldpack_updated_at and current_worldpack_updated_at:
                            world_bucket["synced"] = str(applied_worldpack_updated_at) == str(current_worldpack_updated_at)

                        patch_artifacts: Dict[str, Any] = {
                            "rag": {
                                "mode": effective_rag_mode,
                                "query": last.get("rag_query"),
                                "rerank_mode": requested_rerank_mode,
                                "enable_rerank": bool(effective_enable_rerank),
                                "hits": compact_hits,
                            },
                            "agents": {
                                "used": bool(last.get("agent_used")),
                                "overlay": agent_overlay,
                            },
                            "t2i": {
                                "scene_image_job_id": scene_image_job_id,
                                "scene_image": scene_image.model_dump() if scene_image else None,
                            },
                            "world": world_bucket,
                        }

                        existing_artifacts = last.get("artifacts")
                        if not isinstance(existing_artifacts, dict):
                            existing_artifacts = {}
                        last["artifacts"] = _deep_merge_dict(existing_artifacts, patch_artifacts)
                    except Exception:
                        pass
            except Exception:  # noqa: BLE001
                pass

            story_engine.save_session(session)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip persisting story extras: %s", exc)

        return StoryTurnResponse(
            session_id=session.session_id,
            world_id=session.world_id,
            turn_count=turn_result["turn_count"],
            narrative=turn_result["narrative"],
            choices=turn_result["choices"],
            stats=turn_result["stats"],
            inventory=turn_result["inventory"],
            scene_id=turn_result.get("scene_id") or session.current_state.scene_id,
            flags=turn_result.get("flags") or session.current_state.flags or {},
            agent_used=bool(agent_overlay),
            agent_overlay=agent_overlay,
            knowledge_used=knowledge_used or None,
            context=context_snapshot,
            scene_image_job_id=scene_image_job_id,
            scene_image=scene_image,
        )
    except GameError as exc:
        logger.error("Story creation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected story creation error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create story session")


@router.post("/story/turn_job", response_model=StoryTurnJobResponse)
async def enqueue_story_turn_job(request: StoryTurnRequest):
    """
    Enqueue a Story turn as a background job (GPU worker recommended).

    The worker updates the session file; frontend should poll `/jobs/{job_id}` and
    refetch `/story/session/{session_id}` once completed.
    """
    try:
        story_engine = get_story_engine()
        story_engine.get_session(request.session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unable to load session: {exc}") from exc

    # Prevent concurrent turn jobs per session (avoids clobbering session history/state).
    try:
        existing_jobs = story_turn_job_manager.list_jobs() or []
        for job in existing_jobs:
            if str(job.get("job_type") or "").strip() != "story_turn":
                continue
            status = str(job.get("status") or "").lower()
            if status in {"completed", "failed", "cancelled"}:
                continue
            payload = job.get("payload") or {}
            if str(payload.get("session_id") or "").strip() == str(request.session_id).strip():
                raise HTTPException(
                    status_code=409,
                    detail=f"Turn already in progress (job_id={job.get('job_id')})",
                )
    except HTTPException:
        raise
    except Exception:
        pass

    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    job_id = story_turn_job_manager.create_job("story_turn", payload, status="queued")

    # Prefer sync execution for local/portfolio demo. Real worker deployments can
    # set JOBS_SYNC_FALLBACK=0 to enqueue to Celery instead.
    if _run_jobs_sync_fallback():
        try:
            story_turn_job_manager.update_job(job_id, status="running", progress=1.0)
            with job_context(job_id, "story_turn"):
                resp = await process_story_turn(request)
            try:
                story_turn_job_manager.update_job(
                    job_id,
                    status="completed",
                    progress=100.0,
                    result=resp.model_dump()
                    if hasattr(resp, "model_dump")
                    else resp.dict(),
                )
            except Exception:
                pass
        except Exception as sync_exc:  # noqa: BLE001
            story_turn_job_manager.update_job(
                job_id, status="failed", progress=0.0, error=str(sync_exc)[:2000]
            )
            raise
    else:
        try:
            from workers.tasks.story import story_turn_task

            async_result = story_turn_task.delay({"job_id": job_id, "payload": payload})
            try:
                story_turn_job_manager.update_job(job_id, celery_task_id=str(async_result.id))
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            logger.info("Celery dispatch skipped (%s), running sync", exc)
            try:
                story_turn_job_manager.update_job(job_id, status="running", progress=1.0)
                with job_context(job_id, "story_turn"):
                    resp = await process_story_turn(request)
                story_turn_job_manager.update_job(
                    job_id,
                    status="completed",
                    progress=100.0,
                    result=resp.model_dump()
                    if hasattr(resp, "model_dump")
                    else resp.dict(),
                )
            except Exception as sync_exc:  # noqa: BLE001
                story_turn_job_manager.update_job(
                    job_id, status="failed", progress=0.0, error=str(sync_exc)[:2000]
                )
                raise

    return StoryTurnJobResponse(success=True, job_id=job_id, status="queued")


@router.post("/story/turn", response_model=StoryTurnResponse)
async def process_story_turn(request: StoryTurnRequest):
    """Process a story turn, optionally enriching with RAG and agent output."""
    try:
        story_engine = get_story_engine()
        _job_report("load_session", 5.0, "載入故事狀態…")
        session = story_engine.get_session(request.session_id)
        world_id = getattr(session, "world_id", "default")
        _job_report("load_session", 8.0, f"world_id={world_id}")
        world_rerank_default = _world_enable_rerank(world_id)
        # Resolve RAG preference: request overrides, otherwise use session default (or world auto if unset)
        _job_report("resolve_modes", 10.0, "解析本回合設定…")
        try:
            story_ctx = getattr(session.current_state, "story_context", {}) or {}
            rag_auto = bool(story_ctx.get("rag_auto", True))

            requested_mode = getattr(request, "rag_mode", None)
            if requested_mode in {"auto", "on", "off"}:
                rag_available = _world_has_rag_documents(world_id)
                if requested_mode == "auto":
                    rag_auto = True
                    effective_enrich_with_rag = bool(rag_available)
                elif requested_mode == "on":
                    rag_auto = False
                    effective_enrich_with_rag = True
                else:  # off
                    rag_auto = False
                    effective_enrich_with_rag = False
            else:
                if request.enrich_with_rag is None:
                    if rag_auto:
                        rag_available = _world_has_rag_documents(world_id)
                        effective_enrich_with_rag = bool(rag_available)
                    else:
                        rag_available = bool(
                            story_ctx.get("rag_available", _world_has_rag_documents(world_id))
                        )
                        effective_enrich_with_rag = bool(
                            story_ctx.get("enrich_with_rag", False)
                        )
                else:
                    rag_available = bool(
                        story_ctx.get("rag_available", _world_has_rag_documents(world_id))
                    )
                    effective_enrich_with_rag = bool(request.enrich_with_rag)
                    rag_auto = False

            session.current_state.story_context["rag_auto"] = bool(rag_auto)
            session.current_state.story_context["rag_available"] = bool(rag_available)
            session.current_state.story_context["enrich_with_rag"] = bool(effective_enrich_with_rag)
            if request.rag_query is not None:
                session.current_state.story_context["rag_query"] = request.rag_query
        except Exception:  # noqa: BLE001
            effective_enrich_with_rag = bool(request.enrich_with_rag)

        # Reranker mode (stored per-session; auto follows world default)
        effective_rerank_mode = "auto"
        effective_enable_rerank = bool(world_rerank_default)
        try:
            story_ctx = getattr(session.current_state, "story_context", {}) or {}
            if isinstance(story_ctx, dict):
                stored = story_ctx.get("rerank_mode")
                if stored in {"auto", "on", "off"}:
                    effective_rerank_mode = stored
                requested = getattr(request, "rerank_mode", None)
                if requested in {"auto", "on", "off"}:
                    effective_rerank_mode = requested
                    story_ctx["rerank_mode"] = requested
                    session.current_state.story_context = story_ctx
        except Exception:
            effective_rerank_mode = "auto"

        if effective_rerank_mode == "on":
            effective_enable_rerank = True
        elif effective_rerank_mode == "off":
            effective_enable_rerank = False
        else:
            effective_enable_rerank = bool(world_rerank_default)

        try:
            _job_report(
                "resolve_modes",
                12.0,
                f"RAG={'on' if bool(effective_enrich_with_rag) else 'off'} / rerank={effective_rerank_mode} / agents={'on' if bool(request.use_agent) else 'off'}",
            )
        except Exception:
            pass

        # RAG enrichment
        knowledge_used: List[Dict[str, Any]] = []
        raw_player_input = request.player_input
        player_input = raw_player_input
        if effective_enrich_with_rag:
            _job_report("rag_search", 20.0, "知識庫檢索中…")
            knowledge_used = _safe_rag_search(
                request.rag_query or request.player_input,
                top_k=request.top_k,
                world_id=world_id,
                enable_rerank=effective_enable_rerank,
            )
            _job_report("rag_search", 25.0, f"RAG 命中 {len(knowledge_used)} 段", hits=len(knowledge_used))
            if knowledge_used:
                snippets = [k["content"] for k in knowledge_used[:2]]
                player_input += "\n背景知識：" + " ".join(snippets)
        else:
            _job_report("rag_skip", 20.0, "本回合未啟用 RAG")

        player_input = _filter_text(player_input)

        # Agent assistance
        agent_overlay: Optional[Dict[str, Any]] = None
        if request.use_agent:
            _job_report("agent_assist", 35.0, "Agents 推理中…")
            agent_overlay = await _agent_assist(
                session,
                player_input,
                scenario_type=request.scenario_type,
                scenario_data=request.scenario_data,
            )
            _job_report("agent_assist", 45.0, "Agent overlay 已產生")
            if agent_overlay.get("narrative"):
                player_input += f"\n[agent_hint] {agent_overlay['narrative']}"
        else:
            _job_report("agent_skip", 35.0, "本回合未啟用 Agents")

        enriched_player_input = player_input

        _job_report("story_engine", 70.0, "生成敘事回應…")
        result = await story_engine.process_turn(
            session_id=request.session_id,
            player_input=player_input,
            choice_id=request.choice_id,
        )
        _job_report("story_engine", 75.0, "敘事已生成")

        _job_report("context_snapshot", 80.0, "整理世界/角色狀態…")
        context_snapshot = story_engine.get_session_context(
            request.session_id
        ) or {}
        try:
            story_ctx = getattr(session.current_state, "story_context", {}) or {}
            if isinstance(story_ctx, dict):
                context_snapshot.update(
                    {
                        "world_id": getattr(session, "world_id", "default"),
                        "setting": story_ctx.get("setting") or getattr(session, "setting", None) or "fantasy",
                        "difficulty": story_ctx.get("difficulty") or getattr(session, "difficulty", None) or "medium",
                        "persona_id": getattr(session, "persona_id", None),
                        "runtime_preset_id": story_ctx.get("runtime_preset_id"),
                    }
                )
            else:
                context_snapshot.update(
                    {
                        "world_id": getattr(session, "world_id", "default"),
                        "persona_id": getattr(session, "persona_id", None),
                    }
                )
        except Exception:
            context_snapshot.update(
                {
                    "world_id": getattr(session, "world_id", "default"),
                    "persona_id": getattr(session, "persona_id", None),
                }
            )

        # Generate scene image if triggered
        scene_image = None
        scene_image_job_id = None
        if request.include_image:
            _job_report("scene_image", 90.0, "送出場景圖片任務…")
            scene_image_job_id = _enqueue_scene_image_job(
                session=session,
                narrative_text=result["narrative"],
                context_snapshot=context_snapshot,
                force=False,
            )
            if not scene_image_job_id:
                _job_report("scene_image", 92.0, "改為同步生成場景圖片…")
                scene_image = await _generate_scene_image(
                    narrative_text=result["narrative"],
                    context_snapshot=context_snapshot,
                    force=False,
                )
            _job_report(
                "scene_image",
                95.0,
                "場景圖片已準備",
                scene_image_job_id=str(scene_image_job_id or ""),
                has_image=bool(scene_image),
            )
        else:
            _job_report("scene_image_skip", 90.0, "本回合未生成場景圖片")

        # Check for missing character portraits and trigger generation
        try:
            present_characters = context_snapshot.get("present_characters", [])
            from core.worldpacks import get_worldpack_manager

            wpm = get_worldpack_manager()
            worldpack = wpm.get_worldpack(world_id)
            if worldpack:
                for char_data in present_characters:
                    char_id = char_data.get("character_id")
                    if not char_id:
                        continue

                    # Find character in worldpack
                    matching_char = next(
                        (c for c in worldpack.characters if c.character_id == char_id),
                        None,
                    )
                    if matching_char and not matching_char.image_url:
                        # Character missing image, trigger generation
                        _job_report(
                            "character_portrait", 96.0, f"為 {matching_char.name} 生成立繪…"
                        )
                        appearance = (
                            matching_char.background_story
                            or f"Anime style character {matching_char.name}"
                        )
                        _enqueue_character_portrait_job(
                            character_id=char_id,
                            character_name=matching_char.name,
                            appearance_desc=appearance,
                            world_id=world_id,
                            visual_style=getattr(worldpack.visual, "prompt_prefix", ""),
                        )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Character portrait trigger failed: %s", exc)

        # Persist UI-friendly extras on session
        try:
            _job_report("persist", 97.0, "保存回合與 artifacts…")
            if scene_image:
                session.current_state.story_context["last_scene_image"] = scene_image.model_dump()
            if scene_image_job_id:
                session.current_state.story_context["last_scene_image_job_id"] = scene_image_job_id
            if agent_overlay:
                session.current_state.story_context["last_agent_overlay"] = agent_overlay
            if result.get("agent_actions"):
                session.current_state.story_context["last_agent_actions"] = result.get("agent_actions")

            # Persist per-turn artifacts into history (Turn Inspector)
            try:
                if getattr(session, "history", None) and isinstance(session.history[-1], dict):
                    last = session.history[-1]
                    last["player_input"] = raw_player_input
                    if raw_player_input != enriched_player_input:
                        last["enriched_player_input"] = enriched_player_input

                    effective_rag_mode = (
                        request.rag_mode
                        if getattr(request, "rag_mode", None) in {"auto", "on", "off"}
                        else ("auto" if bool(rag_auto) else ("on" if bool(effective_enrich_with_rag) else "off"))
                    )
                    last["rag_mode"] = effective_rag_mode
                    last["rag_query"] = request.rag_query or raw_player_input
                    last["rerank_mode"] = effective_rerank_mode
                    compact_hits = _compact_knowledge_used(knowledge_used) if knowledge_used else []
                    last["knowledge_used"] = compact_hits

                    if agent_overlay:
                        last["agent_overlay"] = agent_overlay
                    last["agent_used"] = bool(agent_overlay) or bool(last.get("agent_actions"))

                    if scene_image:
                        last["scene_image"] = scene_image.model_dump()
                    if scene_image_job_id:
                        last["scene_image_job_id"] = scene_image_job_id

                    # Normalized artifacts for Turn Inspector (do not clobber engine/worker buckets)
                    try:
                        world_bucket: Dict[str, Any] = {"world_id": world_id}
                        applied_worldpack_updated_at = None
                        try:
                            story_ctx = getattr(session.current_state, "story_context", {}) or {}
                            if isinstance(story_ctx, dict):
                                applied_worldpack_updated_at = story_ctx.get("worldpack_updated_at")
                        except Exception:
                            applied_worldpack_updated_at = None
                        if applied_worldpack_updated_at is not None:
                            world_bucket["applied_worldpack_updated_at"] = applied_worldpack_updated_at

                        current_worldpack_updated_at = None
                        try:
                            from core.worldpacks import get_worldpack_manager

                            wpm = get_worldpack_manager()
                            pack = wpm.get_worldpack(str(world_id or "default").strip() or "default")
                            current_worldpack_updated_at = getattr(pack, "updated_at", None) if pack else None
                        except Exception:
                            current_worldpack_updated_at = None
                        if current_worldpack_updated_at is not None:
                            world_bucket["worldpack_updated_at_current"] = current_worldpack_updated_at

                        if applied_worldpack_updated_at and current_worldpack_updated_at:
                            world_bucket["synced"] = str(applied_worldpack_updated_at) == str(current_worldpack_updated_at)

                        patch_artifacts: Dict[str, Any] = {
                            "rag": {
                                "mode": effective_rag_mode,
                                "query": last.get("rag_query"),
                                "rerank_mode": effective_rerank_mode,
                                "enable_rerank": bool(effective_enable_rerank),
                                "hits": compact_hits,
                            },
                            "agents": {
                                "used": bool(last.get("agent_used")),
                                "overlay": agent_overlay,
                            },
                            "t2i": {
                                "scene_image_job_id": scene_image_job_id,
                                "scene_image": scene_image.model_dump() if scene_image else None,
                            },
                            "world": world_bucket,
                        }
                        try:
                            ctx = get_job_context()
                            if ctx and str(ctx.job_id or "").strip():
                                job = story_turn_job_manager.get_job(
                                    str(ctx.job_id).strip(), auto_progress=False
                                ) or {}
                                stage_events = job.get("stage_events")
                                if not isinstance(stage_events, list):
                                    stage_events = []
                                compact_events: List[Dict[str, Any]] = []
                                for e in stage_events[-30:]:
                                    if not isinstance(e, dict):
                                        continue
                                    compact_events.append(
                                        {
                                            "ts": e.get("ts"),
                                            "stage": e.get("stage"),
                                            "progress": e.get("progress"),
                                            "message": e.get("message"),
                                            "meta": e.get("meta"),
                                        }
                                    )

                                patch_artifacts["job"] = {
                                    "job_id": str(ctx.job_id).strip(),
                                    "job_type": "story_turn",
                                    "stage": job.get("stage"),
                                    "stage_message": job.get("stage_message"),
                                    "progress": job.get("progress"),
                                    "started_at": job.get("started_at") or job.get("created_at"),
                                    "duration_seconds": job.get("duration_seconds"),
                                    "stage_events": compact_events,
                                }
                        except Exception:
                            pass

                        existing_artifacts = last.get("artifacts")
                        if not isinstance(existing_artifacts, dict):
                            existing_artifacts = {}
                        last["artifacts"] = _deep_merge_dict(existing_artifacts, patch_artifacts)
                    except Exception:
                        pass
            except Exception:  # noqa: BLE001
                pass

            story_engine.save_session(session)
            _job_report("done", 99.0, "完成")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip persisting story extras: %s", exc)

        return StoryTurnResponse(
            session_id=result["session_id"],
            world_id=getattr(session, "world_id", "default"),
            turn_count=result["turn_count"],
            narrative=result["narrative"],
            choices=result["choices"],
            stats=result["stats"],
            inventory=result["inventory"],
            scene_id=result.get("scene_id") or session.current_state.scene_id,
            flags=result.get("flags") or session.current_state.flags or {},
            agent_used=bool(agent_overlay) or bool(result.get("agent_actions")),
            agent_overlay=agent_overlay,
            agent_actions=result.get("agent_actions"),
            knowledge_used=knowledge_used or None,
            context=context_snapshot,
            scene_image_job_id=scene_image_job_id,
            scene_image=scene_image,
        )
    except (SessionNotFoundError, InvalidChoiceError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except GameError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Story turn failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process story turn")


@router.post("/story/turn/stream")
async def process_story_turn_stream(request: StoryTurnRequest):
    """
    Process a story turn with SSE streaming for narrative text.

    Returns an EventSource stream where:
      - `data: {"type": "token", "content": "..."}` — narrative text tokens
      - `data: {"type": "done", "result": {...}}`     — final complete result
    """
    try:
        story_engine = get_story_engine()
        story_engine.get_session(request.session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to load session: {exc}")

    async def event_generator():
        try:
            async for chunk in story_engine.process_turn_stream(
                session_id=request.session_id,
                player_input=request.player_input,
                choice_id=request.choice_id,
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("Stream generation error: %s", exc, exc_info=True)
            error_payload = {"type": "done", "result": {"error": str(exc), "narrative": "", "choices": []}}
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/story/session/{session_id}/save")
async def save_game(session_id: str, slot_name: str = Body(..., embed=True)):
    """Create a save slot for the current session."""
    try:
        story_engine = get_story_engine()
        slot_info = story_engine.save_game_slot(session_id, slot_name)
        return slot_info
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to save game: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/story/session/{session_id}/saves")
async def list_saves(session_id: str):
    """List all save slots for a session."""
    try:
        story_engine = get_story_engine()
        return story_engine.list_game_slots(session_id)
    except Exception as exc:
        logger.error(f"Failed to list saves: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/story/session/{session_id}/load/{slot_id}")
async def load_game(session_id: str, slot_id: str):
    """Load a specific save slot into the session."""
    try:
        story_engine = get_story_engine()
        session = story_engine.load_game_slot(session_id, slot_id)
        return {"success": True, "session_id": session.session_id, "turn_count": session.turn_count}
    except Exception as exc:
        logger.error(f"Failed to load game: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/story/session/{session_id}/flowchart")
async def get_flowchart(session_id: str):
    """Get the narrative flowchart data."""
    try:
        story_engine = get_story_engine()
        return story_engine.get_narrative_flowchart(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to get flowchart: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/story/sessions", response_model=List[StorySessionInfo])
async def list_story_sessions(active_only: bool = True):
    """List story sessions."""
    story_engine = get_story_engine()
    sessions = story_engine.list_sessions(active_only=active_only)
    results: List[StorySessionInfo] = []
    for session in sessions:
        results.append(
            StorySessionInfo(
                session_id=session["session_id"],
                player_name=session["player_name"],
                persona_id=session.get("persona_id"),
                world_id=session.get("world_id", "default"),
                turn_count=session["turn_count"],
                is_active=session["is_active"],
                updated_at=session["updated_at"],
                enhanced_mode=session.get("enhanced_mode", False),
                current_scene=session.get("current_scene"),
            )
        )
    return results


@router.get("/story/session/{session_id}", response_model=StorySessionDetail)
async def get_story_session(session_id: str):
    """Get story session detail with current game state."""
    try:
        story_engine = get_story_engine()
        session = story_engine.get_session(session_id)

        narrative = (
            (session.current_state.scene_description or "").strip()
            if session.current_state
            else ""
        )
        if not narrative and getattr(session, "history", None):
            last = session.history[-1] or {}
            narrative = (
                str(last.get("ai_response") or last.get("response") or "").strip()
            )

        raw_choices = (
            getattr(session.current_state, "available_choices", None) if session.current_state else None
        ) or []
        choices: List[Dict[str, Any]] = []
        for c in raw_choices:
            if not isinstance(c, dict):
                continue
            choice_id = c.get("choice_id") or c.get("id")
            text = c.get("text") or c.get("label")
            if not choice_id or not text:
                continue
            choices.append({**c, "choice_id": choice_id, "text": text})

        # GUARD: Ensure choices are never empty at API level
        if not choices:
            logger.warning(
                "Session %s has no available_choices — injecting emergency fallback",
                session.session_id,
            )
            choices = [
                {"choice_id": "continue_forward", "text": "繼續前進", "type": "action", "difficulty": "easy"},
                {"choice_id": "look_around", "text": "仔細觀察周圍環境", "type": "exploration", "difficulty": "easy"},
                {"choice_id": "check_status", "text": "檢查自身狀態和裝備", "type": "action", "difficulty": "easy"},
                {"choice_id": "wait_observe", "text": "等待並觀察情況發展", "type": "action", "difficulty": "easy"},
            ]

        # Optional: last generated scene image / agent actions stored on session context
        scene_image: Optional[SceneImage] = None
        agent_actions: Optional[Dict[str, Any]] = None
        player_template_id: Optional[str] = None
        scene_image_job_id: Optional[str] = None
        turn_job_id: Optional[str] = None
        worldpack_updated_at: Optional[str] = None
        runtime_preset_id: Optional[str] = None
        try:
            story_ctx = getattr(session.current_state, "story_context", {}) or {}
            if isinstance(story_ctx, dict):
                player_template_id = story_ctx.get("player_template_id")
                worldpack_updated_at = story_ctx.get("worldpack_updated_at")
                runtime_preset_id = story_ctx.get("runtime_preset_id")
                scene_image_job_id = story_ctx.get("last_scene_image_job_id")
                payload = story_ctx.get("last_scene_image")
                if isinstance(payload, dict) and payload.get("image_url"):
                    try:
                        scene_image = SceneImage(**payload)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Invalid scene_image payload: %s", exc)

                agent_payload = story_ctx.get("last_agent_actions")
                if isinstance(agent_payload, dict):
                    agent_actions = agent_payload
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip loading story extras: %s", exc)

        # Active Story turn job id (so UI can resume polling after refresh)
        try:
            jobs = story_turn_job_manager.list_jobs() or []
            candidates = []
            for job in jobs:
                if str(job.get("job_type") or "").strip() != "story_turn":
                    continue
                status = str(job.get("status") or "").lower()
                if status in {"completed", "failed", "cancelled"}:
                    continue
                payload = job.get("payload") or {}
                if str(payload.get("session_id") or "").strip() != str(session_id).strip():
                    continue
                candidates.append(job)
            if candidates:
                # Prefer newest by created_at/updated_at (ISO string sortable)
                candidates.sort(key=lambda j: str(j.get("updated_at") or j.get("created_at") or ""))
                turn_job_id = str(candidates[-1].get("job_id") or "").strip() or None
        except Exception:
            turn_job_id = None

        # Memory stats/context (best-effort)
        memory_stats = None
        memory_context = None
        try:
            from core.story.memory_manager import get_memory_manager

            memory_manager = get_memory_manager(session_id)
            memory_stats = memory_manager.get_statistics()
            memory_context = await memory_manager.retrieve_relevant_context(
                query=narrative or session_id,
                max_results=5,
                include_short_term=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip story memory snapshot: %s", exc)

        # Persisted turn history for UI timeline (best-effort)
        turn_history: List[StoryTurnHistoryEntry] = []
        try:
            raw_history = list(getattr(session, "history", []) or [])
            for item in raw_history[-20:]:
                if not isinstance(item, dict):
                    continue
                # Relaxed: require at least one of player_input / ai_response
                if "player_input" not in item and "ai_response" not in item:
                    continue
                player_input = str(item.get("player_input") or "").strip()
                ai_response = str(item.get("ai_response") or item.get("response") or "").strip()
                if not player_input and not ai_response:
                    continue

                # Normalized artifacts (Turn Inspector): prefer persisted artifacts, otherwise synthesize from legacy keys.
                artifacts: Optional[Dict[str, Any]] = None
                try:
                    raw_artifacts = item.get("artifacts")
                    artifacts = raw_artifacts if isinstance(raw_artifacts, dict) else None
                except Exception:
                    artifacts = None

                if artifacts is None:
                    try:
                        synthesized: Dict[str, Any] = {}

                        # RAG
                        hits = item.get("knowledge_used") if isinstance(item.get("knowledge_used"), list) else []
                        if item.get("rag_mode") or item.get("rag_query") or item.get("rerank_mode") or hits:
                            synthesized["rag"] = {
                                "mode": item.get("rag_mode"),
                                "query": item.get("rag_query"),
                                "rerank_mode": item.get("rerank_mode"),
                                "hits": hits,
                            }

                        # Agents
                        if (
                            item.get("agent_used") is not None
                            or item.get("agent_overlay") is not None
                            or item.get("agent_actions") is not None
                        ):
                            synthesized["agents"] = {
                                "used": item.get("agent_used"),
                                "overlay": item.get("agent_overlay"),
                                "actions": item.get("agent_actions"),
                            }

                        # Diff
                        if isinstance(item.get("state_delta"), dict):
                            synthesized["diff"] = item.get("state_delta")

                        # T2I
                        if item.get("scene_image_job_id") or item.get("scene_image"):
                            synthesized["t2i"] = {
                                "scene_image_job_id": item.get("scene_image_job_id"),
                                "scene_image": item.get("scene_image"),
                            }

                        # World sync snapshot (best-effort; may reflect current worldpack rather than historical)
                        try:
                            world_bucket: Dict[str, Any] = {"world_id": getattr(session, "world_id", "default")}
                            story_ctx = getattr(session.current_state, "story_context", {}) or {}
                            applied_at = story_ctx.get("worldpack_updated_at") if isinstance(story_ctx, dict) else None
                            if applied_at is not None:
                                world_bucket["applied_worldpack_updated_at"] = applied_at
                            current_at = None
                            try:
                                from core.worldpacks import get_worldpack_manager

                                wpm = get_worldpack_manager()
                                pack = wpm.get_worldpack(str(world_bucket["world_id"] or "default").strip() or "default")
                                current_at = getattr(pack, "updated_at", None) if pack else None
                            except Exception:
                                current_at = None
                            if current_at is not None:
                                world_bucket["worldpack_updated_at_current"] = current_at
                            if applied_at and current_at:
                                world_bucket["synced"] = str(applied_at) == str(current_at)
                            synthesized["world"] = world_bucket
                        except Exception:
                            pass

                        artifacts = synthesized
                    except Exception:
                        artifacts = None

                try:
                    turn_history.append(
                        StoryTurnHistoryEntry(
                            turn=int(item.get("turn", 0) or 0),
                            timestamp=item.get("timestamp"),
                            player_input=player_input,
                            ai_response=ai_response,
                            choice_id=item.get("choice_id"),
                            scene_id=item.get("scene_id"),
                            agent_used=item.get("agent_used"),
                            enriched_player_input=item.get("enriched_player_input"),
                            rag_mode=item.get("rag_mode"),
                            rag_query=item.get("rag_query"),
                            rerank_mode=item.get("rerank_mode"),
                            knowledge_used=item.get("knowledge_used"),
                            agent_overlay=item.get("agent_overlay"),
                            agent_actions=item.get("agent_actions"),
                            state_delta=item.get("state_delta"),
                            scene_image_job_id=item.get("scene_image_job_id"),
                            scene_image=item.get("scene_image"),
                            artifacts=artifacts,
                        )
                    )
                except Exception:
                    continue
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip turn_history snapshot: %s", exc)

        # RAG status (world-aware, recomputed for UI)
        rag_auto = None
        rag_mode = None
        rag_available = None
        enrich_with_rag = None
        rag_next_turn = None
        rag_query = None
        rerank_mode = None
        rerank_next_turn = None
        try:
            story_ctx = getattr(session.current_state, "story_context", {}) or {}
            if isinstance(story_ctx, dict):
                rag_query = story_ctx.get("rag_query")
                rag_auto_val = story_ctx.get("rag_auto")
                rag_auto = rag_auto_val if rag_auto_val is not None else True
                rag_available = _world_has_rag_documents(getattr(session, "world_id", "default"))
                if bool(rag_auto):
                    rag_mode = "auto"
                    rag_next_turn = bool(rag_available)
                    enrich_with_rag = bool(rag_next_turn)
                else:
                    enrich_with_rag = bool(story_ctx.get("enrich_with_rag", False))
                    rag_mode = "on" if bool(enrich_with_rag) else "off"
                    rag_next_turn = bool(enrich_with_rag)

                stored_rerank = story_ctx.get("rerank_mode")
                effective_rerank_mode = (
                    stored_rerank
                    if stored_rerank in {"auto", "on", "off"}
                    else "auto"
                )
                rerank_mode = effective_rerank_mode
                world_default_rerank = _world_enable_rerank(
                    getattr(session, "world_id", "default")
                )
                if effective_rerank_mode == "on":
                    rerank_next_turn = True
                elif effective_rerank_mode == "off":
                    rerank_next_turn = False
                else:
                    rerank_next_turn = bool(world_default_rerank)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip RAG status snapshot: %s", exc)

        return StorySessionDetail(
            session_id=session.session_id,
            player_name=session.player_name,
            persona_id=session.persona_id,
            world_id=getattr(session, "world_id", "default"),
            runtime_preset_id=runtime_preset_id,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            turn_count=session.turn_count,
            is_active=session.is_active,
            current_scene=session.current_state.scene_id,
            player_template_id=player_template_id,
            worldpack_updated_at=worldpack_updated_at,
            stats=session.stats.to_dict(),
            inventory=session.inventory,
            flags=session.current_state.flags,
            narrative=narrative,
            choices=choices,
            turn_job_id=turn_job_id,
            scene_image_job_id=scene_image_job_id,
            scene_image=scene_image,
            memory_stats=memory_stats,
            memory_context=memory_context,
            turn_history=turn_history,
            agent_actions=agent_actions,
            rag_auto=rag_auto,
            rag_mode=rag_mode,
            rag_available=rag_available,
            enrich_with_rag=enrich_with_rag,
            rag_next_turn=rag_next_turn,
            rag_query=rag_query,
            rerank_mode=rerank_mode,
            rerank_next_turn=rerank_next_turn,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to get story session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Unable to load story session")


@router.get("/story/session/{session_id}/context", response_model=StoryContextSnapshot)
async def get_story_context(session_id: str):
    """Get enhanced context snapshot for a story session."""
    story_engine = get_story_engine()
    context = story_engine.get_session_context(session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Context not available")
    return StoryContextSnapshot(**context)


@router.get(
    "/story/session/{session_id}/agent_profile",
    response_model=StoryAgentProfileResponse,
)
async def get_story_agent_profile(session_id: str):
    """Get the session-level agent_profile used by Story Orchestrator (does not read worldpack)."""
    try:
        story_engine = get_story_engine()
        session = story_engine.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        from schemas.world import WorldAgentProfile

        raw = _load_session_agent_profile(session)
        agent_profile = WorldAgentProfile(**raw) if isinstance(raw, dict) else WorldAgentProfile()
    except Exception:  # noqa: BLE001
        from schemas.world import WorldAgentProfile

        agent_profile = WorldAgentProfile()

    return StoryAgentProfileResponse(
        session_id=session_id,
        world_id=str(getattr(session, "world_id", "default") or "default"),
        agent_profile=agent_profile,
    )


@router.put(
    "/story/session/{session_id}/agent_profile",
    response_model=StoryAgentProfileResponse,
)
async def set_story_agent_profile(session_id: str, request: StoryAgentProfileUpdateRequest):
    """Replace the session-level agent_profile (takes effect next turn)."""
    try:
        story_engine = get_story_engine()
        session = story_engine.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        payload = request.agent_profile.model_dump()
        story_ctx = getattr(getattr(session, "current_state", None), "story_context", {}) or {}
        if not isinstance(story_ctx, dict):
            story_ctx = {}
        story_ctx["agent_profile"] = payload
        session.current_state.story_context = story_ctx
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid agent_profile: {exc}")

    story_engine.save_session(session)
    return await get_story_agent_profile(session_id)


@router.patch(
    "/story/session/{session_id}/agent_profile",
    response_model=StoryAgentProfileResponse,
)
async def patch_story_agent_profile(session_id: str, request: StoryAgentProfilePatchRequest):
    """Patch the session-level agent_profile (takes effect next turn)."""
    try:
        story_engine = get_story_engine()
        session = story_engine.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        from schemas.world import WorldAgentProfile

        current = WorldAgentProfile(**_load_session_agent_profile(session))
        base = current.model_dump()
        patch = request.model_dump(exclude_none=True)
        for k, v in patch.items():
            base[k] = v
        next_profile = WorldAgentProfile(**base)

        story_ctx = getattr(getattr(session, "current_state", None), "story_context", {}) or {}
        if not isinstance(story_ctx, dict):
            story_ctx = {}
        story_ctx["agent_profile"] = next_profile.model_dump()
        session.current_state.story_context = story_ctx
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid agent_profile patch: {exc}")

    story_engine.save_session(session)
    return await get_story_agent_profile(session_id)


@router.post(
    "/story/session/{session_id}/sync_worldpack",
    response_model=StoryWorldSyncResponse,
)
async def sync_story_worldpack(session_id: str, request: StoryWorldSyncRequest):
    """Apply the latest WorldPack changes to an ongoing story session (enhanced mode)."""
    try:
        story_engine = get_story_engine()
        payload = story_engine.sync_worldpack_into_session(
            session_id=session_id,
            mode=request.mode,
        )
        return StoryWorldSyncResponse(**payload)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except GameError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to sync worldpack for %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to sync worldpack")


@router.post(
    "/story/session/{session_id}/world/writeback/suggest",
    response_model=StoryWorldWritebackSuggestResponse,
)
async def suggest_world_writeback(session_id: str, request: StoryWorldWritebackSuggestRequest):
    """
    Suggest a WorldPack patch based on what's emerged in the current story session.

    注意：此 endpoint 只產生「預覽」與 patch，不會直接寫回 worldpack / RAG；
    需要前端明確確認後，才保存 worldpack 或寫入知識庫。
    """
    try:
        story_engine = get_story_engine()
        session = story_engine.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    world_id = str(getattr(session, "world_id", "default") or "default").strip() or "default"

    try:
        from core.worldpacks import get_worldpack_manager

        wpm = get_worldpack_manager()
        pack = wpm.get_worldpack(world_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load worldpack %s: %s", world_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load worldpack")

    if pack is None:
        raise HTTPException(status_code=404, detail=f"World not found: {world_id}")

    errors: List[str] = []
    patch: Dict[str, Any] = {}

    # Best-effort access to enhanced context memory (for richer character export)
    context_memory = None
    try:
        if getattr(story_engine, "enhanced_mode", False) and hasattr(story_engine, "context_memories"):
            context_memory = getattr(story_engine, "context_memories", {}).get(session_id)
    except Exception:
        context_memory = None

    # ---- Flags (quests / npc / location / items / events / achievements) --
    flags_added: Dict[str, bool] = {}
    if bool(request.include_flags):
        try:
            base_flags = dict(getattr(pack, "world_flags", {}) or {})
            combined: Dict[str, Any] = {}

            # Session flags (runtime)
            try:
                combined.update(dict(getattr(session.current_state, "flags", {}) or {}))
            except Exception:
                pass

            # Enhanced memory flags (best-effort)
            try:
                if context_memory is not None:
                    combined.update(dict(getattr(context_memory, "world_flags", {}) or {}))
            except Exception:
                pass

            allowed_prefixes = (
                "quest_",
                "npc_met_",
                "location_discovered_",
                "item_acquired_",
                "event_",
                "achievement_",
            )
            for k, v in combined.items():
                key = str(k or "").strip()
                if not key or not key.startswith(allowed_prefixes):
                    continue
                if bool(v) is not True:
                    continue
                if bool(base_flags.get(key, False)) is True:
                    continue
                flags_added[key] = True

            if flags_added:
                patch["world_flags"] = flags_added
        except Exception as exc:  # noqa: BLE001
            errors.append(f"flags 匯出失敗：{exc}")

    # ---- Characters (export new ones only; do not overwrite existing) -----
    new_characters: List[Dict[str, Any]] = []
    if bool(request.include_characters) and context_memory is not None:
        try:
            existing_ids = {c.character_id for c in (getattr(pack, "characters", []) or [])}
            max_new = int(getattr(request, "max_new_characters", 10) or 0)
            max_new = max(0, min(50, max_new))

            characters = getattr(context_memory, "characters", {}) or {}
            for char in characters.values():
                char_id = str(getattr(char, "character_id", "") or "").strip()
                if not char_id or char_id in {"player", "narrator"}:
                    continue
                if char_id in existing_ids:
                    continue

                role = getattr(getattr(char, "role", None), "value", None) or getattr(char, "role", None)
                role_str = str(role or "").strip()
                if role_str not in {"npc", "companion", "antagonist"}:
                    continue

                new_characters.append(
                    {
                        "character_id": char_id,
                        "name": str(getattr(char, "name", "") or char_id),
                        "role": role_str,
                        "personality_traits": list(getattr(char, "personality_traits", []) or []),
                        "speaking_style": str(getattr(char, "speaking_style", "") or ""),
                        "background_story": str(getattr(char, "background_story", "") or ""),
                        "motivations": list(getattr(char, "motivations", []) or []),
                        "relationships": dict(getattr(char, "relationships", {}) or {}),
                        "persona_prompt": str(getattr(char, "persona_prompt", "") or ""),
                        "content_restrictions": list(getattr(char, "content_restrictions", []) or []),
                        "start_in_opening": False,
                    }
                )

                if max_new and len(new_characters) >= max_new:
                    break

            if new_characters:
                patch["characters"] = new_characters
        except Exception as exc:  # noqa: BLE001
            errors.append(f"角色匯出失敗：{exc}")

    # ---- Build candidate worldpack preview --------------------------------
    candidate = pack
    try:
        if patch:
            from schemas.world import WorldPack

            base_dict = pack.model_dump()
            if isinstance(patch.get("world_flags"), dict):
                merged_flags = dict(base_dict.get("world_flags") or {})
                for fk, fv in patch["world_flags"].items():
                    kk = str(fk or "").strip()
                    if kk:
                        merged_flags[kk] = bool(fv)
                base_dict["world_flags"] = merged_flags

            if isinstance(patch.get("characters"), list):
                existing = {
                    str(c.get("character_id")): dict(c)
                    for c in (base_dict.get("characters") or [])
                    if isinstance(c, dict) and str(c.get("character_id") or "").strip()
                }
                for c in patch["characters"]:
                    if not isinstance(c, dict):
                        continue
                    cid = str(c.get("character_id") or "").strip()
                    if not cid:
                        continue
                    if cid in existing:
                        continue
                    existing[cid] = dict(c)
                base_dict["characters"] = list(existing.values())

            candidate = WorldPack(**base_dict)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"預覽 worldpack 產生失敗：{exc}")
        candidate = pack

    # ---- Build optional RAG note ------------------------------------------
    rag_note: Optional[str] = None
    if bool(request.include_rag_note):
        try:
            def _sorted_keys(prefix: str) -> List[str]:
                keys = [k for k in flags_added.keys() if str(k).startswith(prefix)]
                return sorted(keys)

            lines: List[str] = []
            lines.append(f"【世界回寫摘要】world_id={world_id} / session_id={session_id}")
            if new_characters:
                lines.append("")
                lines.append("新增角色（建議寫回 worldpack.characters）：")
                for c in new_characters[:20]:
                    lines.append(f"- {c.get('name')} ({c.get('character_id')}) / role={c.get('role')}")

            locs = _sorted_keys("location_discovered_")
            if locs:
                lines.append("")
                lines.append("已探索/發現地點：")
                for k in locs[:30]:
                    lines.append(f"- {k.replace('location_discovered_', '')}")

            quests = _sorted_keys("quest_")
            if quests:
                lines.append("")
                lines.append("任務旗標：")
                for k in quests[:40]:
                    lines.append(f"- {k}")

            items = _sorted_keys("item_acquired_")
            if items:
                lines.append("")
                lines.append("取得物品：")
                for k in items[:40]:
                    lines.append(f"- {k.replace('item_acquired_', '')}")

            events = _sorted_keys("event_")
            if events:
                lines.append("")
                lines.append("事件：")
                for k in events[:40]:
                    lines.append(f"- {k}")

            achievements = _sorted_keys("achievement_")
            if achievements:
                lines.append("")
                lines.append("成就：")
                for k in achievements[:40]:
                    lines.append(f"- {k}")

            rag_note = "\n".join(lines).strip() or None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"RAG 摘要生成失敗：{exc}")

    return StoryWorldWritebackSuggestResponse(
        success=(len(errors) == 0),
        session_id=session_id,
        world_id=world_id,
        patch=patch,
        worldpack=candidate,
        rag_note=rag_note,
        summary={
            "flags_added": len(flags_added),
            "characters_added": len(new_characters),
            "has_rag_note": bool(rag_note),
        },
        errors=errors,
    )


@router.post(
    "/story/session/{session_id}/choice/preview", response_model=StoryChoicePreview
)
async def preview_choice(session_id: str, choice_id: str):
    """Preview choice consequences."""
    try:
        story_engine = get_story_engine()
        preview = story_engine.get_choice_preview(choice_id, session_id)
        if "error" in preview:
            raise HTTPException(status_code=404, detail=preview["error"])
        return StoryChoicePreview(**preview)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Choice preview failed: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to preview choice")


@router.get("/story/session/{session_id}/export", response_model=StoryExportResponse)
async def export_story_session(session_id: str, include_context: bool = True):
    """Export story session data."""
    story_engine = get_story_engine()
    exported = story_engine.export_session_data(
        session_id=session_id, include_context=include_context
    )
    if "error" in exported:
        raise HTTPException(status_code=404, detail=exported["error"])
    return StoryExportResponse(session_id=session_id, exported=exported)


@router.post("/story/import", response_model=StoryImportResponse)
async def import_story_session(request: StoryImportRequest):
    """Import story session data."""
    story_engine = get_story_engine()
    try:
        success = story_engine.import_session_data(request.session_data)
        session_id = request.session_data.get("session_data", {}).get("session_id")
        return StoryImportResponse(success=success, session_id=session_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Story import failed: %s", exc)
        return StoryImportResponse(success=False, error=str(exc))


@router.get("/story/metrics", response_model=StoryMetricsResponse)
async def story_metrics():
    """Return story system metrics plus integration readiness."""
    story_engine = get_story_engine()
    metrics = story_engine.get_system_performance_metrics()

    llm_ready = bool(
        getattr(story_engine.narrative_generator, "llm", None)  # type: ignore[attr-defined]
    )
    agents_ready = True
    rag_ready = False

    return StoryMetricsResponse(
        **metrics,
        agents_ready=agents_ready,
        rag_ready=rag_ready,
        llm_ready=llm_ready,
    )


@router.post("/story/agent/assist", response_model=StoryAgentActionResponse)
async def story_agent_assist(request: StoryAgentActionRequest):
    """Provide agent-assisted narrative without mutating session state."""
    try:
        story_engine = get_story_engine()
        session = story_engine.get_session(request.session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    overlay = await _agent_assist(
        session,
        request.player_input,
        scenario_type=request.scenario_type,
        scenario_data=request.scenario_data,
    )

    if overlay.get("success"):
        return StoryAgentActionResponse(
            success=True,
            narrative=overlay.get("narrative", ""),
            available_actions=overlay.get("available_actions", [])
            or ["continue", "wait", "look around"],
            consequences=overlay.get("consequences", []) or [],
            agent_steps=overlay.get("agent_steps"),
            tools_used=overlay.get("tools_used"),
            fallback_used=overlay.get("fallback_used", False),
        )
    return StoryAgentActionResponse(
        success=False,
        narrative=overlay.get("narrative", "Agent assist unavailable."),
        available_actions=overlay.get("available_actions", []) or [],
        consequences=overlay.get("consequences", []) or [],
        fallback_used=overlay.get("fallback_used", False),
    )


@router.post(
    "/story/knowledge/search", response_model=StoryKnowledgeSearchResponse
)
async def story_knowledge_search(request: StoryKnowledgeSearchRequest):
    """Expose RAG search for story support."""
    results = _safe_rag_search(request.query, top_k=request.top_k)
    available = bool(results)
    return StoryKnowledgeSearchResponse(results=results, available=available)


@router.get("/story/personas")
async def list_story_personas():
    """List available story personas."""
    try:
        story_engine = get_story_engine()
        personas = story_engine.list_personas()
        return [
            GamePersonaInfo(
                persona_id=p["persona_id"],
                name=p["name"],
                description=p["description"],
                personality_traits=p.get("personality_traits", []),
                special_abilities=p.get("special_abilities", []),
            )
            for p in personas
        ]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Failed to list personas: {str(exc)}")


@router.get("/story/templates")
async def list_story_templates():
    """List available story templates."""
    try:
        return {
            "templates": [
                {
                    "id": "hero_journey",
                    "name": "英雄旅程",
                    "description": "經典的英雄成長故事結構",
                    "themes": ["成長", "冒險", "友誼"],
                    "difficulty": "normal",
                },
                {
                    "id": "mystery_solve",
                    "name": "推理解謎",
                    "description": "偵探推理類故事模板",
                    "themes": ["推理", "懸疑", "真相"],
                    "difficulty": "hard",
                },
            ]
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Failed to list templates: {str(exc)}")


@router.get("/story/worlds")
async def list_world_settings():
    """List available world settings (legacy alias for WorldPacks)."""
    try:
        from core.worldpacks import get_worldpack_manager

        manager = get_worldpack_manager()
        return {"worlds": [item.model_dump() for item in manager.list_worldpacks()]}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Failed to list worlds: {str(exc)}")


@router.get("/story/worlds/{world_id}")
async def get_world_detail(world_id: str):
    """Get detailed world setting info (legacy alias for WorldPacks)."""
    try:
        from core.worldpacks import get_worldpack_manager

        world = get_worldpack_manager().get_worldpack(world_id)
        if world is None:
            raise HTTPException(404, f"World not found: {world_id}")
        return world.model_dump()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Failed to get world: {str(exc)}")


@router.get("/story/classes")
async def list_character_classes():
    """List all available character classes."""
    try:
        from core.story.character_classes import list_classes
        return {"classes": list_classes()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Failed to list classes: {str(exc)}")


@router.get("/story/classes/{class_id}")
async def get_class_detail(class_id: str):
    """Get detailed class info."""
    try:
        from core.story.character_classes import get_class
        cls = get_class(class_id)
        if not cls:
            raise HTTPException(404, f"Class not found: {class_id}")
        return {
            "id": cls.id,
            "name": cls.name,
            "description": cls.description,
            "base_stats": cls.base_stats,
            "skills": cls.skills,
            "advancements": cls.advancements,
            "passive_abilities": cls.passive_abilities,
            "recommended_weapon_types": cls.recommended_weapon_types,
            "lore": cls.lore,
            "synergy_with": cls.synergy_with,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Failed to get class: {str(exc)}")
