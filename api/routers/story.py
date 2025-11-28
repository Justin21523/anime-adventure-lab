# api/routers/story.py
"""
Story Engine Router (separate from game.py)
Provides story-specific APIs with LLM/agent/RAG integration.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from core.story.engine import get_story_engine
from core.agents import (
    StoryContext as AgentStoryContext,
    get_story_agent_manager,
)
from core.exceptions import GameError, SessionNotFoundError, InvalidChoiceError
from core.safety import get_content_filter
from schemas.game import GamePersonaInfo
from schemas.story import (
    StoryAgentActionRequest,
    StoryAgentActionResponse,
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
    StoryTurnResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Helpers ---------------------------------------------------------------------
def _safe_rag_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Try to retrieve RAG context; degrade gracefully offline."""
    if not query:
        return []
    try:
        from core.rag import get_rag_engine

        rag_engine = get_rag_engine()
        results = rag_engine.search(query, top_k=top_k)
        simplified = []
        for res in results:
            simplified.append(
                {
                    "content": res.document.content,
                    "score": float(getattr(res, "score", 0.0)),
                    "metadata": res.document.metadata,
                }
            )
        return simplified
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG search skipped: %s", exc)
        return []


def _filter_text(text: str) -> str:
    """Run safety filter to avoid unsafe prompts."""
    try:
        cf = get_content_filter()
        result = cf.check_text_safety(text)
        return result.get("filtered_text", text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Safety filter fallback: %s", exc)
        return text


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
        )

        # Optional RAG context
        knowledge_used: List[Dict[str, Any]] = []
        opening_input = f"開始故事，玩家 {request.player_name}。"
        if request.enrich_with_rag:
            knowledge_used = _safe_rag_search(
                request.rag_query or request.setting, top_k=3
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

        turn_result = await story_engine.process_turn(
            session_id=session.session_id,
            player_input=opening_input,
            choice_id=None,
        )

        return StoryTurnResponse(
            session_id=session.session_id,
            turn_count=turn_result["turn_count"],
            narrative=turn_result["narrative"],
            choices=turn_result["choices"],
            stats=turn_result["stats"],
            inventory=turn_result["inventory"],
            scene_id=turn_result.get("scene_id"),
            flags=turn_result.get("flags", {}),
            agent_used=bool(agent_overlay),
            agent_overlay=agent_overlay,
            knowledge_used=knowledge_used or None,
            context={
                "setting": request.setting,
                "difficulty": request.difficulty,
                "persona_id": request.persona_id,
            },
        )
    except GameError as exc:
        logger.error("Story creation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected story creation error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create story session")


@router.post("/story/turn", response_model=StoryTurnResponse)
async def process_story_turn(request: StoryTurnRequest):
    """Process a story turn, optionally enriching with RAG and agent output."""
    try:
        story_engine = get_story_engine()
        session = story_engine.get_session(request.session_id)

        # RAG enrichment
        knowledge_used: List[Dict[str, Any]] = []
        player_input = request.player_input
        if request.enrich_with_rag:
            knowledge_used = _safe_rag_search(
                request.rag_query or request.player_input, top_k=request.top_k
            )
            if knowledge_used:
                snippets = [k["content"] for k in knowledge_used[:2]]
                player_input += "\n背景知識：" + " ".join(snippets)

        player_input = _filter_text(player_input)

        # Agent assistance
        agent_overlay: Optional[Dict[str, Any]] = None
        if request.use_agent:
            agent_overlay = await _agent_assist(
                session,
                player_input,
                scenario_type=request.scenario_type,
                scenario_data=request.scenario_data,
            )
            if agent_overlay.get("narrative"):
                player_input += f"\n[agent_hint] {agent_overlay['narrative']}"

        result = await story_engine.process_turn(
            session_id=request.session_id,
            player_input=player_input,
            choice_id=request.choice_id,
        )

        context_snapshot = story_engine.get_session_context(
            request.session_id
        ) or {}

        return StoryTurnResponse(
            session_id=result["session_id"],
            turn_count=result["turn_count"],
            narrative=result["narrative"],
            choices=result["choices"],
            stats=result["stats"],
            inventory=result["inventory"],
            scene_id=result.get("scene_id"),
            flags=result.get("flags", {}),
            agent_used=bool(agent_overlay),
            agent_overlay=agent_overlay,
            knowledge_used=knowledge_used or None,
            context=context_snapshot,
        )
    except (SessionNotFoundError, InvalidChoiceError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except GameError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Story turn failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process story turn")


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
    """Get story session detail."""
    try:
        story_engine = get_story_engine()
        session = story_engine.get_session(session_id)
        return StorySessionDetail(
            session_id=session.session_id,
            player_name=session.player_name,
            persona_id=session.persona_id,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            turn_count=session.turn_count,
            is_active=session.is_active,
            current_scene=session.current_state.scene_id,
            stats=session.stats.to_dict(),
            inventory=session.inventory,
            flags=session.current_state.flags,
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
