# api/routers/story.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime

from ..dependencies import get_story_engine, get_persona_manager
from core.story.game_state import GameState

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/story", tags=["story"])


# Pydantic models for API
class TurnRequestModel(BaseModel):
    player_input: str = Field(..., description="玩家輸入")
    choice_id: Optional[str] = Field(None, description="選擇ID（如果是選擇回應）")
    world_id: str = Field(..., description="世界ID")
    session_id: str = Field(..., description="會話ID")


class GameStateResponse(BaseModel):
    session_id: str
    world_id: str
    current_scene: str
    current_location: str
    turn_count: int
    player_name: str
    player_health: int
    player_energy: int
    relationships: Dict[str, Any]
    inventory: Dict[str, Any]
    world_flags: Dict[str, Any]
    recent_events: List[Dict[str, Any]]


class StoryResponseModel(BaseModel):
    narration: str
    dialogues: List[Dict[str, str]]
    choices: List[Dict[str, Any]]
    citations: List[str]
    game_state_changes: Dict[str, Any]
    metadata: Dict[str, Any]


class SaveGameRequest(BaseModel):
    session_id: str
    save_name: str = Field(..., description="存檔名稱")


class LoadGameRequest(BaseModel):
    session_id: str
    save_name: Optional[str] = None


class PersonaCreateRequest(BaseModel):
    character_id: str
    name: str
    description: str
    background: Optional[str] = ""
    personality: Optional[Dict[str, Any]] = {}
    speech_style: Optional[str] = ""
    appearance: Optional[str] = ""


# In-memory storage for demo (in production, use Redis/DB)
game_sessions: Dict[str, GameState] = {}
save_files: Dict[str, Dict[str, GameState]] = {}


@router.post("/turn", response_model=StoryResponseModel)
async def process_turn(
    request: TurnRequestModel, story_engine=Depends(get_story_engine)
):
    """Process a story turn using DI-injected engine."""
    try:
        # get or create state
        game_state = game_sessions.get(request.session_id) or GameState(
            session_id=request.session_id, world_id=request.world_id
        )
        game_sessions[request.session_id] = game_state

        # Build and process
        turn_req = story_engine.build_turn_request(
            player_input=request.player_input,
            choice_id=request.choice_id,
            world_id=request.world_id,
            session_id=request.session_id,
        )
        response = await story_engine.process_turn(turn_req, game_state)

        return StoryResponseModel(
            narration=response.narration,
            dialogues=response.dialogues,
            choices=response.choices,
            citations=response.citations,
            game_state_changes=response.game_state_changes,
            metadata=response.metadata,
        )
    except Exception as e:
        logger.exception("Error processing turn")
        raise HTTPException(status_code=500, detail=f"Story processing error: {e}")


@router.get("/state/{session_id}", response_model=GameStateResponse)
async def get_game_state(session_id: str):
    """Get current game state."""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    gs = game_sessions[session_id]

    recent_events = [
        {
            "event_id": ev.event_id,
            "type": ev.event_type.value,
            "description": ev.description,
            "timestamp": ev.timestamp.isoformat(),
            "characters": ev.characters_involved,
        }
        for ev in gs.timeline[-5:]
    ]
    return GameStateResponse(
        session_id=gs.session_id,
        world_id=gs.world_id,
        current_scene=gs.current_scene,
        current_location=gs.current_location,
        turn_count=gs.turn_count,
        player_name=gs.player_name,
        player_health=gs.player_health,
        player_energy=gs.player_energy,
        relationships={
            k: f"{v.relation_type.value} (aff:{v.affinity}, trust:{v.trust})"
            for k, v in gs.relationships.items()
        },
        inventory={k: f"{v.name} x{v.quantity}" for k, v in gs.inventory.items()},
        world_flags={k: v.value for k, v in gs.world_flags.items()},
        recent_events=recent_events,
    )


@router.post("/save")
async def save_game(request: SaveGameRequest):
    """Save current session state."""
    if request.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    save_files.setdefault(request.session_id, {})[request.save_name] = game_sessions[
        request.session_id
    ]
    return {
        "success": True,
        "save_name": request.save_name,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/load")
async def load_game(request: LoadGameRequest):
    """Load a saved session or list saves."""
    if request.session_id not in save_files:
        raise HTTPException(status_code=404, detail="No saves for this session")
    saves = save_files[request.session_id]
    if request.save_name:
        if request.save_name not in saves:
            raise HTTPException(status_code=404, detail="Save file not found")
        game_sessions[request.session_id] = saves[request.save_name]
        return {
            "success": True,
            "loaded_save": request.save_name,
            "turn_count": saves[request.save_name].turn_count,
        }
    return {
        "saves": [
            {
                "save_name": name,
                "turn_count": state.turn_count,
                "last_updated": state.last_updated.isoformat(),
                "world_id": state.world_id,
                "current_scene": state.current_scene,
            }
            for name, state in saves.items()
        ]
    }


@router.post("/new_session")
async def create_new_session(
    world_id: str, player_name: str = "player", starting_scene: str = "opening"
):
    """Create a new game session."""
    import uuid

    session_id = str(uuid.uuid4())
    gs = GameState(
        session_id=session_id,
        world_id=world_id,
        player_name=player_name,
        current_scene=starting_scene,
    )
    game_sessions[session_id] = gs
    return {
        "session_id": session_id,
        "world_id": world_id,
        "player_name": player_name,
        "starting_scene": starting_scene,
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a game session and its saves."""
    game_sessions.pop(session_id, None)
    save_files.pop(session_id, None)
    return {"success": True, "deleted_session": session_id}


# Persona management endpoints
@router.post("/persona")
async def create_persona(req: PersonaCreateRequest, pm=Depends(get_persona_manager)):
    """Create a persona."""
    try:
        persona = pm.create_persona_from_data(
            {
                "id": req.character_id,
                "name": req.name,
                "description": req.description,
                "background": req.background,
                "personality": req.personality,
                "speech_style": req.speech_style,
                "appearance": req.appearance,
            }
        )
        pm.add_persona(persona)
        return {"success": True, "character_id": req.character_id}
    except Exception as e:
        logger.exception("create_persona failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/persona/{character_id}")
async def get_persona(character_id: str, pm=Depends(get_persona_manager)):
    """Get persona data."""
    persona = pm.get_persona(character_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Character not found")
    return persona.to_dict()


@router.post("/persona/{character_id}/memory")
async def add_persona_memory(
    character_id: str,
    content: str,
    importance: int = 5,
    emotional_impact: int = 0,
    related_characters: List[str] = [],
    pm=Depends(get_persona_manager),
):
    """Append persona memory entry."""
    persona = pm.get_persona(character_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Character not found")
    pm.update_persona_memory(
        character_id=character_id,
        event_description=content,
        importance=importance,
        emotional_impact=emotional_impact,
        related_characters=related_characters,
    )
    return {
        "success": True,
        "character_id": character_id,
        "memory_added": content[:50] + ("..." if len(content) > 50 else ""),
    }


@router.get("/persona/{char1_id}/relationship/{char2_id}")
async def analyze_relationship(
    char1_id: str,
    char2_id: str,
    persona_manager: PersonaManager = Depends(get_persona_manager),
):
    """分析兩個角色的關係動態"""
    relationship_analysis = persona_manager.analyze_relationship_dynamics(
        char1_id, char2_id
    )
    return relationship_analysis


@router.get("/choices/{session_id}")
async def get_available_choices(session_id: str):
    """Return available choices (placeholder)."""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    choices = [
        {
            "id": "explore",
            "text": "Explore surroundings",
            "description": "Look around carefully",
        },
        {"id": "rest", "text": "Rest", "description": "Recover stamina"},
        {
            "id": "continue",
            "text": "Continue",
            "description": "Proceed with the current plot",
        },
    ]
    return {"choices": choices}


@router.get("/health")
async def health_check():
    """Basic health info for story router."""
    return {
        "status": "healthy",
        "active_sessions": len(game_sessions),
        "total_saves": sum(len(s) for s in save_files.values()),
        "timestamp": datetime.now().isoformat(),
    }
