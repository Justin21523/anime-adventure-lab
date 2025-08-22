# api/routers/story.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from core.shared_cache import bootstrap_cache
from core.story.engine import StoryEngine, TurnRequest
from core.story.game_state import GameState, RelationType
from core.story.persona import PersonaManager
from core.rag.engine import ChineseRAGEngine
from core.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)

# Setup cache on module import
cache = bootstrap_cache()

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


# Dependencies
async def get_story_engine() -> StoryEngine:
    # This would be injected from the main app
    # For now, placeholder
    pass


async def get_persona_manager() -> PersonaManager:
    # This would be injected from the main app
    pass


# In-memory storage for demo (in production, use Redis/DB)
game_sessions: Dict[str, GameState] = {}
save_files: Dict[str, Dict[str, GameState]] = {}


@router.post("/turn", response_model=StoryResponseModel)
async def process_turn(
    request: TurnRequestModel, story_engine: StoryEngine = Depends(get_story_engine)
):
    """處理一個故事回合"""
    try:
        # Get or create game state
        if request.session_id not in game_sessions:
            game_state = GameState(
                session_id=request.session_id, world_id=request.world_id
            )
            game_sessions[request.session_id] = game_state
        else:
            game_state = game_sessions[request.session_id]

        # Create turn request
        turn_req = TurnRequest(
            player_input=request.player_input,
            choice_id=request.choice_id,
            world_id=request.world_id,
            session_id=request.session_id,
        )

        # Process the turn
        response = await story_engine.process_turn(turn_req, game_state)

        # Update stored game state
        game_sessions[request.session_id] = game_state

        return StoryResponseModel(
            narration=response.narration,
            dialogues=response.dialogues,
            choices=response.choices,
            citations=response.citations,
            game_state_changes=response.game_state_changes,
            metadata=response.metadata,
        )

    except Exception as e:
        logger.error(f"Error processing turn: {e}")
        raise HTTPException(status_code=500, detail=f"Story processing error: {str(e)}")


@router.get("/state/{session_id}", response_model=GameStateResponse)
async def get_game_state(session_id: str):
    """取得遊戲狀態"""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    game_state = game_sessions[session_id]

    # Get recent events (last 5)
    recent_events = []
    for event in game_state.timeline[-5:]:
        recent_events.append(
            {
                "event_id": event.event_id,
                "type": event.event_type.value,
                "description": event.description,
                "timestamp": event.timestamp.isoformat(),
                "characters": event.characters_involved,
            }
        )

    return GameStateResponse(
        session_id=game_state.session_id,
        world_id=game_state.world_id,
        current_scene=game_state.current_scene,
        current_location=game_state.current_location,
        turn_count=game_state.turn_count,
        player_name=game_state.player_name,
        player_health=game_state.player_health,
        player_energy=game_state.player_energy,
        relationships={
            k: f"{v.relation_type.value} (親密:{v.affinity}, 信任:{v.trust})"
            for k, v in game_state.relationships.items()
        },
        inventory={
            k: f"{v.name} x{v.quantity}" for k, v in game_state.inventory.items()
        },
        world_flags={k: v.value for k, v in game_state.world_flags.items()},
        recent_events=recent_events,
    )


@router.post("/save")
async def save_game(request: SaveGameRequest):
    """儲存遊戲"""
    if request.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.session_id not in save_files:
        save_files[request.session_id] = {}

    # Deep copy the game state
    game_state = game_sessions[request.session_id]
    save_files[request.session_id][request.save_name] = game_state

    logger.info(f"Saved game {request.save_name} for session {request.session_id}")

    return {
        "success": True,
        "save_name": request.save_name,
        "timestamp": datetime.now().isoformat(),
        "turn_count": game_state.turn_count,
    }


@router.post("/load")
async def load_game(request: LoadGameRequest):
    """讀取遊戲"""
    if request.session_id not in save_files:
        raise HTTPException(status_code=404, detail="No saves found for this session")

    saves = save_files[request.session_id]

    if request.save_name:
        if request.save_name not in saves:
            raise HTTPException(status_code=404, detail="Save file not found")

        # Load specific save
        game_sessions[request.session_id] = saves[request.save_name]

        return {
            "success": True,
            "loaded_save": request.save_name,
            "turn_count": saves[request.save_name].turn_count,
        }
    else:
        # List available saves
        save_list = []
        for save_name, save_state in saves.items():
            save_list.append(
                {
                    "save_name": save_name,
                    "turn_count": save_state.turn_count,
                    "last_updated": save_state.last_updated.isoformat(),
                    "world_id": save_state.world_id,
                    "current_scene": save_state.current_scene,
                }
            )

        return {"saves": save_list}


@router.post("/new_session")
async def create_new_session(
    world_id: str, player_name: str = "玩家", starting_scene: str = "opening"
):
    """建立新的遊戲會話"""
    import uuid

    session_id = str(uuid.uuid4())

    game_state = GameState(
        session_id=session_id,
        world_id=world_id,
        player_name=player_name,
        current_scene=starting_scene,
    )

    game_sessions[session_id] = game_state

    logger.info(f"Created new session {session_id} for world {world_id}")

    return {
        "session_id": session_id,
        "world_id": world_id,
        "player_name": player_name,
        "starting_scene": starting_scene,
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """刪除遊戲會話"""
    if session_id in game_sessions:
        del game_sessions[session_id]

    if session_id in save_files:
        del save_files[session_id]

    return {"success": True, "deleted_session": session_id}


# Persona management endpoints
@router.post("/persona")
async def create_persona(
    request: PersonaCreateRequest,
    persona_manager: PersonaManager = Depends(get_persona_manager),
):
    """創建角色人設"""
    try:
        persona = persona_manager.create_persona_from_data(
            {
                "id": request.character_id,
                "name": request.name,
                "description": request.description,
                "background": request.background,
                "personality": request.personality,
                "speech_style": request.speech_style,
                "appearance": request.appearance,
            }
        )

        persona_manager.add_persona(persona)

        return {
            "success": True,
            "character_id": request.character_id,
            "message": f"角色 {request.name} 已創建",
        }

    except Exception as e:
        logger.error(f"Error creating persona: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating persona: {str(e)}")


@router.get("/persona/{character_id}")
async def get_persona(
    character_id: str, persona_manager: PersonaManager = Depends(get_persona_manager)
):
    """取得角色人設"""
    persona = persona_manager.get_persona(character_id)
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
    persona_manager: PersonaManager = Depends(get_persona_manager),
):
    """為角色添加記憶"""
    persona = persona_manager.get_persona(character_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Character not found")

    persona_manager.update_persona_memory(
        character_id=character_id,
        event_description=content,
        importance=importance,
        emotional_impact=emotional_impact,
        related_characters=related_characters,
    )

    return {
        "success": True,
        "character_id": character_id,
        "memory_added": content[:50] + "..." if len(content) > 50 else content,
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
    """取得當前可用的選擇"""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    game_state = game_sessions[session_id]

    # This would use the choice resolver to get available choices
    # For now, return basic choices
    choices = [
        {"id": "explore", "text": "探索周圍", "description": "仔細查看當前環境"},
        {"id": "rest", "text": "休息", "description": "恢復體力和精神"},
        {"id": "continue", "text": "繼續", "description": "繼續當前情節"},
    ]

    return {"choices": choices}


@router.get("/health")
async def health_check():
    """health check"""
    return {
        "status": "healthy",
        "active_sessions": len(game_sessions),
        "total_saves": sum(len(saves) for saves in save_files.values()),
        "timestamp": datetime.now().isoformat(),
    }
