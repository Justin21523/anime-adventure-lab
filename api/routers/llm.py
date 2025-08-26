# api/routers/llm.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

from core.shared_cache import bootstrap_cache
from core.llm.transformers_llm import TransformersLLM
from core.story.engine import StoryEngine
from core.story.data_structures import (
    Persona,
    GameState,
    TurnRequest,
    TurnResponse,
    DialogueEntry,
    Choice,
    Relationship,
    RelationType,
)

from ..dependencies import get_llm  # shared LLM singleton
from core.story.engine import StoryEngine  # use core engine, no sys.path hacks


router = APIRouter(tags=["llm"])


# Pydantic models for API
class PersonaAPI(BaseModel):
    name: str
    age: Optional[int] = None
    personality: List[str] = Field(default_factory=list)
    background: str = ""
    speaking_style: str = ""
    appearance: str = ""
    goals: List[str] = Field(default_factory=list)
    secrets: List[str] = Field(default_factory=list)
    memory_preferences: Dict[str, float] = Field(default_factory=dict)


class GameStateAPI(BaseModel):
    scene_id: str = "prologue"
    turn_count: int = 0
    flags: Dict[str, Any] = Field(default_factory=dict)
    inventory: List[str] = Field(default_factory=list)
    current_location: str = ""
    timeline_notes: List[str] = Field(default_factory=list)


class TurnRequestAPI(BaseModel):
    player_input: str = Field(..., description="Player input or action")
    persona: PersonaAPI
    game_state: GameStateAPI
    choice_id: Optional[str] = None


class DialogueAPI(BaseModel):
    speaker: str
    text: str
    emotion: Optional[str] = None


class ChoiceAPI(BaseModel):
    id: str
    text: str
    description: str = ""


class TurnResponseAPI(BaseModel):
    narration: str
    dialogues: List[DialogueAPI] = Field(default_factory=list)
    choices: List[ChoiceAPI] = Field(default_factory=list)
    updated_state: Optional[GameStateAPI] = None
    scene_change: Optional[str] = None


# Global story engine (will be initialized on first use)
_story_engine: Optional[StoryEngine] = None


def get_story_engine(llm) -> StoryEngine:
    """Get or create story engine instance"""
    global _story_engine
    if _story_engine is None:
        _story_engine = StoryEngine(llm)
    return _story_engine


@router.get("/llm/health")
async def llm_health(llm=Depends(get_llm)):
    """Health check for LLM-backed story engine."""
    eng = get_story_engine(llm)
    is_available = getattr(eng.llm, "is_available", lambda: True)()
    model_name = getattr(eng.llm, "model_name", "unknown")
    return {"status": "ok", "llm_available": is_available, "model": model_name}


@router.post("/llm/turn", response_model=TurnResponseAPI)
async def process_turn(request: TurnRequestAPI, llm=Depends(get_llm)):
    """Orchestrate a story turn with the shared LLM."""
    try:
        eng = get_story_engine(llm)
        if hasattr(eng.llm, "is_available") and not eng.llm.is_available():
            raise HTTPException(status_code=503, detail="LLM model not available")

        # Build core objects using engine helpers to avoid tight coupling
        persona = eng.build_persona(**request.persona.dict())
        game_state = eng.build_game_state(**request.game_state.dict())

        turn_req = eng.build_turn_request(
            player_input=request.player_input,
            persona=persona,
            game_state=game_state,
            choice_id=request.choice_id,
        )
        resp = eng.process_turn(turn_req)

        api_resp = TurnResponseAPI(
            narration=resp.narration,
            dialogues=[
                DialogueAPI(speaker=d.speaker, text=d.text, emotion=d.emotion)
                for d in resp.dialogues
            ],
            choices=[
                ChoiceAPI(id=c.id, text=c.text, description=c.description)
                for c in resp.choices
            ],
            scene_change=resp.scene_change,
        )
        if resp.updated_state:
            api_resp.updated_state = GameStateAPI(**resp.updated_state.dict())
        return api_resp
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Turn processing failed: {e}")


@router.get("/llm/persona/sample", response_model=PersonaAPI)
async def get_sample_persona(llm=Depends(get_llm)):
    """Get a sample persona for testing"""
    engine = get_story_engine(llm)
    persona = engine.create_sample_persona()

    return PersonaAPI(
        name=persona.name,
        age=persona.age,
        personality=persona.personality,
        background=persona.background,
        speaking_style=persona.speaking_style,
        appearance=persona.appearance,
        goals=persona.goals,
        secrets=persona.secrets,
        memory_preferences=persona.memory_preferences,
    )


@router.get("/llm/gamestate/sample", response_model=GameStateAPI)
async def get_sample_game_state(llm=Depends(get_llm)):
    """Get a sample game state for testing"""
    engine = get_story_engine(llm)
    state = engine.create_sample_game_state()

    return GameStateAPI(
        scene_id=state.scene_id,
        turn_count=state.turn_count,
        flags=state.flags,
        inventory=state.inventory,
        current_location=state.current_location,
        timeline_notes=state.timeline_notes,
    )
