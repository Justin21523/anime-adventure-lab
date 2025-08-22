# api/routers/llm.py
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

# Setup cache on module import
cache = bootstrap_cache()

router = APIRouter(prefix="/api/v1", tags=["LLM"])


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


def get_story_engine() -> StoryEngine:
    """Get or create story engine instance"""
    global _story_engine
    if _story_engine is None:
        # Initialize with a lightweight model for testing
        # In production, this should be configurable
        llm = TransformersLLM(
            model_name="microsoft/DialoGPT-medium",  # Small model for testing
            use_4bit=True,
            trust_remote_code=False,
        )
        _story_engine = StoryEngine(llm)
    return _story_engine


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    engine = get_story_engine()
    return {
        "status": "ok",
        "llm_available": engine.llm.is_available(),
        "model": engine.llm.model_name,
    }


@router.post("/turn", response_model=TurnResponseAPI)
async def process_turn(request: TurnRequestAPI):
    """Process a story turn"""
    try:
        engine = get_story_engine()

        if not engine.llm.is_available():
            raise HTTPException(status_code=503, detail="LLM model not available")

        # Convert API models to core models
        persona = Persona(
            name=request.persona.name,
            age=request.persona.age,
            personality=request.persona.personality,
            background=request.persona.background,
            speaking_style=request.persona.speaking_style,
            appearance=request.persona.appearance,
            goals=request.persona.goals,
            secrets=request.persona.secrets,
            memory_preferences=request.persona.memory_preferences,
        )

        game_state = GameState(
            scene_id=request.game_state.scene_id,
            turn_count=request.game_state.turn_count,
            flags=request.game_state.flags,
            inventory=request.game_state.inventory,
            current_location=request.game_state.current_location,
            timeline_notes=request.game_state.timeline_notes,
        )

        turn_request = TurnRequest(
            player_input=request.player_input,
            persona=persona,
            game_state=game_state,
            choice_id=request.choice_id,
        )

        # Process turn
        response = engine.process_turn(turn_request)

        # Convert back to API models
        api_response = TurnResponseAPI(
            narration=response.narration,
            dialogues=[
                DialogueAPI(speaker=d.speaker, text=d.text, emotion=d.emotion)
                for d in response.dialogues
            ],
            choices=[
                ChoiceAPI(id=c.id, text=c.text, description=c.description)
                for c in response.choices
            ],
            scene_change=response.scene_change,
        )

        # Add updated state if available
        if response.updated_state:
            api_response.updated_state = GameStateAPI(
                scene_id=response.updated_state.scene_id,
                turn_count=response.updated_state.turn_count,
                flags=response.updated_state.flags,
                inventory=response.updated_state.inventory,
                current_location=response.updated_state.current_location,
                timeline_notes=response.updated_state.timeline_notes,
            )

        return api_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Turn processing failed: {str(e)}")


@router.get("/persona/sample", response_model=PersonaAPI)
async def get_sample_persona():
    """Get a sample persona for testing"""
    engine = get_story_engine()
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


@router.get("/gamestate/sample", response_model=GameStateAPI)
async def get_sample_game_state():
    """Get a sample game state for testing"""
    engine = get_story_engine()
    state = engine.create_sample_game_state()

    return GameStateAPI(
        scene_id=state.scene_id,
        turn_count=state.turn_count,
        flags=state.flags,
        inventory=state.inventory,
        current_location=state.current_location,
        timeline_notes=state.timeline_notes,
    )
