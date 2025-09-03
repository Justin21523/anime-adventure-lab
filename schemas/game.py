# schemas/game.py
"""
Text Adventure Game API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from .base import BaseRequest, BaseResponse


class NewGameRequest(BaseRequest):
    """Create new game session request"""

    player_name: str = Field(
        ..., min_length=1, max_length=50, description="Player character name"
    )
    persona_id: str = Field("default", description="Game master persona ID")
    setting: str = Field("fantasy", description="Game world setting")
    difficulty: str = Field("normal", description="Game difficulty level")

    @field_validator("player_name", mode="after")
    def validate_player_name(cls, v):
        if not v.strip():
            raise ValueError("Player name cannot be empty")
        return v.strip()

    @field_validator("setting", mode="after")
    def validate_setting(cls, v):
        allowed_settings = ["fantasy", "sci-fi", "modern", "historical", "mystery"]
        if v not in allowed_settings:
            raise ValueError(f"Setting must be one of: {', '.join(allowed_settings)}")
        return v

    @field_validator("difficulty", mode="after")
    def validate_difficulty(cls, v):
        allowed_difficulties = ["easy", "normal", "hard", "nightmare"]
        if v not in allowed_difficulties:
            raise ValueError(
                f"Difficulty must be one of: {', '.join(allowed_difficulties)}"
            )
        return v


class GameStepRequest(BaseRequest):
    """Game action/step request"""

    session_id: str = Field(..., description="Game session ID")
    action: str = Field(
        ..., min_length=1, max_length=500, description="Player action description"
    )
    choice_id: Optional[str] = Field(
        None, description="Predefined choice ID if applicable"
    )

    @field_validator("action", mode="after")
    def validate_action(cls, v):
        if not v.strip():
            raise ValueError("Action cannot be empty")
        return v.strip()


class GameChoice(BaseModel):
    """Single player choice option"""

    id: str = Field(..., description="Choice identifier")
    text: str = Field(..., description="Choice display text")
    description: str = Field("", description="Detailed choice description")
    requirements: Optional[Dict[str, Any]] = Field(
        None, description="Requirements to select this choice"
    )


class GameDialogue(BaseModel):
    """Single dialogue entry"""

    speaker: str = Field(..., description="Character/speaker name")
    text: str = Field(..., description="Dialogue content")
    emotion: Optional[str] = Field(None, description="Speaker emotion/tone")


class GameResponse(BaseResponse):
    """Game turn response"""

    session_id: str = Field(..., description="Game session ID")
    turn_count: int = Field(..., description="Current turn number")
    scene: str = Field(..., description="Current scene/location")
    narration: str = Field(..., description="Narrative description of events")
    dialogues: List[GameDialogue] = Field(
        default_factory=list, description="Character dialogues"
    )
    choices: List[GameChoice] = Field(..., description="Available player choices")

    # Game state
    game_state: Dict[str, Any] = Field(
        ..., description="Current game state (inventory, stats, flags)"
    )
    status: str = Field("active", description="Game session status")

    # Optional analysis
    story_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Story progression analysis"
    )


class GameSessionSummary(BaseResponse):
    """Game session summary information"""

    session_id: str = Field(..., description="Session identifier")
    player_name: str = Field(..., description="Player character name")
    current_scene: str = Field(..., description="Current scene/location")
    turn_count: int = Field(..., description="Number of turns played")
    inventory_count: int = Field(..., description="Number of items in inventory")
    stats: Dict[str, int] = Field(..., description="Player statistics")
    active_flags: List[str] = Field(
        default_factory=list, description="Active story flags"
    )
    relationships: Dict[str, int] = Field(
        default_factory=dict, description="NPC relationship levels"
    )
    status: str = Field("active", description="Session status (active/saved/completed)")
    created_at: Optional[str] = Field(None, description="Session creation timestamp")
    last_action: Optional[str] = Field(None, description="Last player action")


class GamePersonaInfo(BaseModel):
    """Game persona information"""

    name: str = Field(..., description="Persona display name")
    description: str = Field(..., description="Persona description")
    personality: List[str] = Field(
        default_factory=list, description="Personality traits"
    )
    speech_style: str = Field("", description="Speaking style description")
    recommended_for: Optional[List[str]] = Field(
        None, description="Recommended game settings"
    )


class GameInventoryResponse(BaseResponse):
    """Player inventory response"""

    session_id: str = Field(..., description="Game session ID")
    inventory: List[str] = Field(..., description="Player inventory items")
    inventory_count: int = Field(..., description="Total item count")
    stats: Dict[str, int] = Field(..., description="Player stats")

    # Item analysis
    item_categories: Optional[Dict[str, List[str]]] = Field(
        None, description="Items grouped by category"
    )
    total_value: Optional[int] = Field(None, description="Total inventory value")


class GameStatsRequest(BaseModel):
    """Request for game statistics"""

    session_ids: Optional[List[str]] = Field(
        None, description="Specific session IDs to analyze"
    )
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range filter")


class GameStatsResponse(BaseResponse):
    """Game statistics response"""

    total_sessions: int = Field(..., description="Total number of sessions")
    active_sessions: int = Field(..., description="Currently active sessions")
    average_turns_per_session: float = Field(
        ..., description="Average turns per session"
    )
    most_popular_personas: List[str] = Field(..., description="Most used personas")
    most_popular_settings: List[str] = Field(..., description="Most used settings")

    # Player engagement
    longest_session_turns: int = Field(
        ..., description="Highest turn count in any session"
    )
    total_turns_played: int = Field(..., description="Total turns across all sessions")

    # Temporal analysis
    sessions_today: int = Field(..., description="Sessions created today")
    sessions_this_week: int = Field(..., description="Sessions created this week")
