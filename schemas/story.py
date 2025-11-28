"""
Story API Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class StorySessionCreateRequest(BaseModel):
    """Create a new story session."""

    player_name: str = Field(..., description="Player name or protagonist")
    persona_id: Optional[str] = Field(
        "wise_sage", description="Persona to narrate the story"
    )
    setting: str = Field("fantasy", description="Story setting")
    difficulty: str = Field("medium", description="Story difficulty")
    enhanced_mode: bool = Field(
        True, description="Enable enhanced contextual story mode"
    )
    use_agent: bool = Field(False, description="Leverage agent assistance")
    enrich_with_rag: bool = Field(
        False, description="Augment prompts with RAG knowledge"
    )
    rag_query: Optional[str] = Field(
        None, description="Optional query for retrieving lore/background"
    )

    @field_validator("player_name", mode="after")
    def validate_player(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Player name cannot be empty")
        return value

    @field_validator("persona_id", mode="after")
    def validate_persona(cls, value: Optional[str]) -> str:
        if value is None or not str(value).strip():
            return "wise_sage"
        return value


class StoryTurnRequest(BaseModel):
    """Process a turn in an existing story session."""

    session_id: str = Field(..., description="Story session ID")
    player_input: str = Field(..., description="Player action or input text")
    choice_id: Optional[str] = Field(None, description="Optional choice to execute")
    use_agent: bool = Field(False, description="Use story agent for guidance")
    scenario_type: Optional[str] = Field(None, description="Scenario type for agent")
    scenario_data: Optional[Dict[str, Any]] = Field(
        None, description="Scenario data passed to agents"
    )
    enrich_with_rag: bool = Field(
        False, description="Retrieve knowledge context from RAG"
    )
    rag_query: Optional[str] = Field(
        None, description="Custom RAG query; defaults to player input"
    )
    top_k: int = Field(3, description="Number of RAG results to include")

    @field_validator("player_input", mode="after")
    def validate_input(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Player input cannot be empty")
        return value


class SceneImage(BaseModel):
    """Generated scene image data."""

    image_url: str = Field(..., description="URL to generated scene image")
    prompt: str = Field(..., description="Positive prompt used for generation")
    negative_prompt: str = Field(..., description="Negative prompt used")
    generation_time: float = Field(..., description="Time taken to generate (seconds)")
    seed: Optional[int] = Field(None, description="Random seed used")
    width: int = Field(768, description="Image width")
    height: int = Field(768, description="Image height")


class StoryTurnResponse(BaseModel):
    """Story turn response."""

    session_id: str
    turn_count: int
    narrative: str
    choices: List[Dict[str, Any]]
    stats: Dict[str, Any]
    inventory: List[str]
    scene_id: Optional[str] = None
    flags: Dict[str, Any] = Field(default_factory=dict)
    agent_used: bool = False
    agent_overlay: Optional[Dict[str, Any]] = None
    knowledge_used: Optional[List[Dict[str, Any]]] = None
    context: Optional[Dict[str, Any]] = None
    scene_image: Optional[SceneImage] = Field(
        None, description="Auto-generated scene image (if triggered)"
    )


class StorySessionInfo(BaseModel):
    """Lightweight session listing."""

    session_id: str
    player_name: str
    persona_id: Optional[str] = None
    turn_count: int
    is_active: bool
    updated_at: str
    enhanced_mode: bool = False
    current_scene: Optional[str] = None


class StorySessionDetail(BaseModel):
    """Detailed session information."""

    session_id: str
    player_name: str
    persona_id: Optional[str] = None
    created_at: str
    updated_at: str
    turn_count: int
    is_active: bool
    current_scene: Optional[str] = None
    stats: Dict[str, Any]
    inventory: List[str]
    flags: Dict[str, Any] = Field(default_factory=dict)


class StoryContextSnapshot(BaseModel):
    """Enhanced context snapshot."""

    session_id: str
    player_name: str
    current_scene: Optional[Dict[str, Any]] = None
    present_characters: List[Dict[str, Any]] = Field(default_factory=list)
    world_flags: Dict[str, Any] = Field(default_factory=dict)
    main_plot_points: List[str] = Field(default_factory=list)
    recent_decisions: List[Any] = Field(default_factory=list)
    total_scenes: int = 0
    total_characters: int = 0


class StoryChoicePreview(BaseModel):
    """Preview for a choice."""

    choice_id: str
    display_text: Optional[str] = None
    type: Optional[str] = None
    difficulty: Optional[str] = None
    success_chance: Optional[float] = None
    requirements: Optional[Any] = None
    consequences_preview: Optional[Any] = None
    description: Optional[str] = None


class StoryExportResponse(BaseModel):
    """Export story session data."""

    session_id: str
    exported: Dict[str, Any]


class StoryImportRequest(BaseModel):
    """Import story session payload."""

    session_data: Dict[str, Any]


class StoryImportResponse(BaseModel):
    """Import story session result."""

    success: bool
    session_id: Optional[str] = None
    error: Optional[str] = None


class StoryMetricsResponse(BaseModel):
    """Story system metrics."""

    session_metrics: Dict[str, Any]
    system_metrics: Dict[str, Any]
    enhanced_metrics: Optional[Dict[str, Any]] = None
    agents_ready: bool = False
    rag_ready: bool = False
    llm_ready: bool = False


class StoryAgentActionRequest(BaseModel):
    """Request agent assistance for story action."""

    session_id: str
    player_input: str
    narrative_style: str = Field("adventure", description="Narrative flavor")
    scenario_type: Optional[str] = Field(None, description="Scenario type")
    scenario_data: Optional[Dict[str, Any]] = Field(None, description="Scenario data")


class StoryAgentActionResponse(BaseModel):
    """Agent-assisted story action response."""

    success: bool
    narrative: str
    available_actions: List[str]
    consequences: List[str] = Field(default_factory=list)
    agent_steps: Optional[int] = None
    tools_used: Optional[List[Any]] = None
    fallback_used: bool = False


class StoryKnowledgeSearchRequest(BaseModel):
    """Search RAG knowledge for story support."""

    query: str
    top_k: int = 3


class StoryKnowledgeSearchResponse(BaseModel):
    """RAG search results for story."""

    results: List[Dict[str, Any]]
    used_engine: str = "rag"
    available: bool = True
