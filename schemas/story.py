"""
Story API Schemas
"""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

from schemas.world import WorldAgentProfile, WorldPack


class StorySessionCreateRequest(BaseModel):
    """Create a new story session."""

    player_name: str = Field(..., description="Player name or protagonist")
    persona_id: Optional[str] = Field(
        "wise_sage", description="Persona to narrate the story"
    )
    world_id: str = Field(
        "default", description="World/namespace identifier for RAG & worldpacks"
    )
    runtime_preset_id: Optional[str] = Field(
        None,
        description="Runtime preset override (LLM/T2I). None/empty = follow world default.",
    )
    setting: str = Field("fantasy", description="Story setting")
    difficulty: str = Field("medium", description="Story difficulty")
    enhanced_mode: bool = Field(
        True, description="Enable enhanced contextual story mode"
    )
    use_agent: bool = Field(False, description="Leverage agent assistance")
    enrich_with_rag: Optional[bool] = Field(
        None,
        description="Augment prompts with RAG knowledge (None = auto by world/session default)",
    )
    rag_mode: Optional[Literal["auto", "on", "off"]] = Field(
        None,
        description="RAG mode override (auto/on/off). When provided, takes precedence over enrich_with_rag.",
    )
    rerank_mode: Optional[Literal["auto", "on", "off"]] = Field(
        None,
        description="Reranker mode override (auto/on/off). When auto, follows world default.",
    )
    rag_query: Optional[str] = Field(
        None, description="Optional query for retrieving lore/background"
    )
    initial_prompt: Optional[str] = Field(
        None, description="Optional opening scene prompt"
    )
    player_template_id: Optional[str] = Field(
        None, description="Optional player style template id (from WorldPack.player_templates)"
    )
    include_image: bool = Field(
        True, description="Whether to attempt auto scene image generation"
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

    @field_validator("world_id", mode="after")
    def validate_world_id(cls, value: str) -> str:
        value = str(value or "").strip()
        return value or "default"

    @field_validator("runtime_preset_id", mode="after")
    def validate_runtime_preset_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None


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
    enrich_with_rag: Optional[bool] = Field(
        None,
        description="Retrieve knowledge context from RAG (None = use session/world default)",
    )
    rag_mode: Optional[Literal["auto", "on", "off"]] = Field(
        None,
        description="RAG mode override (auto/on/off). When provided, takes precedence over enrich_with_rag.",
    )
    rerank_mode: Optional[Literal["auto", "on", "off"]] = Field(
        None,
        description="Reranker mode override (auto/on/off). When auto, follows world default.",
    )
    rag_query: Optional[str] = Field(
        None, description="Custom RAG query; defaults to player input"
    )
    top_k: int = Field(3, description="Number of RAG results to include")
    include_image: bool = Field(
        True, description="Whether to attempt auto scene image generation"
    )

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
    world_id: str = Field("default", description="World/namespace identifier")
    turn_count: int
    narrative: str
    choices: List[Dict[str, Any]]
    stats: Dict[str, Any]
    inventory: List[str]
    scene_id: Optional[str] = None
    flags: Dict[str, Any] = Field(default_factory=dict)
    agent_used: bool = False
    agent_overlay: Optional[Dict[str, Any]] = None
    agent_actions: Optional[Dict[str, Any]] = Field(
        None, description="Agent autonomous actions (modify flags/stats, etc.)"
    )
    knowledge_used: Optional[List[Dict[str, Any]]] = None
    context: Optional[Dict[str, Any]] = None
    scene_image_job_id: Optional[str] = Field(
        None, description="Background job id for scene image generation (if any)"
    )
    scene_image: Optional[SceneImage] = Field(
        None, description="Auto-generated scene image (if triggered)"
    )


class StoryTurnJobResponse(BaseModel):
    """Async story turn job enqueue response."""

    success: bool = True
    job_id: str = Field(..., description="Background job id for the story turn")
    status: str = Field("queued", description="Job status (queued/running/completed/failed/cancelled)")


class StorySessionInfo(BaseModel):
    """Lightweight session listing."""

    session_id: str
    player_name: str
    persona_id: Optional[str] = None
    world_id: str = Field("default", description="World/namespace identifier")
    turn_count: int
    is_active: bool
    updated_at: str
    enhanced_mode: bool = False
    current_scene: Optional[str] = None


class StoryMemoryStats(BaseModel):
    """Three-layer memory statistics (short/mid/long)."""

    short_term_count: int = 0
    summaries_count: int = 0
    total_turns_covered: int = 0
    turns_since_last_summary: int = 0
    rag_available: bool = False


class StoryShortTermMemory(BaseModel):
    """Short-term turn memory."""

    turn: int
    action: str
    result: str
    scene: Optional[str] = None


class StoryMemorySummary(BaseModel):
    """Mid-term compressed summary."""

    turn_range: str
    summary: str
    key_events: List[str] = Field(default_factory=list)


class StoryMemoryContext(BaseModel):
    """Memory context snapshot for UI."""

    short_term: List[StoryShortTermMemory] = Field(default_factory=list)
    summaries: List[StoryMemorySummary] = Field(default_factory=list)
    rag_results: List[Dict[str, Any]] = Field(default_factory=list)


class StoryTurnHistoryEntry(BaseModel):
    """Turn history entry for timeline UI (persisted in session.history)."""

    turn: int
    timestamp: Optional[str] = None
    player_input: str
    ai_response: str
    choice_id: Optional[str] = None
    scene_id: Optional[str] = None
    agent_used: Optional[bool] = None
    enriched_player_input: Optional[str] = Field(
        None, description="The enriched prompt actually sent to the engine (RAG snippets / agent_hint)."
    )
    rag_mode: Optional[Literal["auto", "on", "off"]] = Field(
        None, description="RAG mode used for this turn (if known)"
    )
    rag_query: Optional[str] = Field(None, description="RAG query used (if any)")
    rerank_mode: Optional[Literal["auto", "on", "off"]] = Field(
        None, description="Reranker mode used for this turn (if known)"
    )
    knowledge_used: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="RAG snippets used for this turn (content may be truncated)"
    )
    agent_overlay: Optional[Dict[str, Any]] = Field(
        None, description="Optional agent overlay (use_agent) applied to the prompt"
    )
    agent_actions: Optional[Dict[str, Any]] = None
    state_delta: Optional[Dict[str, Any]] = Field(
        None, description="State diff for this turn (flags/stats/inventory/relationships)"
    )
    scene_image_job_id: Optional[str] = Field(
        None, description="Background job id for scene image generation (if any)"
    )
    scene_image: Optional[SceneImage] = Field(
        None, description="Scene image generated for this turn (if any)"
    )
    artifacts: Optional[Dict[str, Any]] = Field(
        None,
        description="Normalized per-turn artifacts for Turn Inspector (rag/agents/diff/t2i/world sync, etc.)",
    )


class StorySessionDetail(BaseModel):
    """Detailed session information."""

    session_id: str
    player_name: str
    persona_id: Optional[str] = None
    world_id: str = Field("default", description="World/namespace identifier")
    runtime_preset_id: Optional[str] = Field(
        None, description="Runtime preset id used by this session (effective)"
    )
    created_at: str
    updated_at: str
    turn_count: int
    is_active: bool
    current_scene: Optional[str] = None
    player_template_id: Optional[str] = Field(
        None, description="Optional player template id chosen at session start"
    )
    worldpack_updated_at: Optional[str] = Field(
        None,
        description="Last worldpack updated_at applied into this session (set on init/sync)",
    )
    stats: Dict[str, Any]
    inventory: List[str]
    flags: Dict[str, Any] = Field(default_factory=dict)
    # Current game state fields for frontend
    narrative: Optional[str] = Field(None, description="Current narrative text")
    choices: List[Dict[str, Any]] = Field(default_factory=list, description="Available choices")
    turn_job_id: Optional[str] = Field(
        None, description="Active Story turn job id (if a turn is currently running)"
    )
    scene_image_job_id: Optional[str] = Field(
        None, description="Latest background job id for scene image generation (if any)"
    )
    scene_image: Optional[SceneImage] = Field(
        None, description="Most recent generated scene image"
    )
    memory_stats: Optional[StoryMemoryStats] = None
    memory_context: Optional[StoryMemoryContext] = None
    turn_history: List[StoryTurnHistoryEntry] = Field(
        default_factory=list,
        description="Recent persisted turn history entries for timeline UI",
    )
    agent_actions: Optional[Dict[str, Any]] = Field(
        None, description="Agent autonomous actions (tool results, etc.)"
    )
    # RAG status (world-aware)
    rag_auto: Optional[bool] = Field(
        None, description="RAG auto mode (None/True means follow world availability)"
    )
    rag_mode: Optional[Literal["auto", "on", "off"]] = Field(
        None, description="RAG mode derived from rag_auto/enrich_with_rag (for UI convenience)"
    )
    rag_available: Optional[bool] = Field(
        None, description="Does this world_id currently have indexed RAG documents?"
    )
    enrich_with_rag: Optional[bool] = Field(
        None, description="Effective enrich_with_rag for next turn"
    )
    rag_next_turn: Optional[bool] = Field(
        None, description="Whether next turn will be enriched with RAG"
    )
    rag_query: Optional[str] = Field(
        None, description="Optional stored query hint for RAG"
    )
    rerank_mode: Optional[Literal["auto", "on", "off"]] = Field(
        None, description="Reranker mode for next turn (auto/on/off)"
    )
    rerank_next_turn: Optional[bool] = Field(
        None, description="Whether next turn will use reranker stage"
    )


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


class StoryWorldSyncRequest(BaseModel):
    """Sync the latest WorldPack into an existing story session."""

    mode: Literal["add_only", "merge"] = Field(
        "add_only", description="add_only: 只新增缺少項目；merge: 合併更新角色 persona 欄位"
    )


class StoryWorldSyncResponse(BaseModel):
    """Result summary of a worldpack sync."""

    session_id: str
    world_id: str
    mode: str
    flags_added: List[str] = Field(default_factory=list)
    flags_updated: List[str] = Field(default_factory=list)
    characters_added: List[str] = Field(default_factory=list)
    characters_updated: List[str] = Field(default_factory=list)
    worldpack_updated_at: Optional[str] = None


class StoryWorldWritebackSuggestRequest(BaseModel):
    """Suggest a WorldPack patch based on the current story session state (requires user confirmation)."""

    include_flags: bool = Field(True, description="是否包含 flags（任務/地點/NPC/物品/事件/成就）")
    include_characters: bool = Field(True, description="是否匯出故事中出現的新角色到 worldpack.characters")
    include_rag_note: bool = Field(True, description="是否產生可寫回 RAG 的摘要文字（不會自動寫入）")
    max_new_characters: int = Field(10, ge=0, le=50, description="最多匯出幾個新角色模板")


class StoryWorldWritebackSuggestResponse(BaseModel):
    """Writeback suggestion result."""

    success: bool = True
    session_id: str
    world_id: str
    patch: Dict[str, Any] = Field(default_factory=dict)
    worldpack: WorldPack
    rag_note: Optional[str] = None
    summary: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class StoryAgentProfileResponse(BaseModel):
    """Session-level agent_profile snapshot for Story Orchestrator."""

    session_id: str
    world_id: str
    agent_profile: WorldAgentProfile


class StoryAgentProfileUpdateRequest(BaseModel):
    """Replace the session-level agent_profile (does not modify worldpack)."""

    agent_profile: WorldAgentProfile


class StoryAgentProfilePatchRequest(BaseModel):
    """Patch the session-level agent_profile (does not modify worldpack)."""

    enabled: Optional[bool] = Field(None, description="Enable Story multi-agent director")
    enabled_agents: Optional[List[str]] = Field(
        None,
        description="Empty list = enable all default sub-agents; otherwise only these names",
    )
    max_tool_calls: Optional[int] = Field(
        None, ge=0, le=20, description="Max tool calls per turn"
    )
    max_llm_calls: Optional[int] = Field(
        None, ge=0, le=10, description="Max LLM director calls per turn"
    )
    allowed_tools: Optional[List[str]] = Field(
        None,
        description="Empty list = default; otherwise restrict to these tool names",
    )
