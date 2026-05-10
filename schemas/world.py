"""
WorldPack (World Studio) API Schemas

WorldPack 是 Story 的「世界設定 + 角色/NPC 模板 + 視覺風格(LoRA)」整包資料，
會被存放在 AI_WORLDPACKS_ROOT（見 core/shared_cache.py）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


_WORLD_ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$"


class WorldLoRAConfig(BaseModel):
    """LoRA adapter applied for this world's visual style."""

    lora_id: str = Field(..., min_length=1, description="LoRA identifier/name")
    weight: float = Field(0.8, ge=0.0, le=2.0, description="LoRA weight/scale")


class WorldVisualStyle(BaseModel):
    """Visual language settings for T2I scene generation."""

    prompt_prefix: str = Field(
        "", description="Prepended to prompts for consistent style (comma-separated tags ok)"
    )
    negative_prompt: str = Field(
        "", description="Extra negative prompt tokens for this world"
    )
    base_model: Optional[str] = Field(
        None, description="Optional base diffusion model id for this world"
    )
    default_loras: List[WorldLoRAConfig] = Field(
        default_factory=list, description="Default LoRAs to apply for scene generation"
    )


class WorldAgentProfile(BaseModel):
    """Controls Story orchestrator sub-agents + budgets for this world."""

    enabled: bool = Field(True, description="Enable Story multi-agent director")
    enabled_agents: List[str] = Field(
        default_factory=list,
        description="Empty = enable all default sub-agents; otherwise only these names",
    )
    max_tool_calls: int = Field(6, ge=0, le=20, description="Max tool calls per turn")
    max_llm_calls: int = Field(1, ge=0, le=10, description="Max LLM director calls per turn")
    allowed_tools: List[str] = Field(
        default_factory=list,
        description="Empty = default; otherwise restrict to these tool names",
    )


class WorldRagProfile(BaseModel):
    """World-level RAG knobs (world_id scoped)."""

    enable_rerank: bool = Field(
        False, description="Enable reranker stage for this world (CPU recommended)"
    )


class WorldPlayerTemplate(BaseModel):
    """Player character template (role style)."""

    template_id: str = Field(..., min_length=1, description="Template identifier")
    name: str = Field(..., min_length=1, description="Template name")
    description: str = Field("", description="Short description for UI")
    personality_traits: List[str] = Field(default_factory=list)
    speaking_style: str = Field("", description="How the player character speaks/acts")
    background_story: str = Field("", description="Background story")
    motivations: List[str] = Field(default_factory=list)
    persona_prompt: str = Field("", description="Optional persona prompt injected into story context")


class WorldCharacterTemplate(BaseModel):
    """NPC / companion / antagonist template."""

    character_id: str = Field(..., min_length=1, description="Unique character id")
    name: str = Field(..., min_length=1, description="Display name")
    role: Literal["npc", "companion", "antagonist"] = Field("npc")
    image_url: Optional[str] = Field(None, description="Character portrait/sprite URL")
    personality_traits: List[str] = Field(default_factory=list)
    speaking_style: str = Field("", description="Speech style")
    background_story: str = Field("", description="Background story")
    motivations: List[str] = Field(default_factory=list)
    relationships: Dict[str, str] = Field(default_factory=dict)
    persona_prompt: str = Field("", description="System prompt for this character")
    content_restrictions: List[str] = Field(default_factory=list)
    start_in_opening: bool = Field(
        False, description="Whether this character appears in the opening scene"
    )


class WorldPack(BaseModel):
    """A complete world definition used by World Studio and Story initialization."""

    world_id: str = Field(..., pattern=_WORLD_ID_PATTERN)
    name: str = Field(..., min_length=1)
    description: str = Field("", description="World description")
    setting: str = Field("fantasy", description="Default story setting")
    difficulty: str = Field("medium", description="Default difficulty")
    runtime_preset_id: Optional[str] = Field(
        None,
        description="Runtime preset id (hardware/quality profile). Used as world default for Story/LLM/T2I.",
    )

    visual: WorldVisualStyle = Field(default_factory=WorldVisualStyle)
    player_templates: List[WorldPlayerTemplate] = Field(default_factory=list)
    characters: List[WorldCharacterTemplate] = Field(default_factory=list)
    world_flags: Dict[str, bool] = Field(default_factory=dict)
    agent_profile: WorldAgentProfile = Field(default_factory=WorldAgentProfile)
    rag_profile: WorldRagProfile = Field(default_factory=WorldRagProfile)

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @field_validator("world_id", mode="after")
    def validate_world_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("world_id cannot be empty")
        return value


class WorldSummary(BaseModel):
    """Lightweight listing item for World selector."""

    world_id: str
    name: str
    description: str = ""
    setting: str = "fantasy"
    difficulty: str = "medium"
    updated_at: str
    player_templates_count: int = 0
    characters_count: int = 0
    default_loras_count: int = 0


class WorldCreateRequest(BaseModel):
    world_id: str = Field(..., pattern=_WORLD_ID_PATTERN)
    name: str = Field(..., min_length=1)
    description: str = ""
    setting: str = "fantasy"
    difficulty: str = "medium"


class WorldUpdateRequest(BaseModel):
    """Full replace update for now (client sends complete WorldPack)."""

    world: WorldPack


class WorldAgentSuggestRequest(BaseModel):
    """World Studio: ask multi-agent orchestrator to propose a WorldPack update."""

    instruction: str = Field(..., min_length=1, description="你想讓世界工作室 AI 幫你完成的事情")
    apply: bool = Field(False, description="是否直接把建議套用並保存到 worldpack")
    rag_top_k: int = Field(6, ge=0, le=20, description="提供給代理參考的 RAG snippets 數量")
    max_new_characters: int = Field(3, ge=0, le=12, description="最多新增幾個 NPC/角色模板")
    max_new_player_templates: int = Field(1, ge=0, le=6, description="最多新增幾個玩家模板")
    include_visual: bool = Field(True, description="是否讓代理同時建議視覺風格/LoRA")


class WorldAgentSuggestResponse(BaseModel):
    """World Studio agent suggestion result."""

    success: bool = True
    applied: bool = False
    world_id: str
    patch: Dict[str, Any] = Field(default_factory=dict, description="合併後的 patch（可用於前端顯示）")
    worldpack: WorldPack
    contributors: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
