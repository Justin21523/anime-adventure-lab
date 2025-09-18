# schemas/game.py
"""
Text Adventure Game API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from .schemas_base import BaseRequest, BaseResponse, BaseParameters


class GameParameters(BaseModel):
    """Game creation parameters"""

    persona_id: str = Field(default="wise_sage", description="遊戲角色ID")
    setting: str = Field(default="fantasy", description="遊戲設定")
    difficulty: str = Field(default="medium", description="難度等級")

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


class NewGameRequest(BaseModel):
    """Create new game session request"""

    player_name: str = Field(
        ..., min_length=1, max_length=50, description="Player character name"
    )
    parameters: Optional[GameParameters] = Field(default_factory=GameParameters)  # type: ignore

    @field_validator("player_name", mode="after")
    def validate_player_name(cls, v):
        if not v.strip():
            raise ValueError("Player name cannot be empty")
        return v.strip()


class GameStepRequest(BaseModel):
    """Request to process game step"""

    session_id: str = Field(..., description="遊戲會話ID")
    player_input: str = Field(..., description="玩家輸入")
    choice_id: Optional[str] = Field(None, description="選擇ID")


class GameChoice(BaseModel):
    """Game choice option"""

    choice_id: str = Field(..., description="選擇ID")
    text: str = Field(..., description="選擇文字")
    type: str = Field(..., description="選擇類型")
    difficulty: str = Field(..., description="難度等級")
    can_choose: bool = Field(..., description="是否可選擇")


class GameResponse(BaseModel):
    """Game response with narrative and choices"""

    session_id: str = Field(..., description="遊戲會話ID")
    turn_count: int = Field(..., description="回合數")
    narrative: str = Field(..., description="故事敘述")
    choices: List[Dict[str, Any]] = Field(..., description="可選擇選項")
    stats: Dict[str, int] = Field(..., description="玩家統計")
    inventory: List[str] = Field(..., description="背包物品")
    scene_id: str = Field(..., description="場景ID")
    flags: Dict[str, bool] = Field(default_factory=dict, description="遊戲標記")
    choice_result: Optional[Dict[str, Any]] = Field(None, description="選擇結果")
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="回應訊息")


class GameSessionSummary(BaseModel):
    """Game session summary"""

    session_id: str = Field(..., description="會話ID")
    player_name: str = Field(..., description="玩家名稱")
    persona_id: Optional[str] = Field(None, description="角色ID")
    persona_name: Optional[str] = Field(None, description="角色名稱")
    turn_count: int = Field(..., description="回合數")
    created_at: str = Field(..., description="創建時間")
    updated_at: str = Field(..., description="更新時間")
    is_active: bool = Field(..., description="是否活躍")
    current_scene: Optional[str] = Field(None, description="當前場景")
    total_history: Optional[int] = Field(None, description="歷史記錄數")


class GamePersonaInfo(BaseModel):
    """Game persona information"""

    persona_id: str = Field(..., description="角色ID")
    name: str = Field(..., description="角色名稱")
    description: str = Field(..., description="角色描述")
    personality_traits: List[str] = Field(..., description="性格特徵")
    special_abilities: List[str] = Field(..., description="特殊能力")


class GameStatsResponse(BaseModel):
    """Detailed game statistics response"""

    session_id: str = Field(..., description="會話ID")
    player_name: str = Field(..., description="玩家名稱")
    stats: Dict[str, int] = Field(..., description="玩家統計")
    inventory: List[str] = Field(..., description="背包物品")
    turn_count: int = Field(..., description="回合數")
    flags: Dict[str, bool] = Field(..., description="遊戲標記")
    created_at: str = Field(..., description="創建時間")
    updated_at: str = Field(..., description="更新時間")


class GameStatsRequest(BaseModel):
    """Request for game statistics"""

    session_ids: Optional[List[str]] = Field(
        None, description="Specific session IDs to analyze"
    )
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range filter")


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
