# core/story/game_state.py
import json
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
from enum import Enum
import json
import uuid

from .story_system import GameCharacter, CharacterRole, CharacterState


@dataclass
class PlayerStats:
    """Player character statistics"""

    health: int = 100
    energy: int = 100
    intelligence: int = 10
    charisma: int = 10
    luck: int = 10
    experience: int = 0
    level: int = 1

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "PlayerStats":
        return cls(**data)


@dataclass
class GameState:
    """Current game state snapshot"""

    scene_id: Optional[str]
    scene_description: Optional[str]
    available_choices: Optional[List[Dict[str, Any]]]
    story_context: Dict[str, Any]
    flags: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameState":
        return cls(**data)


@dataclass
class GameSession:
    """Complete game session data"""

    session_id: str
    player_name: str
    persona_id: str

    created_at: datetime
    updated_at: datetime
    turn_count: int
    is_active: bool = True

    current_state: "GameState" = field(
        default_factory=lambda: GameState("", "", [], {})
    )
    stats: "PlayerStats" = field(default_factory=lambda: PlayerStats())
    inventory: List[str] = field(default_factory=list)

    history: List[Dict[str, Any]] = field(default_factory=list)

    # Character roster - 新增支持角色系統
    character_roster: Dict[str, GameCharacter] = field(default_factory=dict)

    def _initialize_default_roster(self):
        """Initialize character roster with player character and default NPCs"""

        # 清空現有角色名冊（如果有的話）
        self.character_roster = {}

        # 添加玩家角色
        self.character_roster["player"] = GameCharacter(
            character_id="player",
            name=self.player_name,
            role=CharacterRole.PLAYER,
            personality_traits=["好奇", "勇敢", "適應力強"],
            speaking_style="第一人稱視角",
            background_story=f"一個踏上冒險旅程的{self.player_name}",
            motivations=["探索未知", "成長", "幫助他人"],
            relationships={},
            current_state=CharacterState.NEUTRAL,
            current_location="起始點",
            health=100,
            mood="neutral",
            dialogue_history=[],
            interaction_count=0,
            last_seen_turn=0,
            persona_prompt="",
            content_restrictions=[],
        )

        # 添加預設NPC角色
        default_npcs = self._get_default_npcs()

        for npc in default_npcs:
            self.character_roster[npc.character_id] = npc

    def _get_default_npcs(self) -> List[GameCharacter]:
        """Get list of default NPC characters"""

        default_npcs = []

        # NPC 1: 智者嚮導
        wise_guide = GameCharacter(
            character_id="wise_guide",
            name="智者艾莉亞",
            role=CharacterRole.COMPANION,
            personality_traits=["智慧", "神秘", "指導性"],
            speaking_style="充滿智慧且神秘的語調",
            background_story="一位古老的魔法師，掌握著許多秘密知識",
            motivations=["指導後輩", "保護古老知識", "維持平衡"],
            relationships={"player": "mentor"},
            current_state=CharacterState.NEUTRAL,
            current_location="起始點",
            health=100,
            mood="wise",
            dialogue_history=[],
            interaction_count=0,
            last_seen_turn=0,
            persona_prompt="你是一位智慧的嚮導，總是給予有用的建議",
            content_restrictions=["避免透露過多劇透", "保持神秘感"],
        )
        default_npcs.append(wise_guide)

        # NPC 2: 友善商人
        merchant = GameCharacter(
            character_id="merchant_bob",
            name="商人巴布",
            role=CharacterRole.NPC,
            personality_traits=["精明", "友善", "商業頭腦"],
            speaking_style="熱情且具說服力的商業語調",
            background_story="一個經驗豐富的旅行商人，足跡遍布各地",
            motivations=["獲利", "建立人脈", "分享故事"],
            relationships={"player": "business_partner"},
            current_state=CharacterState.HAPPY,
            current_location="起始點",
            health=100,
            mood="cheerful",
            dialogue_history=[],
            interaction_count=0,
            last_seen_turn=0,
            persona_prompt="你是一個友善的商人，喜歡與人交易和聊天",
            content_restrictions=["保持商業道德", "不進行欺詐交易"],
        )
        default_npcs.append(merchant)

        # NPC 3: 神秘守護者（可選出現）
        guardian = GameCharacter(
            character_id="mysterious_guardian",
            name="守護者凱爾",
            role=CharacterRole.COMPANION,
            personality_traits=["謹慎", "保護欲", "忠誠"],
            speaking_style="嚴肅且正式的守護者語調",
            background_story="一位神秘的守護者，職責是保護重要的秘密",
            motivations=["守護職責", "保護無辜", "維護正義"],
            relationships={"player": "protector"},
            current_state=CharacterState.NEUTRAL,
            current_location="隱藏地點",
            health=120,
            mood="vigilant",
            dialogue_history=[],
            interaction_count=0,
            last_seen_turn=0,
            persona_prompt="你是一個盡職的守護者，始終將保護他人放在第一位",
            content_restrictions=["不透露守護的秘密", "保持職業操守"],
        )
        default_npcs.append(guardian)

        return default_npcs

    def add_to_history(
        self, player_input: str, ai_response: str, choice_id: Optional[str] = None
    ):
        """Add turn to history"""
        self.history.append(
            {
                "turn": self.turn_count,
                "timestamp": datetime.now().isoformat(),
                "player_input": player_input,
                "ai_response": ai_response,
                "choice_id": choice_id,
                "scene_id": self.current_state.scene_id,
            }
        )
        self.turn_count += 1
        self.updated_at = datetime.now()

    def add_character(self, character: "GameCharacter"):
        """Add a character to the roster"""
        self.character_roster[character.character_id] = character

        # 記錄添加事件
        self.history.append(
            {
                "type": "character_added",
                "character_id": character.character_id,
                "character_name": character.name,
                "timestamp": datetime.now().isoformat(),
                "turn": self.turn_count,
            }
        )

    def remove_character(self, character_id: str) -> bool:
        """Remove a character from the roster"""
        if character_id in self.character_roster:
            removed_char = self.character_roster.pop(character_id)

            # 記錄移除事件
            self.history.append(
                {
                    "type": "character_removed",
                    "character_id": character_id,
                    "character_name": removed_char.name,
                    "timestamp": datetime.now().isoformat(),
                    "turn": self.turn_count,
                }
            )

            return True
        return False

    def get_character(self, character_id: str) -> Optional["GameCharacter"]:
        """Get a character from the roster"""
        return self.character_roster.get(character_id)

    def get_active_characters(self) -> List["GameCharacter"]:
        """Get list of active characters (health > 0)"""
        return [char for char in self.character_roster.values() if char.health > 0]

    def get_characters_by_role(self, role: CharacterRole) -> List["GameCharacter"]:
        """Get characters by their role"""
        return [char for char in self.character_roster.values() if char.role == role]

    def update_character_relationship(
        self, char1_id: str, char2_id: str, relationship_type: str
    ):
        """Update relationship between two characters"""
        if char1_id in self.character_roster:
            self.character_roster[char1_id].relationships[char2_id] = relationship_type

        if char2_id in self.character_roster:
            # 設置反向關係（可能不同）
            reverse_relationships = {
                "friend": "friend",
                "enemy": "enemy",
                "mentor": "student",
                "student": "mentor",
                "business_partner": "business_partner",
                "protector": "protected",
                "protected": "protector",
            }
            reverse_rel = reverse_relationships.get(
                relationship_type, relationship_type
            )
            self.character_roster[char2_id].relationships[char1_id] = reverse_rel

    def __post_init__(self):
        """Initialize character roster with player character if empty"""
        if not self.character_roster:
            self._initialize_default_roster()

    # 保持原有的其他方法...
    def to_dict(
        self,
        scene_id: str = "",
        scene_description: str = "",
        available_choices: List[str] = None,  # type: ignore
        story_context: Dict[str, Any] = None,  # type: ignore
    ) -> Dict[str, Any]:
        """Convert session to dictionary with all required parameters"""

        if available_choices is None:
            available_choices = []
        if story_context is None:
            story_context = {}

        # 基本會話數據
        data = {
            "session_id": self.session_id,
            "player_name": self.player_name,
            "persona_id": self.persona_id,
            "created_at": (
                self.created_at.isoformat()
                if isinstance(self.created_at, datetime)
                else str(self.created_at)
            ),
            "updated_at": (
                self.updated_at.isoformat()
                if isinstance(self.updated_at, datetime)
                else str(self.updated_at)
            ),
            "turn_count": self.turn_count,
            "is_active": self.is_active,
            "inventory": self.inventory.copy(),
            "history": self.history.copy(),
            # 必要的場景參數
            "scene_id": scene_id,
            "scene_description": scene_description,
            "available_choices": available_choices.copy(),
            "story_context": story_context.copy(),
            # 遊戲狀態
            "current_state": (
                self.current_state.to_dict()
                if hasattr(self.current_state, "to_dict")
                else {
                    "scene_id": scene_id,
                    "story_context": story_context,
                    "flags": getattr(self.current_state, "flags", {}),
                }
            ),
            # 玩家數據
            "stats": (
                self.stats.to_dict()
                if hasattr(self.stats, "to_dict")
                else {
                    "health": getattr(self.stats, "health", 100),
                    "mana": getattr(self.stats, "mana", 50),
                    "strength": getattr(self.stats, "strength", 10),
                    "intelligence": getattr(self.stats, "intelligence", 10),
                    "charisma": getattr(self.stats, "charisma", 10),
                    "luck": getattr(self.stats, "luck", 10),
                }
            ),
        }

        # 角色名冊序列化
        if self.character_roster:
            data["character_roster"] = {}
            for char_id, character in self.character_roster.items():
                if hasattr(character, "to_dict"):
                    data["character_roster"][char_id] = character.to_dict()
                else:
                    # 簡化的角色數據
                    data["character_roster"][char_id] = {
                        "character_id": getattr(character, "character_id", char_id),
                        "name": getattr(character, "name", char_id),
                        "role": getattr(character, "role", "unknown"),
                        "current_state": getattr(character, "current_state", "neutral"),
                        "health": getattr(character, "health", 100),
                        "current_location": getattr(
                            character, "current_location", "unknown"
                        ),
                    }

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameSession":
        """Create GameSession from dictionary data"""
        # 轉換時間字段
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if isinstance(data["created_at"], str)
            else data["created_at"]
        )
        updated_at = (
            datetime.fromisoformat(data["updated_at"])
            if isinstance(data["updated_at"], str)
            else data["updated_at"]
        )

        # 重建 GameState
        current_state = GameState(
            scene_id=data.get("scene_id", "scene_001"),
            scene_description="",
            available_choices=[],
            story_context=data.get("story_context", {}),
            flags=data.get("current_state", {}).get("flags", {}),
        )

        # 重建 PlayerStats
        stats_data = data.get("stats", {})
        stats = PlayerStats(
            health=stats_data.get("health", 100),
            energy=stats_data.get("energy", 50),
            experience=stats_data.get("experience", 10),
            intelligence=stats_data.get("intelligence", 10),
            charisma=stats_data.get("charisma", 10),
            luck=stats_data.get("luck", 10),
        )

        # 創建會話實例
        session = cls(
            session_id=data["session_id"],
            player_name=data["player_name"],
            persona_id=data["persona_id"],
            created_at=created_at,
            updated_at=updated_at,
            turn_count=data.get("turn_count", 0),
            is_active=data.get("is_active", True),
            current_state=current_state,
            stats=stats,
            inventory=data.get("inventory", []),
            history=data.get("history", []),
        )

        # 重建角色名冊
        character_roster_data = data.get("character_roster", {})
        for char_id, char_data in character_roster_data.items():
            try:
                # 重建角色實例
                character = GameCharacter(
                    character_id=char_data.get("character_id", char_id),
                    name=char_data.get("name", char_id),
                    role=CharacterRole(char_data.get("role", "npc")),
                    personality_traits=char_data.get("personality_traits", []),
                    speaking_style=char_data.get("speaking_style", "普通語調"),
                    background_story=char_data.get("background_story", "神秘的背景"),
                    motivations=char_data.get("motivations", []),
                    relationships=char_data.get("relationships", {}),
                    current_state=CharacterState(
                        char_data.get("current_state", "neutral")
                    ),
                    current_location=char_data.get("current_location", "unknown"),
                    health=char_data.get("health", 100),
                    mood=char_data.get("mood", "neutral"),
                )
                session.character_roster[char_id] = character

            except Exception as e:
                # 如果角色重建失敗，跳過該角色
                print(f"Warning: Failed to rebuild character {char_id}: {e}")
                continue

        return session
