# =============================================================================
# core/story/persona.py
"""
Game Persona Management
Handles character personalities, worldviews, and behavior rules
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GamePersona:
    """Game character persona definition"""

    persona_id: str
    name: str
    description: str
    worldview: str
    personality_traits: List[str]
    speaking_style: str
    backstory: str
    goals: List[str]
    fears: List[str]
    relationships: Dict[str, str] = field(default_factory=dict)
    special_abilities: List[str] = field(default_factory=list)
    content_restrictions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "description": self.description,
            "worldview": self.worldview,
            "personality_traits": self.personality_traits,
            "speaking_style": self.speaking_style,
            "backstory": self.backstory,
            "goals": self.goals,
            "fears": self.fears,
            "relationships": self.relationships,
            "special_abilities": self.special_abilities,
            "content_restrictions": self.content_restrictions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GamePersona":
        return cls(**data)


class PersonaManager:
    """Manages game personas and their behaviors"""

    def __init__(self, config_path: Optional[Path] = None):
        self.personas: Dict[str, GamePersona] = {}
        self.config_path = config_path
        self._load_personas()

    def _load_personas(self):
        """Load personas from configuration"""
        if not self.config_path or not self.config_path.exists():
            # Load default personas
            self._load_default_personas()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for persona_data in data.get("personas", []):
                    persona = GamePersona.from_dict(persona_data)
                    self.personas[persona.persona_id] = persona
            logger.info(f"Loaded {len(self.personas)} personas from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load personas: {e}")
            self._load_default_personas()

    def _load_default_personas(self):
        """Load default built-in personas"""
        default_personas = [
            {
                "persona_id": "wise_sage",
                "name": "智慧賢者",
                "description": "一位充滿智慧的古老賢者，總是以深刻的洞察力指引冒險者",
                "worldview": "世界充滿奧秘，每個選擇都有其深遠的意義",
                "personality_traits": ["睿智", "神秘", "耐心", "幽默"],
                "speaking_style": "用充滿哲理的語言，時常引用古老的諺語和比喻",
                "backstory": "曾經歷過無數的冒險，現在致力於指導年輕的冒險者",
                "goals": ["傳授智慧", "保護世界平衡", "培養下一代英雄"],
                "fears": ["黑暗勢力的復甦", "智慧的遺失", "年輕人的迷失"],
                "relationships": {"學生們": "慈愛的導師", "黑暗勢力": "永恆的對手"},
                "special_abilities": ["預知危險", "古老魔法", "心靈感應"],
                "content_restrictions": [
                    "避免暴力描述",
                    "禁止不當內容",
                    "保持正面價值觀",
                ],
            },
            {
                "persona_id": "mischievous_fairy",
                "name": "頑皮精靈",
                "description": "一個喜歡惡作劇但心地善良的森林精靈",
                "worldview": "生活應該充滿樂趣和驚喜，但要記住責任",
                "personality_traits": ["頑皮", "好奇", "善良", "活潑"],
                "speaking_style": "輕快活潑，經常使用俏皮話和雙關語",
                "backstory": "在古老森林中長大，對人類世界充滿好奇",
                "goals": ["帶來歡樂", "保護森林", "結交朋友"],
                "fears": ["孤獨", "森林被破壞", "失去魔法"],
                "relationships": {"森林動物": "最好的朋友", "人類": "有趣的玩伴"},
                "special_abilities": ["隱身術", "自然魔法", "動物交流"],
                "content_restrictions": ["保持輕鬆氛圍", "避免恐怖元素", "維持童真感"],
            },
        ]

        for persona_data in default_personas:
            persona = GamePersona.from_dict(persona_data)
            self.personas[persona.persona_id] = persona

        logger.info(f"Loaded {len(default_personas)} default personas")

    def get_persona(self, persona_id: str) -> Optional[GamePersona]:
        """Get persona by ID"""
        return self.personas.get(persona_id)

    def list_personas(self) -> List[GamePersona]:
        """List all available personas"""
        return list(self.personas.values())

    def add_persona(self, persona: GamePersona):
        """Add or update a persona"""
        self.personas[persona.persona_id] = persona

    def generate_persona_prompt(
        self, persona: GamePersona, context: Dict[str, Any]
    ) -> str:
        """Generate system prompt for persona"""
        prompt = f"""你是{persona.name}，{persona.description}

人物設定：
- 世界觀：{persona.worldview}
- 性格特徵：{', '.join(persona.personality_traits)}
- 說話風格：{persona.speaking_style}
- 背景故事：{persona.backstory}
- 目標：{', '.join(persona.goals)}
- 恐懼：{', '.join(persona.fears)}

特殊能力：{', '.join(persona.special_abilities)}

內容限制：
{chr(10).join(f'- {restriction}' for restriction in persona.content_restrictions)}

請以這個角色的身份與玩家互動，創造引人入勝的文字冒險體驗。
保持角色一致性，根據玩家的選擇推進故事發展。"""

        return prompt
