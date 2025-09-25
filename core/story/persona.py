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
from enum import Enum

logger = logging.getLogger(__name__)


class PersonaType(Enum):
    """Types of personas in the game"""

    NARRATOR = "narrator"
    NPC = "npc"
    COMPANION = "companion"
    ANTAGONIST = "antagonist"
    MENTOR = "mentor"
    MERCHANT = "merchant"
    GUIDE = "guide"


class EmotionalState(Enum):
    """Emotional states for dynamic persona behavior"""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    FEARFUL = "fearful"
    SUSPICIOUS = "suspicious"
    CONFIDENT = "confident"
    WORRIED = "worried"
    AMUSED = "amused"
    GRATEFUL = "grateful"


@dataclass
class GamePersona:
    """Game character persona definition"""

    persona_id: str
    name: str
    description: str
    persona_type: PersonaType

    # Core personality
    worldview: str
    personality_traits: List[str]
    speaking_style: str
    backstory: str
    goals: List[str]
    fears: List[str]

    # Relationships and social aspects
    relationships: Dict[str, str] = field(default_factory=dict)
    relationship_attitudes: Dict[str, str] = field(
        default_factory=dict
    )  # How they view different types
    social_preferences: List[str] = field(default_factory=list)

    # Abilities and knowledge
    special_abilities: List[str] = field(default_factory=list)
    knowledge_areas: List[str] = field(default_factory=list)
    skills: Dict[str, int] = field(
        default_factory=dict
    )  # skill_name: proficiency_level

    # Dynamic aspects
    current_emotional_state: EmotionalState = EmotionalState.NEUTRAL
    emotional_triggers: Dict[str, EmotionalState] = field(default_factory=dict)
    adaptability_level: int = 5  # 1-10, how much they change based on player actions

    # Behavioral patterns
    decision_making_style: str = "balanced"  # logical, emotional, impulsive, careful
    communication_patterns: Dict[str, Any] = field(default_factory=dict)
    moral_alignment: str = "neutral"  # good, neutral, evil, chaotic, lawful

    # Content and safety
    content_restrictions: List[str] = field(default_factory=list)
    preferred_topics: List[str] = field(default_factory=list)
    avoided_topics: List[str] = field(default_factory=list)

    # Memory and learning
    memory_strength: int = 7  # 1-10, how well they remember interactions
    learning_rate: int = 5  # 1-10, how quickly they adapt to player behavior

    # Game mechanics
    dialogue_frequency: float = 0.7  # 0-1, how often they speak
    story_importance: int = 5  # 1-10, how central they are to the story

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary"""
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "description": self.description,
            "persona_type": self.persona_type.value,
            "worldview": self.worldview,
            "personality_traits": self.personality_traits,
            "speaking_style": self.speaking_style,
            "backstory": self.backstory,
            "goals": self.goals,
            "fears": self.fears,
            "relationships": self.relationships,
            "relationship_attitudes": self.relationship_attitudes,
            "social_preferences": self.social_preferences,
            "special_abilities": self.special_abilities,
            "knowledge_areas": self.knowledge_areas,
            "skills": self.skills,
            "current_emotional_state": self.current_emotional_state.value,
            "emotional_triggers": {
                k: v.value for k, v in self.emotional_triggers.items()
            },
            "adaptability_level": self.adaptability_level,
            "decision_making_style": self.decision_making_style,
            "communication_patterns": self.communication_patterns,
            "moral_alignment": self.moral_alignment,
            "content_restrictions": self.content_restrictions,
            "preferred_topics": self.preferred_topics,
            "avoided_topics": self.avoided_topics,
            "memory_strength": self.memory_strength,
            "learning_rate": self.learning_rate,
            "dialogue_frequency": self.dialogue_frequency,
            "story_importance": self.story_importance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GamePersona":
        """Create persona from dictionary"""
        # Convert enum fields
        persona_type = PersonaType(data.get("persona_type", "npc"))
        emotional_state = EmotionalState(data.get("current_emotional_state", "neutral"))
        emotional_triggers = {
            k: EmotionalState(v) for k, v in data.get("emotional_triggers", {}).items()
        }

        return cls(
            persona_id=data["persona_id"],
            name=data["name"],
            description=data["description"],
            persona_type=persona_type,
            worldview=data["worldview"],
            personality_traits=data["personality_traits"],
            speaking_style=data["speaking_style"],
            backstory=data["backstory"],
            goals=data["goals"],
            fears=data["fears"],
            relationships=data.get("relationships", {}),
            relationship_attitudes=data.get("relationship_attitudes", {}),
            social_preferences=data.get("social_preferences", []),
            special_abilities=data.get("special_abilities", []),
            knowledge_areas=data.get("knowledge_areas", []),
            skills=data.get("skills", {}),
            current_emotional_state=emotional_state,
            emotional_triggers=emotional_triggers,
            adaptability_level=data.get("adaptability_level", 5),
            decision_making_style=data.get("decision_making_style", "balanced"),
            communication_patterns=data.get("communication_patterns", {}),
            moral_alignment=data.get("moral_alignment", "neutral"),
            content_restrictions=data.get("content_restrictions", []),
            preferred_topics=data.get("preferred_topics", []),
            avoided_topics=data.get("avoided_topics", []),
            memory_strength=data.get("memory_strength", 7),
            learning_rate=data.get("learning_rate", 5),
            dialogue_frequency=data.get("dialogue_frequency", 0.7),
            story_importance=data.get("story_importance", 5),
        )

    def update_emotional_state(
        self, trigger: str, player_relationship: int = 0
    ) -> EmotionalState:
        """Update emotional state based on trigger and relationship"""
        # Check for specific emotional triggers
        if trigger in self.emotional_triggers:
            new_state = self.emotional_triggers[trigger]
        else:
            # Default emotional responses based on common triggers
            trigger_responses = {
                "player_friendly": EmotionalState.HAPPY,
                "player_hostile": EmotionalState.ANGRY,
                "player_helpful": (
                    EmotionalState.GRATEFUL
                    if hasattr(EmotionalState, "GRATEFUL")
                    else EmotionalState.HAPPY
                ),
                "player_dismissive": EmotionalState.SAD,
                "combat_started": EmotionalState.FEARFUL,
                "mystery_discovered": EmotionalState.EXCITED,
                "goal_achieved": EmotionalState.HAPPY,
                "goal_blocked": EmotionalState.WORRIED,
                "betrayal": EmotionalState.ANGRY,
                "surprise": EmotionalState.EXCITED,
            }
            new_state = trigger_responses.get(trigger, self.current_emotional_state)

        # Modify based on adaptability and relationship
        if self.adaptability_level > 5 and player_relationship > 5:
            # High adaptability + good relationship = more positive emotions
            if new_state == EmotionalState.ANGRY:
                new_state = EmotionalState.WORRIED
            elif new_state == EmotionalState.FEARFUL:
                new_state = EmotionalState.SUSPICIOUS
        elif player_relationship < -3:
            # Bad relationship = more negative emotions
            if new_state == EmotionalState.HAPPY:
                new_state = EmotionalState.SUSPICIOUS

        self.current_emotional_state = new_state
        return new_state

    def get_dialogue_style_modifiers(self) -> Dict[str, str]:
        """Get dialogue style modifiers based on current state"""
        emotional_modifiers = {
            EmotionalState.HAPPY: "用愉快和友善的語氣",
            EmotionalState.SAD: "帶著一絲憂傷和低沉的語調",
            EmotionalState.ANGRY: "語氣略顯急躁和直接",
            EmotionalState.EXCITED: "充滿熱情和活力地",
            EmotionalState.FEARFUL: "小心翼翼且略顯緊張地",
            EmotionalState.SUSPICIOUS: "帶著謹慎和懷疑的態度",
            EmotionalState.CONFIDENT: "自信而堅定地",
            EmotionalState.WORRIED: "帶著擔憂和關切地",
            EmotionalState.AMUSED: "略帶幽默和輕鬆地",
        }

        base_modifier = emotional_modifiers.get(self.current_emotional_state, "")

        # Add personality-based modifiers
        personality_modifiers = []
        if "幽默" in self.personality_traits:
            personality_modifiers.append("偶爾開玩笑")
        if "智慧" in self.personality_traits:
            personality_modifiers.append("引用智慧名言")
        if "神秘" in self.personality_traits:
            personality_modifiers.append("話中有話")
        if "直率" in self.personality_traits:
            personality_modifiers.append("直接表達想法")

        return {
            "emotional_tone": base_modifier,
            "personality_elements": personality_modifiers,  # type: ignore
            "speaking_style": self.speaking_style,
        }


class PersonaManager:
    """Manages game personas and their behaviors"""

    def __init__(self, personas_file: Optional[Path] = None):
        self.personas: Dict[str, GamePersona] = {}
        self.relationship_matrix: Dict[str, Dict[str, int]] = (
            {}
        )  # persona_id -> persona_id -> relationship_score
        self.interaction_history: Dict[str, List[Dict[str, Any]]] = (
            {}
        )  # persona_id -> interaction_list
        self.config_path = personas_file  # Store config path for later use

        if personas_file and personas_file.exists():
            self._load_personas_from_file(personas_file)
        else:
            self._create_default_personas()

        logger.info(f"PersonaManager initialized with {len(self.personas)} personas")

    def _load_personas_from_file(self, personas_file: Path):
        """Load personas from JSON file"""
        try:
            with open(personas_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            personas_data = data.get("personas", [])
            for persona_data in personas_data:
                persona = GamePersona.from_dict(persona_data)
                self.personas[persona.persona_id] = persona

            # Load relationship matrix if available
            if "relationships" in data:
                self.relationship_matrix = data["relationships"]

            # Load interaction history if available
            if "interaction_history" in data:
                self.interaction_history = data["interaction_history"]

            logger.info(f"Loaded {len(personas_data)} personas from {personas_file}")

        except Exception as e:
            logger.error(f"Failed to load personas from {personas_file}: {e}")
            self._create_default_personas()

    def _create_default_personas(self):
        """Create enhanced default personas"""
        default_personas = [
            {
                "persona_id": "wise_sage",
                "name": "智者導師",
                "description": "一位博學而慈祥的智者，擁有豐富的人生閱歷",
                "persona_type": "mentor",
                "worldview": "世界充滿學習機會，每個人都有成長的潛力",
                "personality_traits": ["博學", "耐心", "慈祥", "富有洞察力", "幽默"],
                "speaking_style": "溫和而富有智慧，常用比喻和典故來說明道理",
                "backstory": "曾經是知名學者，現在致力於引導年輕冒險者找到正確的道路",
                "goals": ["傳承知識", "保護年輕人", "維護世界和平", "促進理解"],
                "fears": ["知識的失傳", "年輕人走上歧途", "世界陷入混亂"],
                "relationship_attitudes": {
                    "student": "耐心教導",
                    "scholar": "平等交流",
                    "warrior": "欣賞其勇氣",
                    "criminal": "試圖感化",
                },
                "knowledge_areas": ["歷史", "哲學", "魔法理論", "人性洞察"],
                "skills": {
                    "teaching": 9,
                    "magic_theory": 8,
                    "diplomacy": 7,
                    "healing": 6,
                },
                "emotional_triggers": {
                    "student_success": "happy",
                    "knowledge_questioned": "worried",
                    "violence": "sad",
                    "discovery": "excited",
                },
                "adaptability_level": 8,
                "decision_making_style": "careful",
                "moral_alignment": "good",
                "content_restrictions": [
                    "避免暴力描述",
                    "保持正面導向",
                    "不提供危險知識",
                ],
                "preferred_topics": ["學習", "智慧", "成長", "和平"],
                "avoided_topics": ["暴力細節", "黑暗魔法", "絕望"],
                "memory_strength": 9,
                "learning_rate": 6,
                "dialogue_frequency": 0.8,
                "story_importance": 8,
            },
            {
                "persona_id": "mysterious_guide",
                "name": "神秘嚮導",
                "description": "來歷不明的神秘人物，似乎對這個世界瞭若指掌",
                "persona_type": "guide",
                "worldview": "世界充滿秘密，只有經過考驗的人才能獲得真相",
                "personality_traits": ["神秘", "機智", "略帶幽默", "謹慎", "觀察敏銳"],
                "speaking_style": "簡潔有力，偶爾開玩笑，話中常有深意",
                "backstory": "身份成謎的旅行者，似乎經歷過許多冒險",
                "goals": ["完成某個秘密任務", "測試玩家的能力", "保護重要秘密"],
                "fears": ["身份暴露", "任務失敗", "信任被背叛"],
                "relationship_attitudes": {
                    "curious": "欣賞好奇心",
                    "rash": "提醒要謹慎",
                    "wise": "認可其智慧",
                    "hostile": "保持距離",
                },
                "knowledge_areas": ["地理", "歷史秘辛", "生存技巧", "人心理解"],
                "skills": {
                    "stealth": 9,
                    "navigation": 8,
                    "observation": 9,
                    "survival": 7,
                },
                "emotional_triggers": {
                    "trust_shown": "happy",
                    "betrayal": "angry",
                    "curiosity": "amused",
                    "recklessness": "worried",
                },
                "adaptability_level": 7,
                "decision_making_style": "careful",
                "moral_alignment": "neutral",
                "content_restrictions": ["不洩露關鍵秘密", "保持神秘感"],
                "preferred_topics": ["探索", "謎團", "試煉", "成長"],
                "avoided_topics": ["個人身份", "具體任務", "某些禁忌知識"],
                "memory_strength": 10,
                "learning_rate": 8,
                "dialogue_frequency": 0.6,
                "story_importance": 7,
            },
            {
                "persona_id": "cheerful_companion",
                "name": "開朗夥伴",
                "description": "充滿活力的年輕冒險者，總是保持樂觀態度",
                "persona_type": "companion",
                "worldview": "世界美好，冒險充滿樂趣，朋友最重要",
                "personality_traits": ["樂觀", "忠誠", "勇敢", "有趣", "善良"],
                "speaking_style": "活潑開朗，用詞輕鬆，經常鼓勵他人",
                "backstory": "來自小村莊的年輕人，渴望看到更廣闊的世界",
                "goals": ["幫助朋友", "體驗冒險", "保護無辜", "成為英雄"],
                "fears": ["失去朋友", "讓人失望", "孤獨", "邪惡獲勝"],
                "relationship_attitudes": {
                    "friendly": "立即親近",
                    "sad": "想要幫助",
                    "grumpy": "試圖逗樂",
                    "evil": "堅決對抗",
                },
                "knowledge_areas": ["村莊生活", "基礎戰鬥", "野外生存", "民間故事"],
                "skills": {
                    "combat": 6,
                    "encouragement": 8,
                    "survival": 5,
                    "friendship": 9,
                },
                "emotional_triggers": {
                    "friend_happy": "excited",
                    "friend_hurt": "worried",
                    "victory": "happy",
                    "injustice": "angry",
                },
                "adaptability_level": 9,
                "decision_making_style": "emotional",
                "moral_alignment": "good",
                "content_restrictions": [
                    "保持正面積極",
                    "避免過度暴力",
                    "維護友誼價值",
                ],
                "preferred_topics": ["冒險", "友誼", "希望", "英雄故事"],
                "avoided_topics": ["絕望", "背叛", "殘酷現實"],
                "memory_strength": 7,
                "learning_rate": 8,
                "dialogue_frequency": 0.9,
                "story_importance": 6,
            },
            {
                "persona_id": "gruff_merchant",
                "name": "粗獷商人",
                "description": "經驗豐富的商人，外表粗獷但內心善良",
                "persona_type": "merchant",
                "worldview": "世界運轉靠交易，誠信比金錢更重要",
                "personality_traits": ["實用主義", "誠實", "精明", "粗獷", "關愛"],
                "speaking_style": "直接粗獷，但有商人的精明和偶爾的溫暖",
                "backstory": "走遍各地的老商人，見過世間百態",
                "goals": ["公平交易", "保護商路", "幫助需要的人", "維持生計"],
                "fears": ["欺詐", "商路不安全", "失去信譽", "孤獨終老"],
                "relationship_attitudes": {
                    "honest": "信任交易",
                    "dishonest": "警惕提防",
                    "needy": "願意幫助",
                    "wealthy": "平等對待",
                },
                "knowledge_areas": ["貿易", "各地風俗", "物品價值", "人性判斷"],
                "skills": {
                    "trading": 9,
                    "appraisal": 8,
                    "negotiation": 7,
                    "survival": 6,
                },
                "emotional_triggers": {
                    "fair_deal": "happy",
                    "cheating": "angry",
                    "helping_others": "satisfied",
                    "loneliness": "sad",
                },
                "adaptability_level": 5,
                "decision_making_style": "logical",
                "moral_alignment": "neutral",
                "content_restrictions": ["保持商業道德", "避免極端立場"],
                "preferred_topics": ["貿易", "各地見聞", "實用建議"],
                "avoided_topics": ["個人隱私", "政治立場"],
                "memory_strength": 8,
                "learning_rate": 4,
                "dialogue_frequency": 0.5,
                "story_importance": 4,
            },
            {
                "persona_id": "noble_knight",
                "name": "高貴騎士",
                "description": "堅持騎士精神的勇敢戰士，以保護弱者為己任",
                "persona_type": "companion",
                "worldview": "正義必勝，弱者應受保護，榮譽比生命重要",
                "personality_traits": ["勇敢", "榮譽感", "正直", "堅定", "謙遜"],
                "speaking_style": "正式而禮貌，充滿騎士風範，用詞考究",
                "backstory": "出身騎士世家，接受嚴格的騎士訓練",
                "goals": ["維護正義", "保護無辜", "實踐騎士精神", "贖回榮譽"],
                "fears": ["失去榮譽", "無法保護他人", "背叛理想", "邪惡獲勝"],
                "relationship_attitudes": {
                    "honorable": "深表敬意",
                    "evil": "堅決對抗",
                    "innocent": "誓死保護",
                    "cowardly": "試圖激勵",
                },
                "knowledge_areas": ["戰鬥技巧", "騎士法典", "戰術", "領導力"],
                "skills": {"combat": 9, "leadership": 7, "tactics": 8, "honor": 10},
                "emotional_triggers": {
                    "injustice": "angry",
                    "noble_deed": "proud",
                    "failure": "sad",
                    "victory": "satisfied",
                },
                "adaptability_level": 3,
                "decision_making_style": "principled",
                "moral_alignment": "lawful good",
                "content_restrictions": [
                    "維護騎士形象",
                    "避免不當行為",
                    "堅持道德標準",
                ],
                "preferred_topics": ["正義", "榮譽", "保護", "英勇事蹟"],
                "avoided_topics": ["不道德行為", "投機取巧", "背叛"],
                "memory_strength": 8,
                "learning_rate": 3,
                "dialogue_frequency": 0.7,
                "story_importance": 7,
            },
        ]

        for persona_data in default_personas:
            persona = GamePersona.from_dict(persona_data)
            self.personas[persona.persona_id] = persona

        logger.info(f"Created {len(default_personas)} default personas")

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
        if persona.persona_id not in self.relationship_matrix:
            self.relationship_matrix[persona.persona_id] = {}

    def remove_persona(self, persona_id: str) -> bool:
        """Remove a persona"""
        if persona_id in self.personas:
            del self.personas[persona_id]
            if persona_id in self.relationship_matrix:
                del self.relationship_matrix[persona_id]
            # Remove from other personas' relationship matrices
            for relationships in self.relationship_matrix.values():
                if persona_id in relationships:
                    del relationships[persona_id]
            return True
        return False

    def update_relationship(self, persona1_id: str, persona2_id: str, change: int):
        """Update relationship between two personas"""
        if persona1_id not in self.relationship_matrix:
            self.relationship_matrix[persona1_id] = {}

        current_relationship = self.relationship_matrix[persona1_id].get(persona2_id, 0)
        new_relationship = max(-10, min(10, current_relationship + change))
        self.relationship_matrix[persona1_id][persona2_id] = new_relationship

        logger.debug(
            f"Updated relationship {persona1_id} -> {persona2_id}: {new_relationship}"
        )

    def get_relationship(self, persona1_id: str, persona2_id: str) -> int:
        """Get relationship score between two personas"""
        return self.relationship_matrix.get(persona1_id, {}).get(persona2_id, 0)

    def record_interaction(
        self, persona_id: str, interaction_type: str, context: Dict[str, Any]
    ):
        """Record an interaction for learning and adaptation"""
        if persona_id not in self.interaction_history:
            self.interaction_history[persona_id] = []

        interaction = {
            "type": interaction_type,
            "context": context,
            "timestamp": context.get("timestamp", "unknown"),
            "player_relationship": context.get("player_relationship", 0),
        }

        self.interaction_history[persona_id].append(interaction)

        # Keep only recent interactions (last 50)
        if len(self.interaction_history[persona_id]) > 50:
            self.interaction_history[persona_id] = self.interaction_history[persona_id][
                -50:
            ]

    def generate_persona_prompt(
        self,
        persona: GamePersona,
        context: Dict[str, Any],
        player_relationship: int = 0,
    ) -> str:
        """Generate enhanced system prompt for persona"""

        # Get dialogue style modifiers based on current emotional state
        style_modifiers = persona.get_dialogue_style_modifiers()

        # Build relationship context
        relationship_context = ""
        if player_relationship != 0:
            if player_relationship > 5:
                relationship_context = f"你對玩家懷有好感（關係值: +{player_relationship}），會更加友善和幫助。"
            elif player_relationship < -3:
                relationship_context = f"你對玩家有些不信任（關係值: {player_relationship}），會更加謹慎和冷淡。"
            else:
                relationship_context = (
                    f"你對玩家保持中性態度（關係值: {player_relationship}）。"
                )

        # Build emotional context
        emotional_context = f"當前情緒狀態：{persona.current_emotional_state.value}，{style_modifiers['emotional_tone']}"

        # Build knowledge context
        knowledge_context = ""
        if persona.knowledge_areas:
            knowledge_context = f"專業領域：{', '.join(persona.knowledge_areas[:3])}"

        # Build recent interaction context
        interaction_context = ""
        if persona.persona_id in self.interaction_history:
            recent_interactions = self.interaction_history[persona.persona_id][-3:]
            if recent_interactions:
                interaction_summary = []
                for interaction in recent_interactions:
                    interaction_summary.append(
                        f"{interaction['type']}: {interaction['context'].get('summary', '一般互動')}"
                    )
                interaction_context = f"最近互動：{'; '.join(interaction_summary)}"

        prompt = f"""你是{persona.name}，{persona.description}

【核心設定】
- 人物類型：{persona.persona_type.value}
- 世界觀：{persona.worldview}
- 性格特徵：{', '.join(persona.personality_traits)}
- 說話風格：{persona.speaking_style}
- 背景故事：{persona.backstory}

【目標與動機】
- 主要目標：{', '.join(persona.goals)}
- 內心恐懼：{', '.join(persona.fears)}
- 擅長領域：{knowledge_context}

【當前狀態】
- {emotional_context}
- {relationship_context}
- {interaction_context if interaction_context else '這是初次互動'}

【行為指導】
- 決策風格：{persona.decision_making_style}
- 道德取向：{persona.moral_alignment}
- 適應能力：{persona.adaptability_level}/10（數字越高越容易改變態度）
- 記憶能力：{persona.memory_strength}/10（會記住之前的互動）

【對話要求】
1. {style_modifiers['emotional_tone']}
2. 體現角色的{persona.speaking_style}
3. 根據當前情緒狀態調整語氣和態度
4. 考慮與玩家的關係狀況
5. 保持角色一致性，但允許適度的情感變化
6. 推進對話或故事發展
7. 避免提及：{', '.join(persona.content_restrictions) if persona.content_restrictions else '無特殊限制'}

請以這個角色的身份與玩家互動，創造引人入勝的對話體驗。"""

        return prompt

    def adapt_persona_based_on_interaction(
        self,
        persona_id: str,
        interaction_result: Dict[str, Any],
        player_relationship: int,
    ):
        """Adapt persona based on interaction results"""
        persona = self.get_persona(persona_id)
        if not persona:
            return

        # Update emotional state based on interaction
        interaction_type = interaction_result.get("type", "general")
        persona.update_emotional_state(interaction_type, player_relationship)

        # Learn from interaction if adaptability is high
        if persona.adaptability_level > 6 and persona.learning_rate > 5:
            # Adjust personality traits slightly based on successful interactions
            if interaction_result.get("success", False) and player_relationship > 3:
                # Reinforce current traits
                if (
                    "friendly" in interaction_result.get("tags", [])
                    and "友善" not in persona.personality_traits
                ):
                    if len(persona.personality_traits) < 6:  # Limit trait growth
                        persona.personality_traits.append("友善")

        # Record the interaction for future reference
        self.record_interaction(
            persona_id,
            interaction_type,
            {
                "success": interaction_result.get("success", False),
                "player_relationship": player_relationship,
                "summary": interaction_result.get("summary", ""),
                "timestamp": interaction_result.get("timestamp", ""),
            },
        )

    def get_persona_compatibility(self, persona1_id: str, persona2_id: str) -> float:
        """Calculate compatibility between two personas"""
        persona1 = self.get_persona(persona1_id)
        persona2 = self.get_persona(persona2_id)

        if not persona1 or not persona2:
            return 0.0

        compatibility = 0.0

        # Check personality trait compatibility
        common_traits = set(persona1.personality_traits) & set(
            persona2.personality_traits
        )
        compatibility += len(common_traits) * 0.1

        # Check moral alignment compatibility
        alignment_compatibility = {
            ("good", "good"): 0.3,
            ("good", "neutral"): 0.1,
            ("good", "evil"): -0.3,
            ("neutral", "neutral"): 0.2,
            ("neutral", "evil"): 0.1,
            ("evil", "evil"): 0.3,
        }

        alignment_key = (persona1.moral_alignment, persona2.moral_alignment)
        if alignment_key in alignment_compatibility:
            compatibility += alignment_compatibility[alignment_key]

        # Check for complementary skills
        common_knowledge = set(persona1.knowledge_areas) & set(persona2.knowledge_areas)
        compatibility += len(common_knowledge) * 0.05

        # Check relationship history
        existing_relationship = self.get_relationship(persona1_id, persona2_id)
        compatibility += existing_relationship * 0.02

        return max(-1.0, min(1.0, compatibility))

    def suggest_interaction_approaches(
        self, persona_id: str, player_relationship: int
    ) -> List[Dict[str, str]]:
        """Suggest approaches for interacting with a persona"""
        persona = self.get_persona(persona_id)
        if not persona:
            return []

        suggestions = []

        # Based on personality traits
        if "友善" in persona.personality_traits or "善良" in persona.personality_traits:
            suggestions.append(
                {
                    "approach": "friendly",
                    "description": "友善接近",
                    "success_chance": "高",
                }
            )

        if "智慧" in persona.personality_traits or "博學" in persona.personality_traits:
            suggestions.append(
                {
                    "approach": "intellectual",
                    "description": "學術討論",
                    "success_chance": "中高",
                }
            )

        if "謹慎" in persona.personality_traits or "suspicious" in [
            t.lower() for t in persona.personality_traits
        ]:
            suggestions.append(
                {
                    "approach": "careful",
                    "description": "謹慎接近",
                    "success_chance": "中",
                }
            )

        # Based on current emotional state
        emotional_approaches = {
            EmotionalState.HAPPY: {
                "approach": "cheerful",
                "description": "保持愉快氛圍",
                "success_chance": "高",
            },
            EmotionalState.SAD: {
                "approach": "supportive",
                "description": "提供安慰支持",
                "success_chance": "中高",
            },
            EmotionalState.ANGRY: {
                "approach": "calm",
                "description": "冷靜溝通",
                "success_chance": "中",
            },
            EmotionalState.FEARFUL: {
                "approach": "reassuring",
                "description": "給予安全感",
                "success_chance": "中高",
            },
            EmotionalState.SUSPICIOUS: {
                "approach": "honest",
                "description": "展示誠意",
                "success_chance": "中",
            },
        }

        if persona.current_emotional_state in emotional_approaches:
            suggestions.append(emotional_approaches[persona.current_emotional_state])

        # Based on relationship status
        if player_relationship < 0:
            suggestions.append(
                {
                    "approach": "apologetic",
                    "description": "道歉和解",
                    "success_chance": "中",
                }
            )
        elif player_relationship > 5:
            suggestions.append(
                {
                    "approach": "casual",
                    "description": "輕鬆交談",
                    "success_chance": "高",
                }
            )

        return suggestions[:4]  # Return top 4 suggestions

    def export_personas_to_file(self, file_path: Path):
        """Export all personas to JSON file"""
        export_data = {
            "personas": [persona.to_dict() for persona in self.personas.values()],
            "relationships": self.relationship_matrix,
            "interaction_history": {
                k: v[-10:]
                for k, v in self.interaction_history.items()  # Export last 10 interactions
            },
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Exported personas to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export personas: {e}")

    def get_personas_by_type(self, persona_type: PersonaType) -> List[GamePersona]:
        """Get all personas of a specific type"""
        return [
            persona
            for persona in self.personas.values()
            if persona.persona_type == persona_type
        ]

    def get_story_relevant_personas(
        self, importance_threshold: int = 5
    ) -> List[GamePersona]:
        """Get personas that are important to the story"""
        return [
            persona
            for persona in self.personas.values()
            if persona.story_importance >= importance_threshold
        ]
