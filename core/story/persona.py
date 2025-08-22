# core/story/persona.py
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from .game_state import GameState, RelationType

logger = logging.getLogger(__name__)


@dataclass
class PersonaTrait:
    """Character personality trait"""

    name: str
    value: int  # -10 to 10 scale
    description: str = ""


@dataclass
class PersonaMemory:
    """Character memory entry"""

    memory_id: str
    content: str
    importance: int  # 1-10 scale
    emotional_impact: int  # -5 to 5 scale
    related_characters: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Persona:
    """Character persona definition"""

    character_id: str
    name: str
    description: str
    background: str = ""

    # Personality traits
    traits: Dict[str, PersonaTrait] = field(default_factory=dict)

    # Speech patterns and behavior
    speech_style: str = ""
    mannerisms: List[str] = field(default_factory=list)
    catchphrases: List[str] = field(default_factory=list)

    # Relationships and social context
    relationships: Dict[str, str] = field(
        default_factory=dict
    )  # character_id -> relationship_description
    social_role: str = ""

    # Goals and motivations
    goals: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)

    # Memory system
    memories: List[PersonaMemory] = field(default_factory=list)
    memory_decay_rate: float = 0.95  # How fast memories fade

    # Visual description
    appearance: str = ""
    clothing_style: str = ""
    distinctive_features: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def add_memory(
        self,
        content: str,
        importance: int = 5,
        emotional_impact: int = 0,
        related_characters: List[str] = None,
        tags: List[str] = None,
    ):
        """Add a new memory to this persona"""
        memory = PersonaMemory(
            memory_id=f"{self.character_id}_{len(self.memories)}",
            content=content,
            importance=importance,
            emotional_impact=emotional_impact,
            related_characters=related_characters or [],
            tags=tags or [],
        )
        self.memories.append(memory)
        self.last_updated = datetime.now()

    def get_relevant_memories(
        self, query: str, context_characters: List[str] = None, top_k: int = 5
    ) -> List[PersonaMemory]:
        """Get memories relevant to current context"""
        # Simple relevance scoring based on keywords and characters
        scored_memories = []

        query_words = set(query.lower().split())
        context_chars = set(context_characters or [])

        for memory in self.memories:
            score = 0

            # Content relevance
            memory_words = set(memory.content.lower().split())
            word_overlap = len(query_words.intersection(memory_words))
            score += word_overlap * 2

            # Character relevance
            memory_chars = set(memory.related_characters)
            char_overlap = len(context_chars.intersection(memory_chars))
            score += char_overlap * 3

            # Importance and emotional impact
            score += memory.importance
            score += abs(memory.emotional_impact)

            # Recency boost (more recent memories are more accessible)
            days_old = (datetime.now() - memory.timestamp).days
            recency_factor = max(0.1, 1.0 - (days_old / 30))  # Fade over 30 days
            score *= recency_factor

            if score > 0:
                scored_memories.append((memory, score))

        # Sort by score and return top k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, score in scored_memories[:top_k]]

    def get_personality_summary(self) -> str:
        """Get a summary of personality traits"""
        trait_descriptions = []
        for trait_name, trait in self.traits.items():
            if abs(trait.value) >= 3:  # Only significant traits
                intensity = (
                    "非常"
                    if abs(trait.value) >= 7
                    else "相當" if abs(trait.value) >= 5 else "有些"
                )
                direction = trait.description if trait.description else trait_name
                trait_descriptions.append(f"{intensity}{direction}")

        return "、".join(trait_descriptions) if trait_descriptions else "性格平衡"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "character_id": self.character_id,
            "name": self.name,
            "description": self.description,
            "background": self.background,
            "traits": {
                k: {"name": v.name, "value": v.value, "description": v.description}
                for k, v in self.traits.items()
            },
            "speech_style": self.speech_style,
            "mannerisms": self.mannerisms,
            "catchphrases": self.catchphrases,
            "relationships": self.relationships,
            "social_role": self.social_role,
            "goals": self.goals,
            "fears": self.fears,
            "secrets": self.secrets,
            "memories": [
                {
                    "memory_id": m.memory_id,
                    "content": m.content,
                    "importance": m.importance,
                    "emotional_impact": m.emotional_impact,
                    "related_characters": m.related_characters,
                    "tags": m.tags,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in self.memories
            ],
            "appearance": self.appearance,
            "clothing_style": self.clothing_style,
            "distinctive_features": self.distinctive_features,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


class PersonaManager:
    """Manages character personas and their interactions"""

    def __init__(self):
        self.personas: Dict[str, Persona] = {}
        self.dialogue_templates: Dict[str, str] = self._load_dialogue_templates()

    def _load_dialogue_templates(self) -> Dict[str, str]:
        """Load dialogue templates for different personality types"""
        return {
            "friendly": "以溫暖友善的語調，使用正面詞彙",
            "reserved": "話語簡潔，不太主動表達情感",
            "energetic": "語調活潑，經常使用感嘆詞",
            "serious": "用詞正式，邏輯清晰",
            "mysterious": "話中有話，留下懸念",
            "caring": "關心他人感受，語氣溫柔",
            "confident": "語氣堅定，不輕易動搖",
            "nervous": "用詞謹慎，經常自我懷疑",
        }

    def add_persona(self, persona: Persona):
        """Add a persona to the manager"""
        self.personas[persona.character_id] = persona
        logger.info(f"Added persona for {persona.character_id}")

    def get_persona(self, character_id: str) -> Optional[Persona]:
        """Get persona by character ID"""
        return self.personas.get(character_id)

    def create_persona_from_data(self, character_data: Dict[str, Any]) -> Persona:
        """Create persona from character data"""
        persona = Persona(
            character_id=character_data["id"],
            name=character_data["name"],
            description=character_data.get("description", ""),
            background=character_data.get("background", ""),
            speech_style=character_data.get("speech_style", ""),
            mannerisms=character_data.get("mannerisms", []),
            social_role=character_data.get("role", ""),
            appearance=character_data.get("appearance", ""),
            clothing_style=character_data.get("clothing", ""),
        )

        # Add personality traits
        if "personality" in character_data:
            for trait_name, trait_data in character_data["personality"].items():
                if isinstance(trait_data, dict):
                    persona.traits[trait_name] = PersonaTrait(
                        name=trait_name,
                        value=trait_data.get("value", 0),
                        description=trait_data.get("description", ""),
                    )
                else:
                    # Simple value
                    persona.traits[trait_name] = PersonaTrait(
                        name=trait_name, value=trait_data
                    )

        # Add relationships
        if "relationships" in character_data:
            persona.relationships = character_data["relationships"]

        # Add goals and motivations
        persona.goals = character_data.get("goals", [])
        persona.fears = character_data.get("fears", [])
        persona.secrets = character_data.get("secrets", [])

        return persona

    def generate_dialogue_style_prompt(
        self, character_id: str, context: str = ""
    ) -> str:
        """Generate dialogue style prompt for a character"""
        persona = self.get_persona(character_id)
        if not persona:
            return f"以{character_id}的身份回應"

        prompt_parts = [f"你是{persona.name}。"]

        # Add description
        if persona.description:
            prompt_parts.append(f"角色設定：{persona.description}")

        # Add personality traits
        personality_summary = persona.get_personality_summary()
        if personality_summary:
            prompt_parts.append(f"性格特點：{personality_summary}")

        # Add speech style
        if persona.speech_style:
            prompt_parts.append(f"說話風格：{persona.speech_style}")

        # Add mannerisms
        if persona.mannerisms:
            prompt_parts.append(f"行為特徵：{', '.join(persona.mannerisms)}")

        # Add relevant memories
        if context:
            relevant_memories = persona.get_relevant_memories(context, top_k=3)
            if relevant_memories:
                memory_texts = [m.content for m in relevant_memories]
                prompt_parts.append(f"相關記憶：{'; '.join(memory_texts)}")

        # Add current goals
        if persona.goals:
            prompt_parts.append(f"當前目標：{', '.join(persona.goals[:2])}")

        return "\n".join(prompt_parts)

    def update_persona_memory(
        self,
        character_id: str,
        event_description: str,
        importance: int = 5,
        emotional_impact: int = 0,
        related_characters: List[str] = None,
    ):
        """Update persona's memory with new event"""
        persona = self.get_persona(character_id)
        if persona:
            persona.add_memory(
                content=event_description,
                importance=importance,
                emotional_impact=emotional_impact,
                related_characters=related_characters or [],
            )
            logger.info(
                f"Updated memory for {character_id}: {event_description[:50]}..."
            )

    def get_character_context(
        self, character_id: str, scene_context: str = ""
    ) -> Dict[str, Any]:
        """Get complete character context for story generation"""
        persona = self.get_persona(character_id)
        if not persona:
            return {"character_id": character_id, "context": "未知角色"}

        # Get relevant memories
        relevant_memories = persona.get_relevant_memories(scene_context, top_k=5)

        context = {
            "character_id": character_id,
            "name": persona.name,
            "description": persona.description,
            "personality": persona.get_personality_summary(),
            "speech_style": persona.speech_style,
            "current_goals": persona.goals[:3],
            "relationships": persona.relationships,
            "relevant_memories": [m.content for m in relevant_memories],
            "appearance": persona.appearance,
            "social_role": persona.social_role,
        }

        return context

    def analyze_relationship_dynamics(
        self, char1_id: str, char2_id: str
    ) -> Dict[str, Any]:
        """Analyze relationship dynamics between two characters"""
        persona1 = self.get_persona(char1_id)
        persona2 = self.get_persona(char2_id)

        if not persona1 or not persona2:
            return {"status": "unknown", "reason": "Missing persona data"}

        # Check mutual relationships
        char1_view = persona1.relationships.get(char2_id, "未知")
        char2_view = persona2.relationships.get(char1_id, "未知")

        # Analyze personality compatibility
        compatibility_score = 0
        if persona1.traits and persona2.traits:
            # Simple compatibility based on complementary traits
            for trait_name in persona1.traits:
                if trait_name in persona2.traits:
                    trait1_val = persona1.traits[trait_name].value
                    trait2_val = persona2.traits[trait_name].value

                    # Some traits work better when similar, others when different
                    if trait_name in ["kindness", "loyalty", "honesty"]:
                        # Similar values are better
                        compatibility_score += 10 - abs(trait1_val - trait2_val)
                    else:
                        # Different values can be complementary
                        compatibility_score += abs(trait1_val - trait2_val) / 2

        return {
            "char1_view": char1_view,
            "char2_view": char2_view,
            "mutual": char1_view == char2_view,
            "compatibility_score": compatibility_score,
            "potential_conflicts": self._identify_potential_conflicts(
                persona1, persona2
            ),
            "shared_interests": self._find_shared_interests(persona1, persona2),
        }

    def _identify_potential_conflicts(
        self, persona1: Persona, persona2: Persona
    ) -> List[str]:
        """Identify potential sources of conflict between characters"""
        conflicts = []

        # Goal conflicts
        for goal1 in persona1.goals:
            for goal2 in persona2.goals:
                if any(
                    word in goal1.lower() and word in goal2.lower()
                    for word in ["競爭", "對立", "反對"]
                ):
                    conflicts.append(f"目標衝突：{goal1} vs {goal2}")

        # Personality clashes
        opposing_traits = [
            ("honest", "deceptive"),
            ("patient", "impulsive"),
            ("cautious", "reckless"),
            ("social", "antisocial"),
        ]

        for trait1, trait2 in opposing_traits:
            if (
                trait1 in persona1.traits
                and trait2 in persona2.traits
                and persona1.traits[trait1].value > 5
                and persona2.traits[trait2].value > 5
            ):
                conflicts.append(f"性格衝突：{trait1} vs {trait2}")

        return conflicts

    def _find_shared_interests(self, persona1: Persona, persona2: Persona) -> List[str]:
        """Find shared interests between characters"""
        shared = []

        # Common goals
        common_goals = set(persona1.goals).intersection(set(persona2.goals))
        shared.extend([f"共同目標：{goal}" for goal in common_goals])

        # Similar personality traits
        for trait_name in persona1.traits:
            if trait_name in persona2.traits:
                val1 = persona1.traits[trait_name].value
                val2 = persona2.traits[trait_name].value
                if abs(val1 - val2) <= 2 and abs(val1) >= 3:  # Similar and significant
                    shared.append(f"相似特質：{trait_name}")

        return shared
