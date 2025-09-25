# =============================================================================
# core/story/story_system.py
"""
Enhanced Text Adventure Game System
完整的文字冒險遊戲增強系統，支持角色管理、場景轉換、上下文追蹤
"""

import logging
import json
import random
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# Enhanced imports
from ..llm import ChatMessage, LLMResponse, get_llm_adapter
from .engine import StoryEngine
from ..config import get_config
from ..exceptions import (
    GameError,
    ValidationError,
    SessionNotFoundError,
    InvalidChoiceError,
)
from .narrative import NarrativeGenerator


logger = logging.getLogger(__name__)


# =============================================================================
# Enhanced Character System
# =============================================================================


class CharacterRole(Enum):
    """Character roles in the story"""

    PLAYER = "player"
    NPC = "npc"
    COMPANION = "companion"
    ANTAGONIST = "antagonist"
    NARRATOR = "narrator"


class CharacterState(Enum):
    """Character emotional/physical states"""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    FEARFUL = "fearful"
    EXCITED = "excited"
    TIRED = "tired"
    WOUNDED = "wounded"
    SUSPICIOUS = "suspicious"


@dataclass
class GameCharacter:
    """Enhanced character with full personality and state tracking"""

    character_id: str
    name: str
    role: CharacterRole

    # Personality & Behavior
    personality_traits: List[str]
    speaking_style: str
    background_story: str
    motivations: List[str]
    relationships: Dict[str, str]  # character_id -> relationship_type

    # Current State
    current_state: CharacterState = CharacterState.NEUTRAL
    current_location: str = "unknown"
    health: int = 100
    mood: str = "neutral"

    # Story Integration
    dialogue_history: List[Dict[str, Any]] = field(default_factory=list)
    interaction_count: int = 0
    last_seen_turn: int = 0

    # AI Generation Context
    persona_prompt: str = ""
    content_restrictions: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"GameCharacter({self.name}, {self.role.value if hasattr(self.role, 'value') else self.role})"

    def __repr__(self) -> str:
        return (
            f"GameCharacter(character_id='{self.character_id}', name='{self.name}', "
            f"role={self.role}, current_state={self.current_state})"
        )

    def add_dialogue(self, content: str, turn_number: int, context: str = ""):
        """Add dialogue to character history"""
        self.dialogue_history.append(
            {
                "turn": turn_number,
                "content": content,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "mood": self.current_state.value,
            }
        )
        self.interaction_count += 1
        self.last_seen_turn = turn_number

    def get_recent_dialogue(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent dialogue history"""
        return self.dialogue_history[-count:] if self.dialogue_history else []

    def update_relationship(self, other_character_id: str, relationship_type: str):
        """Update relationship with another character"""
        self.relationships[other_character_id] = relationship_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "character_id": self.character_id,
            "name": self.name,
            "role": self.role.value,
            "personality_traits": self.personality_traits,
            "speaking_style": self.speaking_style,
            "background_story": self.background_story,
            "motivations": self.motivations,
            "relationships": self.relationships,
            "current_state": self.current_state.value,
            "current_location": self.current_location,
            "health": self.health,
            "mood": self.mood,
            "dialogue_history": self.dialogue_history[-10:],  # Keep last 10 only
            "interaction_count": self.interaction_count,
            "last_seen_turn": self.last_seen_turn,
        }


# =============================================================================
# Enhanced Scene Management
# =============================================================================


class SceneType(Enum):
    """Types of story scenes"""

    OPENING = "opening"
    EXPLORATION = "exploration"
    DIALOGUE = "dialogue"
    COMBAT = "combat"
    PUZZLE = "puzzle"
    DRAMATIC = "dramatic"
    TRANSITION = "transition"
    CLIMAX = "climax"
    RESOLUTION = "resolution"


class SceneMood(Enum):
    """Scene atmosphere moods"""

    PEACEFUL = "peaceful"
    TENSE = "tense"
    MYSTERIOUS = "mysterious"
    EXCITING = "exciting"
    MELANCHOLY = "melancholy"
    ROMANTIC = "romantic"
    DANGEROUS = "dangerous"
    HUMOROUS = "humorous"
    NEUTRAL = "neutral"


@dataclass
class SceneContext:
    """Enhanced scene context with full environmental details"""

    scene_id: str
    scene_type: SceneType
    title: str
    description: str
    location: str

    # Environment
    time_of_day: str = "未知"
    weather: str = "未知"
    atmosphere: Optional["SceneMood"] = None

    # Characters Present
    present_characters: List[str] = field(default_factory=list)
    character_interactions: List[Dict[str, Any]] = field(default_factory=list)
    primary_npc: Optional[str] = None

    # Story State
    plot_points: List[str] = field(default_factory=list)
    active_conflicts: List[str] = field(default_factory=list)
    scene_objectives: List[str] = field(default_factory=list)
    scene_events: List[Dict[str, Any]] = field(default_factory=list)

    # Choices & Consequences
    available_actions: List[str] = field(default_factory=list)
    restricted_actions: List[str] = field(default_factory=list)
    available_choices: List[str] = field(default_factory=list)
    choice_history: List[Dict[str, Any]] = field(default_factory=list)

    # Continuity
    previous_scene_id: Optional[str] = None
    potential_next_scenes: List[str] = field(default_factory=list)

    # Meta
    created_at: datetime = field(default_factory=datetime.now)
    turn_number: int = 0

    def __post_init__(self):
        """Initialize default atmosphere if not set"""
        if self.atmosphere is None:
            self.atmosphere = SceneMood.NEUTRAL

    def add_character(self, character_id: str):
        """Add a character to this scene"""
        if character_id not in self.present_characters:
            self.present_characters.append(character_id)

            self.scene_events.append(
                {
                    "type": "character_entered",
                    "character_id": character_id,
                    "turn": self.turn_number,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def remove_character(self, character_id: str):
        """Remove a character from this scene"""
        if character_id in self.present_characters:
            self.present_characters.remove(character_id)

            self.scene_events.append(
                {
                    "type": "character_left",
                    "character_id": character_id,
                    "turn": self.turn_number,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def add_event(self, event: Dict[str, Any]):
        """Add an event to this scene"""
        event_with_metadata = {
            "turn": self.turn_number,
            "timestamp": datetime.now().isoformat(),
            **event,
        }
        self.scene_events.append(event_with_metadata)

    def advance_turn(self):
        """Advance the scene turn number"""
        self.turn_number += 1

    def get_character_count(self) -> int:
        """Get number of characters in scene"""
        return len(self.present_characters)

    def get_recent_events(self, count: int = 3) -> List[Dict[str, Any]]:
        """Get recent scene events"""
        return self.scene_events[-count:] if self.scene_events else []

    def set_atmosphere(self, new_atmosphere: "SceneMood", reason: str = ""):
        """Change scene atmosphere"""
        old_atmosphere = self.atmosphere
        self.atmosphere = new_atmosphere

        self.add_event(
            {
                "type": "atmosphere_change",
                "from": (
                    old_atmosphere
                    if hasattr(old_atmosphere, "value")
                    else str(old_atmosphere)
                ),
                "to": (
                    new_atmosphere.value
                    if hasattr(new_atmosphere, "value")
                    else str(new_atmosphere)
                ),
                "reason": reason,
            }
        )

    def get_scene_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of this scene"""
        return {
            "scene_id": self.scene_id,
            "title": self.title,
            "location": self.location,
            "scene_type": (
                self.scene_type.value
                if hasattr(self.scene_type, "value")
                else str(self.scene_type)
            ),
            "atmosphere": (
                self.atmosphere
                if hasattr(self.atmosphere, "value")
                else str(self.atmosphere)
            ),
            "character_count": len(self.present_characters),
            "present_characters": self.present_characters,
            "event_count": len(self.scene_events),
            "turn_number": self.turn_number,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
        }

    def add_plot_point(self, plot_point: str):
        """Add plot point to current scene"""
        if plot_point not in self.plot_points:
            self.plot_points.append(plot_point)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_type": self.scene_type.value,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
            "atmosphere": self.atmosphere,
            "present_characters": self.present_characters,
            "primary_npc": self.primary_npc,
            "plot_points": self.plot_points,
            "active_conflicts": self.active_conflicts,
            "scene_objectives": self.scene_objectives,
            "available_actions": self.available_actions,
            "turn_number": self.turn_number,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Enhanced Context Manager
# =============================================================================


@dataclass
class StoryContextMemory:
    """Enhanced story context with full memory management"""

    def __init__(self, session_id: str):
        self.session_id = session_id

        # Player tracking
        self.player_name: str
        self.player_choices_history: List[Dict[str, Any]] = []
        self.player_personality_profile: Dict[str, Any] = {}

        # Scene and narrative state
        self.current_scene: Optional[SceneContext] = None
        self.current_scene_id: Optional[str] = None
        self.scene_history: List[SceneContext] = []
        self.scenes: Dict[str, SceneContext] = {}  # 新增：場景字典
        self.narrative_memory: List[Dict[str, Any]] = []

        # Character management
        self.characters: Dict[str, GameCharacter] = {}
        self.character_relationships: Dict[str, Dict[str, int]] = {}

        # World state
        self.world_flags: Dict[str, bool] = {}
        self.global_variables: Dict[str, Any] = {}
        self.timeline: List[Dict[str, Any]] = []

    scene_sequence: List[str] = field(default_factory=list)

    # World State
    location_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    time_progression: Dict[str, Any] = field(default_factory=dict)

    # Story Arc Tracking
    main_plot_points: List[str] = field(default_factory=list)
    subplot_threads: Dict[str, List[str]] = field(default_factory=dict)
    completed_objectives: List[str] = field(default_factory=list)

    # Player Progress
    player_decisions: List[Dict[str, Any]] = field(default_factory=list)
    player_relationships: Dict[str, int] = field(
        default_factory=dict
    )  # character_id -> relationship_score

    def add_character(self, character: GameCharacter):
        """Add a character to the context memory"""
        self.characters[character.character_id] = character

        # Initialize relationship tracking for this character
        if character.character_id not in self.character_relationships:
            self.character_relationships[character.character_id] = {}

        # Log the addition
        self.add_narrative_memory(
            {
                "type": "character_added",
                "character_id": character.character_id,
                "character_name": character.name,
                "role": (
                    character.role.value
                    if hasattr(character.role, "value")
                    else str(character.role)
                ),
            }
        )

    def add_narrative_memory(self, event: Dict[str, Any]):
        """Add an event to narrative memory"""
        event_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            "turn_number": len(self.narrative_memory),
            **event,
        }
        self.narrative_memory.append(event_with_timestamp)

        # Keep memory size reasonable
        if len(self.narrative_memory) > 100:
            self.narrative_memory = self.narrative_memory[-100:]

    def get_recent_narrative_memory(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent narrative memories"""
        return self.narrative_memory[-count:] if self.narrative_memory else []

    def transition_to_scene(self, new_scene: SceneContext):
        """Transition to a new scene"""
        if self.current_scene:
            self.scene_history.append(self.current_scene)

        # Add to scenes dict if not already there
        if new_scene.scene_id not in self.scenes:
            self.scenes[new_scene.scene_id] = new_scene

        self.current_scene = new_scene
        self.current_scene_id = new_scene.scene_id

        self.add_narrative_memory(
            {
                "type": "scene_transition",
                "from_scene": (
                    self.scene_history[-1].scene_id if self.scene_history else None
                ),
                "to_scene": new_scene.scene_id,
                "location": new_scene.location,
            }
        )

    def get_character(self, character_id: str) -> Optional[GameCharacter]:
        """Get character by ID"""
        return self.characters.get(character_id)

    def update_character_relationship(self, char1_id: str, char2_id: str, change: int):
        """Update relationship between two characters"""
        if char1_id not in self.character_relationships:
            self.character_relationships[char1_id] = {}

        current_rel = self.character_relationships[char1_id].get(char2_id, 0)
        new_rel = max(-100, min(100, current_rel + change))
        self.character_relationships[char1_id][char2_id] = new_rel

    def add_scene(self, scene: SceneContext):
        """Add a scene to the context memory"""
        self.scenes[scene.scene_id] = scene
        self.scene_history.append(scene)

        # Log scene addition
        self.add_narrative_memory(
            {
                "type": "scene_added",
                "scene_id": scene.scene_id,
                "location": scene.location,
                "scene_type": (
                    scene.scene_type.value
                    if hasattr(scene.scene_type, "value")
                    else str(scene.scene_type)
                ),
            }
        )

    def get_current_scene(self) -> Optional[SceneContext]:
        """Get current scene context"""
        return self.scenes.get(self.current_scene_id)  # type: ignore

    def get_characters_in_scene(
        self, scene_id: Optional[str] = None
    ) -> List[GameCharacter]:
        """Get all characters present in the specified or current scene"""
        target_scene = self.current_scene
        if scene_id:
            target_scene = self.scenes.get(scene_id)

        if not target_scene:
            return []

        present_chars = []
        for char_id in getattr(target_scene, "present_characters", []):
            if char_id in self.characters:
                present_chars.append(self.characters[char_id])

        return present_chars

    def record_player_decision(
        self,
        decision_text: str,
        choice_id: str,
        turn_number: int,
        consequences: Dict[str, Any] = None,  # type: ignore
    ):
        """Record a player decision for context"""
        self.player_decisions.append(
            {
                "turn": turn_number,
                "decision": decision_text,
                "choice_id": choice_id,
                "consequences": consequences or {},
                "timestamp": datetime.now().isoformat(),
                "scene_id": self.current_scene_id,
            }
        )

    def get_relationship_context(self, character_id: str) -> Dict[str, Any]:
        """Get full relationship context for a character"""
        character = self.get_character(character_id)
        if not character:
            return {}

        return {
            "character_info": character.to_dict(),
            "player_relationship_score": self.player_relationships.get(character_id, 0),
            "recent_interactions": character.get_recent_dialogue(3),
            "shared_scenes": [
                scene_id
                for scene_id, scene in self.scenes.items()
                if character_id in scene.present_characters
            ],
        }


# 8. 確保所有輔助函數都正確實作
def safe_get_enum_value(enum_obj: Any, default: str = "unknown") -> str:
    """Safely get enum value, handling both enum and string cases"""
    if hasattr(enum_obj, "value"):
        return enum_obj.value
    elif isinstance(enum_obj, str):
        return enum_obj
    else:
        return default


def create_default_scene(
    location: str = "起始點", scene_id: str = "scene_001"
) -> SceneContext:
    """Create a default scene for initialization"""
    return SceneContext(
        scene_id=scene_id,
        scene_type=SceneType.EXPLORATION,
        title="冒險的開始",
        description=f"你發現自己在{location}，準備開始一段新的冒險旅程。",
        location=location,
        time_of_day="黃昏",
        weather="晴朗",
        atmosphere=SceneMood.MYSTERIOUS,
    )


def validate_story_context(context_memory: StoryContextMemory) -> List[str]:
    """Validate story context and return any issues found"""
    issues = []

    if not context_memory.current_scene:
        issues.append("缺少當前場景")

    if not context_memory.characters:
        issues.append("缺少角色定義")

    if "player" not in context_memory.characters:
        issues.append("缺少玩家角色")

    # Check for essential scene properties
    if context_memory.current_scene:
        scene = context_memory.current_scene
        if not scene.location:
            issues.append("當前場景缺少位置信息")
        if not hasattr(scene, "scene_type") or not scene.scene_type:
            issues.append("當前場景缺少類型定義")

    return issues


# =============================================================================
# Enhanced Choice System with Context Awareness
# =============================================================================


@dataclass
class ContextualChoice:
    """Choice that adapts based on story context"""

    choice_id: str
    base_text: str
    choice_type: str

    # Context Requirements
    required_characters: List[str] = field(default_factory=list)
    required_flags: Dict[str, bool] = field(default_factory=dict)
    required_items: List[str] = field(default_factory=list)
    required_location: Optional[str] = None
    consequences: Dict[str, Any] = field(default_factory=dict)

    # Dynamic Text Generation
    context_sensitive: bool = False
    text_variations: Dict[str, str] = field(default_factory=dict)

    # Consequences
    stat_changes: Dict[str, int] = field(default_factory=dict)
    relationship_changes: Dict[str, int] = field(default_factory=dict)
    flag_changes: Dict[str, bool] = field(default_factory=dict)
    scene_transitions: List[str] = field(default_factory=list)

    # Narrative Impact
    difficulty: str = "medium"
    success_chance: float = 1.0
    dramatic_weight: int = 1  # How much this choice affects the story

    def __str__(self) -> str:
        return f"ContextualChoice({self.choice_id}: {self.base_text})"

    def __repr__(self) -> str:
        return (
            f"ContextualChoice(choice_id='{self.choice_id}', base_text='{self.base_text}', "
            f"choice_type='{self.choice_type}', success_chance={self.success_chance})"
        )

    def get_display_text(self, context: StoryContextMemory) -> str:
        """Get context-appropriate display text"""
        if not self.context_sensitive:
            return self.base_text

        # Check for context variations
        scene = context.get_current_scene()
        if scene and scene.atmosphere in self.text_variations:
            return self.text_variations[scene.atmosphere.value]

        return self.base_text

    def can_execute(
        self,
        context: Union[StoryContextMemory, Dict[str, Any]],
        player_stats: Dict[str, int],
        inventory: List[str],
    ) -> Tuple[bool, str]:
        """Check if choice can be executed in current context (修正版本)"""

        # Handle both StoryContextMemory object and dict context
        if hasattr(context, "characters"):
            # StoryContextMemory object
            context_chars = context.characters  # type: ignore
            world_flags = context.world_flags  # type: ignore
            current_scene = context.get_current_scene()  # type: ignore
        else:
            # Dict context (fallback)
            context_chars = context.get("characters", {})  # type: ignore
            world_flags = context.get("world_flags", {})  # type: ignore
            current_scene = context.get("current_scene")  # type: ignore

        # Check character requirements
        for char_id in self.required_characters:
            if char_id not in context_chars:
                return False, f"需要角色: {char_id}"

            if current_scene and hasattr(current_scene, "present_characters"):
                if char_id not in current_scene.present_characters:
                    char_name = (
                        context_chars[char_id].name
                        if char_id in context_chars
                        else char_id
                    )
                    return False, f"角色不在場: {char_name}"

        # Check flag requirements
        for flag, required_value in self.required_flags.items():
            if world_flags.get(flag, False) != required_value:
                return False, f"條件不符: {flag}"

        # Check item requirements
        for item in self.required_items:
            if item not in inventory:
                return False, f"需要物品: {item}"

        # Check location requirement
        if self.required_location and current_scene:
            if (
                hasattr(current_scene, "location")
                and current_scene.location != self.required_location
            ):
                return False, f"需要在 {self.required_location}"

        return True, ""


# =============================================================================
# Enhanced Narrative Generator with Context Integration
# =============================================================================


class EnhancedNarrativeGenerator(NarrativeGenerator):
    """Advanced narrative generator with full context awareness"""

    def __init__(self):
        super().__init__()

        self.llm = get_llm_adapter()
        self.config = get_config()

        # Narrative templates by scene type and context
        self.narrative_templates = self._load_narrative_templates()
        self.character_voice_cache = {}  # Cache character-specific prompts

    def generate_narrative(
        self, context: Dict[str, Any], choice_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate basic narrative using base generator (fix the missing method error)"""
        try:
            # Use parent class method if available
            return super().generate_narrative(context, choice_result)
        except Exception as e:
            logger.warning(f"Fallback narrative generation: {e}")
            # Simple fallback implementation
            player_input = context.get("player_input", "")
            location = context.get("current_location", "未知地點")

            if choice_result and choice_result.get("success"):
                return f"在{location}中，你的行動取得了積極的結果。故事繼續向前發展..."
            elif choice_result and not choice_result.get("success"):
                return f"在{location}中，儘管遇到了一些困難，但你仍然從中學到了寶貴的經驗..."
            else:
                return f"你在{location}中繼續著你的冒險旅程，每一個選擇都將影響故事的走向..."

    def _load_narrative_templates(self) -> Dict[str, Any]:
        """Load narrative templates for different contexts"""
        return {
            "scene_transitions": {
                "location_change": "隨著{player_name}離開{old_location}，前往{new_location}，環境發生了顯著的變化...",
                "time_passage": "時間悄悄流逝，{time_description}，為接下來的事件蒙上了新的色彩...",
                "mood_shift": "氣氛突然{mood_change}，讓在場的每個人都感受到了這種轉變...",
            },
            "character_interactions": {
                "first_meeting": "{character_name}首次出現時，{appearance_description}。你能感受到{personality_hint}...",
                "relationship_change": "{character_name}看你的眼神{relationship_description}，你們之間的關係似乎{change_direction}...",
                "dialogue_context": "在{location}的{atmosphere}氛圍中，{character_name}{emotion_state}地說道...",
            },
            "plot_development": {
                "revelation": "突然間，一個重要的真相浮出水面: {revelation_content}。這改變了一切...",
                "conflict_escalation": "情況變得更加複雜，{conflict_description}讓事態向{direction}發展...",
                "objective_completion": "隨著{objective}的完成，新的可能性展現在眼前...",
            },
        }

    async def generate_contextual_narrative(
        self,
        context_memory: StoryContextMemory,
        player_input: str,
        choice_result: Optional[Dict[str, Any]] = None,
        forced_scene_type: Optional[SceneType] = None,
    ) -> Dict[str, Any]:
        """Generate narrative with full context awareness"""

        current_scene = context_memory.get_current_scene()
        present_characters = context_memory.get_characters_in_scene()

        # Determine narrative focus
        narrative_focus = self._determine_narrative_focus(
            player_input, current_scene, present_characters, choice_result
        )

        # Build comprehensive context prompt
        context_prompt = self._build_comprehensive_context_prompt(
            context_memory, player_input, narrative_focus
        )

        # Generate main narrative
        main_narrative = await self._generate_main_narrative(
            context_prompt, narrative_focus, forced_scene_type
        )

        # Generate character dialogues if NPCs are present
        character_dialogues = await self._generate_character_dialogues(
            context_memory, present_characters, player_input, main_narrative
        )

        # Determine scene changes and transitions
        scene_changes = self._analyze_scene_changes(
            context_memory, player_input, choice_result
        )

        return {
            "main_narrative": main_narrative,
            "character_dialogues": character_dialogues,
            "scene_changes": scene_changes,
            "narrative_focus": (
                narrative_focus.value
                if isinstance(narrative_focus, Enum)
                else narrative_focus
            ),
            "present_characters": [char.name for char in present_characters],
            "mood_shift": self._detect_mood_shift(context_memory, main_narrative),
        }

    def _determine_narrative_focus(
        self,
        player_input: str,
        current_scene: Optional[SceneContext],
        present_characters: List[GameCharacter],
        choice_result: Optional[Dict[str, Any]],
    ) -> str:
        """Determine what the narrative should focus on"""

        # Keyword-based focus detection
        input_lower = player_input.lower()

        if any(word in input_lower for word in ["說話", "對話", "交談", "問"]):
            if present_characters:
                return "character_dialogue"

        if any(word in input_lower for word in ["探索", "查看", "觀察", "尋找"]):
            return "environment_exploration"

        if any(word in input_lower for word in ["戰鬥", "攻擊", "防禦", "逃跑"]):
            return "combat_action"

        if any(word in input_lower for word in ["移動", "前往", "離開", "進入"]):
            return "location_transition"

        if choice_result and choice_result.get("dramatic_weight", 0) > 5:
            return "dramatic_consequence"

        # Default focus based on scene type
        if current_scene:
            if current_scene.scene_type == SceneType.DIALOGUE:
                return "character_interaction"
            elif current_scene.scene_type == SceneType.COMBAT:
                return "action_sequence"
            elif current_scene.scene_type == SceneType.EXPLORATION:
                return "environment_discovery"

        return "general_progression"

    def _build_comprehensive_context_prompt(
        self,
        context_memory: StoryContextMemory,
        player_input: str,
        narrative_focus: str,
    ) -> str:
        """Build comprehensive context prompt for narrative generation"""

        current_scene = context_memory.get_current_scene()
        present_characters = context_memory.get_characters_in_scene()

        # Recent story context (last 3 scenes)
        recent_scenes = []
        for scene_id in context_memory.scene_sequence[-3:]:
            scene = context_memory.scenes.get(scene_id)
            if scene:
                recent_scenes.append(f"- {scene.title}: {scene.description[:100]}...")

        # Character context
        character_context = []
        for char in present_characters:
            recent_dialogue = char.get_recent_dialogue(2)
            dialogue_summary = ""
            if recent_dialogue:
                last_dialogue = recent_dialogue[-1]
                dialogue_summary = f"最近說過: '{last_dialogue['content'][:50]}...'"

            character_context.append(
                f"- {char.name} ({char.role.value}): {char.current_state.value}狀態, "
                f"關係: {context_memory.player_relationships.get(char.character_id, 0)}/10, "
                f"{dialogue_summary}"
            )

        # World state context
        world_context = []
        if context_memory.world_flags:
            active_flags = [
                f"{k}: {v}" for k, v in context_memory.world_flags.items() if v
            ]
            world_context.append(f"世界狀態: {', '.join(active_flags[:5])}")

        if context_memory.main_plot_points:
            world_context.append(
                f"主要劇情: {'; '.join(context_memory.main_plot_points[-3:])}"
            )

        # Recent player decisions context
        recent_decisions = []
        for decision in context_memory.player_decisions[-3:]:
            recent_decisions.append(f"- 第{decision['turn']}回: {decision['decision']}")

        context_prompt = f"""
故事上下文資訊：

【當前場景】
場景: {current_scene.title if current_scene else '未知'}
位置: {current_scene.location if current_scene else '未知'}
時間: {current_scene.time_of_day if current_scene else '未知'}
氣氛: {current_scene.atmosphere if current_scene else '中性'}
場景目標: {', '.join(current_scene.scene_objectives) if current_scene and current_scene.scene_objectives else '探索'}

【在場角色】
{chr(10).join(character_context) if character_context else '無其他角色'}

【最近場景歷史】
{chr(10).join(recent_scenes) if recent_scenes else '故事剛開始'}

【世界狀態】
{chr(10).join(world_context) if world_context else '無特殊狀態'}

【最近玩家決定】
{chr(10).join(recent_decisions) if recent_decisions else '尚無重要決定'}

【當前玩家行動】
玩家輸入: {player_input}
敘述重點: {narrative_focus}

【生成要求】
1. 根據上下文生成連貫的故事敘述
2. 體現角色個性和當前狀態
3. 推進劇情但保持適度懸念
4. 字數控制在200-400字之間
5. 考慮場景氣氛和時間流逝
        """

        return context_prompt

    async def _generate_main_narrative(
        self,
        context_prompt: str,
        narrative_focus: str,
        forced_scene_type: Optional[SceneType] = None,
    ) -> str:
        """Generate main narrative text"""

        system_prompt = f"""
你是一個專業的互動式故事敘述者。你的任務是：

1. 基於提供的完整上下文生成引人入勝的故事敘述
2. 保持角色一致性和情節連貫性
3. 根據敘述重點 ({narrative_focus}) 調整描述焦點
4. 創造適度的戲劇張力和懸念
5. 使用生動的感官描述和情感表達
6. 為玩家的下一步行動設置自然的情境

敘述風格：
- 使用第二人稱 ("你")
- 現在式描述
- 富有畫面感的描述
- 適度的情感渲染
        """

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=context_prompt),
        ]

        try:
            response = await self.llm.chat(messages)  # type: ignore

            if isinstance(response, LLMResponse):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()

        except Exception as e:
            logger.error(f"Failed to generate main narrative: {e}")
            return "你的行動在這個神秘的世界中引起了一些變化，但具體會發生什麼，還需要時間來揭曉..."

    async def _generate_character_dialogues(
        self,
        context_memory: StoryContextMemory,
        present_characters: List[GameCharacter],
        player_input: str,
        main_narrative: str,
    ) -> List[Dict[str, str]]:
        """Generate contextual character dialogues"""

        dialogues = []

        for character in present_characters:
            if character.role == CharacterRole.PLAYER:
                continue

            # Check if character should speak based on context
            should_speak = self._should_character_speak(
                character, player_input, main_narrative, context_memory
            )

            if should_speak:
                dialogue = await self._generate_single_character_dialogue(
                    character, context_memory, player_input, main_narrative
                )

                if dialogue:
                    dialogues.append(
                        {
                            "character_id": character.character_id,
                            "character_name": character.name,
                            "content": dialogue,
                            "emotional_state": character.current_state.value,
                        }
                    )

                    # Update character dialogue history
                    current_scene = context_memory.get_current_scene()
                    turn_number = current_scene.turn_number if current_scene else 0
                    character.add_dialogue(dialogue, turn_number, player_input)

        return dialogues

    def _should_character_speak(
        self,
        character: GameCharacter,
        player_input: str,
        main_narrative: str,
        context_memory: StoryContextMemory,
    ) -> bool:
        """Determine if character should speak in current context"""

        input_lower = player_input.lower()
        narrative_lower = main_narrative.lower()
        char_name_lower = character.name.lower()

        # Character is directly addressed
        if char_name_lower in input_lower:
            return True

        # Character is mentioned in narrative
        if char_name_lower in narrative_lower:
            return True

        # Dialogue-focused scene
        current_scene = context_memory.get_current_scene()
        if current_scene and current_scene.scene_type == SceneType.DIALOGUE:
            return True

        # Character hasn't spoken recently and is primary NPC
        if (
            character.character_id == current_scene.primary_npc
            if current_scene
            else False
        ):
            if character.interaction_count < 2:  # Keep important NPCs active
                return True

        # Random chance for ambient dialogue (lower for background characters)
        ambient_chance = 0.3 if character.role == CharacterRole.COMPANION else 0.1
        return random.random() < ambient_chance

    async def _generate_single_character_dialogue(
        self,
        character: GameCharacter,
        context_memory: StoryContextMemory,
        player_input: str,
        main_narrative: str,
    ) -> Optional[str]:
        """Generate dialogue for a single character"""

        # Get relationship context
        relationship_info = context_memory.get_relationship_context(
            character.character_id
        )
        relationship_score = relationship_info.get("player_relationship_score", 0)

        # Build character-specific context
        character_prompt = f"""
角色資訊：
- 姓名：{character.name}
- 角色：{character.role.value}
- 個性特徵：{', '.join(character.personality_traits)}
- 說話風格：{character.speaking_style}
- 當前狀態：{character.current_state.value}
- 與玩家關係評分：{relationship_score}/10

最近互動記錄：
{chr(10).join([f"- {d['content']}" for d in character.get_recent_dialogue(2)]) if character.get_recent_dialogue(2) else "無"}

當前情境：
- 玩家行動：{player_input}
- 場景描述：{main_narrative[:100]}...

請生成一段符合角色個性的對話，要求：
1. 體現角色的說話風格和個性特徵
2. 反映當前的情緒狀態
3. 考慮與玩家的關係
4. 推進對話或劇情
5. 長度控制在30-80字之間
        """

        messages = [
            ChatMessage(
                role="system", content=character.persona_prompt or "你是一個故事角色"
            ),
            ChatMessage(role="user", content=character_prompt),
        ]

        try:
            response = await self.llm.chat(messages)  # type: ignore

            if isinstance(response, LLMResponse):
                dialogue = response.content.strip()
            elif isinstance(response, str):
                dialogue = response.strip()
            else:
                dialogue = str(response).strip()

            # Clean up dialogue (remove quotes if present)
            dialogue = dialogue.strip('"').strip("'").strip()

            return dialogue if dialogue else None

        except Exception as e:
            logger.error(f"Failed to generate dialogue for {character.name}: {e}")
            return None

    def _analyze_scene_changes(
        self,
        context_memory: StoryContextMemory,
        player_input: str,
        choice_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze if scene changes are needed"""

        changes = {
            "location_change": False,
            "time_passage": False,
            "mood_shift": False,
            "character_changes": [],
            "new_scene_needed": False,
        }

        input_lower = player_input.lower()

        # Check for location change keywords
        if any(word in input_lower for word in ["離開", "前往", "進入", "回到"]):
            changes["location_change"] = True
            changes["new_scene_needed"] = True

        # Check for time passage indicators
        if any(word in input_lower for word in ["等待", "休息", "睡覺", "過了"]):
            changes["time_passage"] = True

        # Check choice consequences
        if choice_result:
            if choice_result.get("scene_transitions"):
                changes["new_scene_needed"] = True

            if choice_result.get("dramatic_weight", 0) > 3:
                changes["mood_shift"] = True

        return changes

    def _detect_mood_shift(
        self, context_memory: StoryContextMemory, narrative: str
    ) -> Optional[str]:
        """Detect mood shifts from narrative content"""

        narrative_lower = narrative.lower()

        mood_indicators = {
            "tense": ["緊張", "危險", "威脅", "小心", "警戒"],
            "peaceful": ["平靜", "寧靜", "安全", "放鬆", "舒適"],
            "mysterious": ["神秘", "奇怪", "詭異", "未知", "隱藏"],
            "exciting": ["興奮", "激動", "刺激", "驚喜", "發現"],
            "melancholy": ["憂鬱", "悲傷", "失落", "懷念", "遺憾"],
        }

        for mood, indicators in mood_indicators.items():
            if any(indicator in narrative_lower for indicator in indicators):
                return mood

        return None


# =============================================================================
# Enhanced Story Engine Integration
# =============================================================================


class EnhancedStoryEngine:
    """Enhanced story engine with full context management"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config = get_config()
        # Initialize base engine in enhanced mode
        self.base_engine = StoryEngine(config_dir=config_dir, enhanced_mode=True)

        # Enhanced components
        self.context_memories: Dict[str, StoryContextMemory] = {}
        self.character_managers: Dict[str, Dict[str, GameCharacter]] = {}
        self.scene_generators = self._setup_advanced_scene_generators()

        # Advanced narrative features
        self.narrative_cache = {}
        self.relationship_tracker = {}
        self.emotional_ai = EmotionalAI()

        # Enhanced components
        self.narrative_generator = EnhancedNarrativeGenerator()
        self.context_memory = StoryContextMemory("")

        # Story management
        self.active_sessions: Dict[str, StoryContextMemory] = {}

        # Enhanced choice system
        self.contextual_choices: Dict[str, List[ContextualChoice]] = {}
        self._load_contextual_choices()

        logger.info("Enhanced story engine initialized")

    def _load_contextual_choices(self):
        """Load contextual choice templates"""
        # Example contextual choices
        default_choices = {
            "dialogue": [
                ContextualChoice(
                    choice_id="friendly_greeting",
                    base_text="友善地打招呼",
                    choice_type="dialogue",
                    context_sensitive=True,
                    text_variations={
                        "peaceful": "溫和地向{character}問好",
                        "tense": "謹慎地向{character}致意",
                        "mysterious": "小心翼翼地接近{character}",
                    },
                    relationship_changes={"any": 1},
                    success_chance=0.9,
                ),
                ContextualChoice(
                    choice_id="ask_about_location",
                    base_text="詢問關於這個地方的情況",
                    choice_type="dialogue",
                    required_characters=["any_npc"],
                    relationship_changes={"any": 1},
                    success_chance=0.8,
                ),
            ],
            "exploration": [
                ContextualChoice(
                    choice_id="examine_surroundings",
                    base_text="仔細觀察周圍環境",
                    choice_type="exploration",
                    context_sensitive=True,
                    text_variations={
                        "mysterious": "警覺地檢查四周",
                        "peaceful": "悠閒地欣賞景色",
                        "dangerous": "謹慎地掃視環境",
                    },
                    success_chance=0.95,
                ),
                ContextualChoice(
                    choice_id="search_for_clues",
                    base_text="尋找線索或有用的物品",
                    choice_type="exploration",
                    stat_changes={"intelligence": 1},
                    success_chance=0.7,
                ),
            ],
            "action": [
                ContextualChoice(
                    choice_id="help_character",
                    base_text="主動提供幫助",
                    choice_type="action",
                    required_characters=["any_npc"],
                    relationship_changes={"any": 2},
                    stat_changes={"charisma": 1},
                    success_chance=0.8,
                ),
                ContextualChoice(
                    choice_id="wait_and_observe",
                    base_text="等待並觀察情況發展",
                    choice_type="action",
                    success_chance=1.0,
                ),
            ],
        }

        self.contextual_choices = default_choices

    def _setup_advanced_scene_generators(self) -> Dict[str, Callable]:
        """Setup advanced scene generation functions"""
        return {
            "dynamic_exploration": self._generate_dynamic_exploration,
            "character_driven_dialogue": self._generate_character_dialogue,
            "contextual_combat": self._generate_contextual_combat,
            "environmental_puzzle": self._generate_environmental_puzzle,
            "emotional_scene": self._generate_emotional_scene,
        }

    def create_session(
        self,
        player_name: str,
        persona_id: str = "default",
        enhanced_context: bool = True,
        starting_location: str = "神秘森林",
        custom_characters: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Create new enhanced game session with full context setup"""

        # Create base session
        session_id = self.base_engine.create_session(player_name, persona_id)

        if enhanced_context:
            # Initialize context memory
            context_memory = StoryContextMemory(session_id)  # type: ignore
            self.context_memories[session_id] = context_memory  # type: ignore

            # Create initial scene
            initial_scene = self._create_initial_scene(
                session_id, starting_location, player_name  # type: ignore
            )
            context_memory.transition_to_scene(initial_scene)

            # Setup character roster
            character_roster = self._setup_character_roster(
                session_id, custom_characters  # type: ignore
            )
            self.character_managers[session_id] = character_roster  # type: ignore

            # Add characters to context
            for char_id, character in character_roster.items():
                context_memory.add_character(character)

            logger.info(f"Enhanced session created: {session_id}")

        return session_id  # type: ignore

    def _create_initial_scene(
        self, session_id: str, location: str, player_name: str
    ) -> SceneContext:
        """Create the initial scene for a new session"""

        initial_scene = SceneContext(
            scene_id="scene_001",
            scene_type=SceneType.EXPLORATION,
            title=f"{player_name}的冒險開始",
            description=f"你發現自己站在{location}的入口處，周圍充滿了未知的可能性...",
            location=location,
            time_of_day="黃昏",
            weather="微風輕拂",
            atmosphere=SceneMood.MYSTERIOUS,
            turn_number=0,
        )

        # Add player to scene
        initial_scene.add_character("player")

        return initial_scene

    def _setup_character_roster(
        self, session_id: str, custom_characters: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, GameCharacter]:
        """Setup initial character roster for the session"""

        characters = {}

        # Add player character
        characters["player"] = GameCharacter(
            character_id="player",
            name="玩家",
            role=CharacterRole.PLAYER,
            personality_traits=["好奇", "勇敢", "適應力強"],
            speaking_style="第一人稱視角",
            background_story="一個踏上冒險旅程的探索者",
            relationships={},
            motivations=["探索未知", "成長", "幫助他人"],
        )

        # Add default NPCs
        default_npcs = [
            {
                "id": "wise_guide",
                "name": "智者艾莉亞",
                "role": CharacterRole.COMPANION,
                "traits": ["智慧", "神秘", "指導性"],
                "style": "充滿智慧且神秘的語調",
                "background": "一位古老的魔法師，掌握著許多秘密知識",
                "motivations": ["指導後輩", "保護古老知識", "維持平衡"],
            },
            {
                "id": "merchant_bob",
                "name": "商人巴布",
                "role": CharacterRole.NPC,
                "traits": ["精明", "友善", "商業頭腦"],
                "style": "熱情且具說服力的商業語調",
                "background": "一個經驗豐富的旅行商人",
                "motivations": ["獲利", "建立人脈", "分享故事"],
            },
        ]

        for npc_data in default_npcs:
            characters[npc_data["id"]] = GameCharacter(
                character_id=npc_data["id"],
                name=npc_data["name"],
                role=npc_data["role"],
                personality_traits=npc_data["traits"],
                speaking_style=npc_data["style"],
                background_story=npc_data["background"],
                relationships={},
                motivations=npc_data["motivations"],
            )

        # Add custom characters if provided
        if custom_characters:
            for i, char_data in enumerate(custom_characters):
                char_id = char_data.get("id", f"custom_{i}")
                characters[char_id] = GameCharacter(
                    character_id=char_id,
                    name=char_data.get("name", f"角色{i}"),
                    role=CharacterRole.NPC,
                    personality_traits=char_data.get("traits", ["未知"]),
                    speaking_style=char_data.get("style", "普通語調"),
                    background_story=char_data.get("background", "神秘的背景"),
                    relationships={},
                    motivations=char_data.get("motivations", ["未知目標"]),
                )

        return characters

    def make_choice(self, session_id: str, choice_id: str) -> Dict[str, Any]:
        """Enhanced choice execution with full context processing"""

        # Execute base choice
        result = self.base_engine.make_choice(session_id, choice_id)

        # Enhanced processing if context memory exists
        if session_id in self.context_memories:
            context_memory = self.context_memories[session_id]

            # Update character states based on choice
            self._update_character_states(session_id, choice_id, result)

            # Process relationship changes
            self._process_relationship_changes(session_id, result)

            # Update scene context
            self._update_scene_context(session_id, result)

            # Add to narrative memory
            context_memory.add_narrative_memory(
                {
                    "type": "choice_executed",
                    "choice_id": choice_id,
                    "result": result,
                    "success": result.get("success", True),
                }
            )

        return result

    def _update_character_states(
        self, session_id: str, choice_id: str, result: Dict[str, Any]
    ):
        """Update character emotional and relationship states"""

        if session_id not in self.character_managers:
            return

        characters = self.character_managers[session_id]
        context_memory = self.context_memories[session_id]

        # Update based on choice consequences
        if "relationships" in result.get("consequences", {}):
            for char_id, relationship_change in result["consequences"][
                "relationships"
            ].items():
                if char_id in characters:
                    context_memory.update_character_relationship(
                        "player", char_id, relationship_change
                    )

                    # Update character mood based on relationship change
                    character = characters[char_id]
                    if relationship_change > 0:
                        character.current_state = CharacterState.HAPPY
                    elif relationship_change < -5:
                        character.current_state = CharacterState.SUSPICIOUS

    def _process_relationship_changes(self, session_id: str, result: Dict[str, Any]):
        """Process and apply relationship changes from choices"""

        consequences = result.get("consequences", {})
        if "relationship" in consequences:
            context_memory = self.context_memories[session_id]

            for char_id, change in consequences["relationship"].items():
                context_memory.update_character_relationship("player", char_id, change)

    def _update_scene_context(self, session_id: str, result: Dict[str, Any]):
        """Update scene context based on choice results"""

        if session_id not in self.context_memories:
            return

        context_memory = self.context_memories[session_id]
        current_scene = context_memory.get_current_scene()

        if not current_scene:
            return

        consequences = result.get("consequences", {})

        # Check for scene transitions
        if "scene_transition" in consequences:
            new_location = consequences["scene_transition"]
            self._trigger_scene_transition(session_id, new_location)

        # Update scene mood based on choice outcome
        if not result.get("success", True):
            current_scene.atmosphere = SceneMood.TENSE
        elif "positive_outcome" in consequences:
            current_scene.atmosphere = SceneMood.PEACEFUL

    def _trigger_scene_transition(self, session_id: str, new_location: str):
        """Trigger transition to a new scene"""

        context_memory = self.context_memories[session_id]
        current_scene = context_memory.get_current_scene()

        # Create new scene
        scene_count = len(context_memory.scene_history) + 1
        new_scene = SceneContext(
            scene_id=f"scene_{scene_count:03d}",
            scene_type=SceneType.EXPLORATION,  # Can be refined based on location
            title=f"探索{new_location}",
            description=f"你來到了{new_location}，這裡有著不同的氛圍和可能性...",
            location=new_location,
            time_of_day=current_scene.time_of_day if current_scene else "未知",
            weather=current_scene.weather if current_scene else "未知",
            atmosphere=SceneMood.MYSTERIOUS,
            previous_scene_id=current_scene.scene_id if current_scene else None,
            turn_number=0,
        )

        context_memory.transition_to_scene(new_scene)
        logger.info(
            f"Scene transition: {current_scene.location if current_scene else 'unknown'} → {new_location}"
        )

    def get_enhanced_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session information including context"""

        # Get base session info
        base_info = self.base_engine.get_session_summary(session_id)

        # Add enhanced context if available
        if session_id in self.context_memories:
            context_memory = self.context_memories[session_id]
            current_scene = context_memory.get_current_scene()

            enhanced_info = {
                "context_memory": {
                    "current_scene": {
                        "location": (
                            current_scene.location if current_scene else "unknown"
                        ),
                        "atmosphere": (
                            current_scene.atmosphere if current_scene else "unknown"
                        ),
                        "present_characters": len(
                            context_memory.get_characters_in_scene()
                        ),
                    },
                    "total_scenes": len(context_memory.scene_history)
                    + (1 if current_scene else 0),
                    "character_count": len(context_memory.characters),
                    "world_flags": len(context_memory.world_flags),
                    "narrative_memories": len(context_memory.narrative_memory),
                },
                "relationship_summary": self._get_relationship_summary(session_id),
            }

            base_info.update(enhanced_info)

        return base_info

    def _get_relationship_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of character relationships"""

        if session_id not in self.context_memories:
            return {}

        context_memory = self.context_memories[session_id]
        relationships = context_memory.character_relationships.get("player", {})  # type: ignore

        summary = {
            "total_relationships": len(relationships),
            "positive_relationships": len(
                [r for r in relationships.values() if r > 20]
            ),
            "negative_relationships": len(
                [r for r in relationships.values() if r < -20]
            ),
            "neutral_relationships": len(
                [r for r in relationships.values() if -20 <= r <= 20]
            ),
        }

        return summary

    # Advanced scene generators
    def _generate_dynamic_exploration(self, context: Dict[str, Any]) -> str:
        """Generate dynamic exploration narrative"""
        location = context.get("location", "未知地點")
        atmosphere = context.get("atmosphere", "neutral")
        discoveries = context.get("discoveries", [])

        exploration_templates = [
            f"在{location}中探索時，你注意到了一些有趣的細節",
            f"仔細觀察{location}的環境，你發現了一些線索",
            f"在{location}的深處，似乎隱藏著什麼秘密",
            f"探索{location}時，周圍的{atmosphere}氣氛讓你更加謹慎",
        ]

        base_narrative = random.choice(exploration_templates)

        if discoveries:
            discovery_text = f"。你發現了：{', '.join(discoveries[:2])}"
            base_narrative += discovery_text

        return base_narrative + "。每一步都可能帶來新的發現..."

    def _generate_character_dialogue(self, context: Dict[str, Any]) -> str:
        """Generate character dialogue narrative"""
        characters = context.get("present_characters", [])
        relationships = context.get("relationships", {})
        mood = context.get("scene_mood", "neutral")

        if not characters:
            return "這裡很安靜，沒有人可以交談..."

        char_name = characters[0] if characters else "某人"
        relationship_level = relationships.get(char_name, 0)

        dialogue_templates = {
            "high_relationship": [
                f"與{char_name}的對話充滿了默契和理解",
                f"{char_name}對你展現出信任和友善",
                f"你和{char_name}之間的交流自然而愉快",
            ],
            "low_relationship": [
                f"與{char_name}的對話顯得有些緊張",
                f"{char_name}對你保持著警戒的態度",
                f"你需要更小心地選擇與{char_name}交談的方式",
            ],
            "neutral": [
                f"與{char_name}進行著平常的對話",
                f"{char_name}友善地與你交談",
                f"你們的對話在{mood}的氛圍中進行",
            ],
        }

        if relationship_level > 20:
            templates = dialogue_templates["high_relationship"]
        elif relationship_level < -10:
            templates = dialogue_templates["low_relationship"]
        else:
            templates = dialogue_templates["neutral"]

        return random.choice(templates) + "。對話中透露出複雜的情感和隱藏的動機..."

    def _generate_contextual_choices(
        self, context_memory: StoryContextMemory
    ) -> List[ContextualChoice]:
        """Generate contextual choices based on current story state (修正版本)"""

        choices = []
        current_scene = context_memory.get_current_scene()
        present_characters = context_memory.get_characters_in_scene()

        # Import here to avoid circular import issues
        from .story_system import ContextualChoice

        # Basic exploration choice
        choices.append(
            ContextualChoice(
                choice_id="contextual_explore",
                base_text="仔細觀察周圍環境",
                choice_type="exploration",
                consequences={"stats": {"intelligence": 1}},
                success_chance=0.8,
            )
        )

        # Character interaction choices
        for character in present_characters:
            if character.character_id != "player":
                # 獲取關係值
                relationship = context_memory.character_relationships.get(
                    "player", {}
                ).get(character.character_id, 0)

                # 根據關係調整選擇
                if relationship > 20:
                    choice_text = f"與{character.name}進行深度交流"
                    success_chance = 0.9
                    consequences = {"relationships": {character.character_id: 2}}
                elif relationship < -10:
                    choice_text = f"嘗試改善與{character.name}的關係"
                    success_chance = 0.5
                    consequences = {"relationships": {character.character_id: 5}}
                else:
                    choice_text = f"與{character.name}交談"
                    success_chance = 0.7
                    consequences = {"relationships": {character.character_id: 2}}

                choices.append(
                    ContextualChoice(
                        choice_id=f"talk_to_{character.character_id}",
                        base_text=choice_text,
                        choice_type="dialogue",
                        required_characters=[character.character_id],
                        consequences=consequences,  # 明確提供 consequences 參數
                        success_chance=success_chance,
                    )
                )

        # Scene-specific choices
        if current_scene:
            scene_type = (
                current_scene.scene_type.value
                if hasattr(current_scene.scene_type, "value")
                else str(current_scene.scene_type)
            )

            if scene_type == "exploration":
                choices.append(
                    ContextualChoice(
                        choice_id="deep_exploration",
                        base_text=f"深入探索{current_scene.location}",
                        choice_type="exploration",
                        consequences={
                            "discovery_chance": True,
                            "stats": {"intelligence": 1},
                        },
                        success_chance=0.6,
                    )
                )
            elif scene_type == "dialogue":
                choices.append(
                    ContextualChoice(
                        choice_id="listen_carefully",
                        base_text="仔細聆聽對話內容",
                        choice_type="social",
                        consequences={"insight_gained": True, "stats": {"wisdom": 1}},
                        success_chance=0.8,
                    )
                )
            elif scene_type == "combat":
                choices.append(
                    ContextualChoice(
                        choice_id="tactical_retreat",
                        base_text="進行戰術性撤退",
                        choice_type="combat",
                        consequences={"safe_escape": True, "stats": {"health": 5}},
                        success_chance=0.7,
                    )
                )

        return choices

    def _generate_contextual_combat(self, context: Dict[str, Any]) -> str:
        """Generate contextual combat narrative"""
        enemies = context.get("enemies", ["敵人"])
        player_stats = context.get("player_stats", {})
        weapons = context.get("available_weapons", [])
        environment = context.get("location", "戰場")

        combat_intensity = "激烈" if len(enemies) > 1 else "緊張"

        base_narrative = f"在{environment}中，{combat_intensity}的戰鬥正在進行"

        # Add tactical elements based on context
        tactical_elements = []

        if weapons:
            tactical_elements.append(f"你可以使用{weapons[0]}進行攻擊")

        if player_stats.get("agility", 0) > 10:
            tactical_elements.append("你的敏捷讓你能夠靈活移動")

        if len(enemies) > 1:
            tactical_elements.append("多個敵人讓戰況更加複雜")

        if tactical_elements:
            base_narrative += "。" + "，".join(tactical_elements[:2])

        return base_narrative + "。戰鬥的結果將影響接下來的故事發展..."

    def _generate_environmental_puzzle(self, context: Dict[str, Any]) -> str:
        """Generate environmental puzzle narrative"""
        location = context.get("location", "神秘房間")
        puzzle_elements = context.get("puzzle_elements", ["機關", "符文", "石碑"])
        difficulty = context.get("difficulty", "medium")

        difficulty_adjectives = {
            "easy": "看起來相對簡單",
            "medium": "需要仔細思考",
            "hard": "極其複雜",
            "extreme": "幾乎無法理解",
        }

        difficulty_desc = difficulty_adjectives.get(difficulty, "需要仔細思考")
        elements_desc = "、".join(puzzle_elements[:3])

        base_narrative = f"在{location}中，你面前的謎題{difficulty_desc}。{elements_desc}以某種特殊的方式組合在一起"

        # Add environmental clues
        environmental_clues = [
            "牆上的古老雕刻似乎提供了線索",
            "光線的角度暗示著解題的方向",
            "周圍的聲音模式可能是關鍵",
            "地面上的痕跡顯示了前人的嘗試",
        ]

        clue = random.choice(environmental_clues)

        return base_narrative + f"。{clue}，這個謎題與環境緊密相關，需要仔細觀察..."

    def _generate_emotional_scene(self, context: Dict[str, Any]) -> str:
        """Generate emotional scene narrative"""
        emotion_type = context.get("dominant_emotion", "neutral")
        characters = context.get("present_characters", [])
        recent_events = context.get("recent_events", [])

        emotional_frameworks = {
            "joy": "歡樂和滿足的氣氛充滿了整個場景",
            "sadness": "一種深沉的憂傷瀰漫在空氣中",
            "anger": "緊張和憤怒的情緒讓氣氛變得凝重",
            "fear": "不安和恐懼的陰霾籠罩著每個人",
            "surprise": "出人意料的發展讓所有人都感到震驚",
            "neutral": "平靜的情緒讓人能夠冷靜思考",
        }

        base_framework = emotional_frameworks.get(
            emotion_type, emotional_frameworks["neutral"]
        )

        # Add character emotional responses
        if characters:
            char_responses = []
            for char in characters[:2]:
                if char != "player":
                    char_responses.append(f"{char}的表情反映出內心的複雜情感")

            if char_responses:
                base_framework += "。" + "，".join(char_responses)

        # Add event impact
        if recent_events:
            event_impact = "剛才發生的事情讓每個人都在消化和反思"
            base_framework += f"。{event_impact}"

        return (
            base_framework + "。情感的波動在這個場景中尤其明顯，影響著每個人的決定..."
        )

    async def process_enhanced_turn(
        self, session_id: str, player_input: str, choice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process turn with enhanced context management"""

        if session_id not in self.active_sessions:
            raise GameError(f"Session not found: {session_id}")

        context_memory = self.active_sessions[session_id]

        # Execute choice if provided
        choice_result = None
        if choice_id:
            choice_result = await self._execute_contextual_choice(
                context_memory, choice_id, player_input
            )

        # Generate narrative with full context
        narrative_result = await self.narrative_generator.generate_contextual_narrative(
            context_memory, player_input, choice_result
        )

        # Update context memory
        await self._update_context_memory(
            context_memory, player_input, narrative_result, choice_result
        )

        # Generate new contextual choices
        new_choices = self._generate_contextual_choices(context_memory)

        # Update scene if needed
        if narrative_result.get("scene_changes", {}).get("new_scene_needed"):
            await self._create_new_scene(context_memory, narrative_result)

        return {
            "session_id": session_id,
            "narrative": narrative_result["main_narrative"],
            "character_dialogues": narrative_result["character_dialogues"],
            "available_choices": [
                {
                    "choice_id": choice.choice_id,
                    "text": choice.get_display_text(context_memory),
                    "type": choice.choice_type,
                    "difficulty": choice.difficulty,
                }
                for choice in new_choices
            ],
            "scene_info": {
                "current_scene": (
                    context_memory.get_current_scene().to_dict()  # type: ignore
                    if context_memory.get_current_scene()
                    else {}
                ),
                "present_characters": [
                    char.name for char in context_memory.get_characters_in_scene()
                ],
                "mood_shift": narrative_result.get("mood_shift"),
                "scene_changes": narrative_result.get("scene_changes", {}),
            },
            "context_summary": {
                "total_scenes": len(context_memory.scenes),
                "total_characters": len(context_memory.characters),
                "active_plot_points": len(context_memory.main_plot_points),
                "player_decisions": len(context_memory.player_decisions),
            },
        }

    async def _execute_contextual_choice(
        self, context_memory: StoryContextMemory, choice_id: str, player_input: str
    ) -> Dict[str, Any]:
        """Execute a contextual choice and apply consequences"""

        # Find the choice across all categories
        selected_choice = None
        for category_choices in self.contextual_choices.values():
            for choice in category_choices:
                if choice.choice_id == choice_id:
                    selected_choice = choice
                    break
            if selected_choice:
                break

        if not selected_choice:
            raise GameError(f"Choice not found: {choice_id}")

        # Check if choice can be executed
        can_execute, reason = selected_choice.can_execute(
            context_memory, {}, []  # TODO: Add actual player stats and inventory
        )

        if not can_execute:
            return {"success": False, "reason": reason}

        # Apply consequences
        result = {"success": True, "consequences": {}}

        # Apply relationship changes
        for char_id, change in selected_choice.relationship_changes.items():
            if char_id == "any":
                # Apply to all present NPCs
                for char in context_memory.get_characters_in_scene():
                    if char.character_id != "player":
                        context_memory.update_character_relationship(
                            "player", char.character_id, change
                        )
            else:
                context_memory.update_character_relationship("player", char_id, change)

        # Apply flag changes
        for flag, value in selected_choice.flag_changes.items():
            context_memory.world_flags[flag] = value

        # Record in narrative memory
        context_memory.add_narrative_memory(
            {
                "type": "contextual_choice_executed",
                "choice_id": choice_id,
                "result": result,
            }
        )

        return result

    async def _update_context_memory(
        self,
        context_memory: StoryContextMemory,
        player_input: str,
        narrative_result: Dict[str, Any],
        choice_result: Optional[Dict[str, Any]],
    ):
        """Update context memory with new information"""

        current_scene = context_memory.get_current_scene()
        if current_scene:
            current_scene.turn_number += 1

            # Update plot points if narrative suggests new developments
            narrative = narrative_result["main_narrative"].lower()

            # Simple keyword-based plot point detection
            plot_keywords = {
                "發現": "發現了重要線索",
                "秘密": "揭露了隱藏的秘密",
                "危險": "遇到了新的威脅",
                "盟友": "結識了潛在的盟友",
                "敵人": "識別出新的敵對勢力",
            }

            for keyword, plot_point in plot_keywords.items():
                if (
                    keyword in narrative
                    and plot_point not in context_memory.main_plot_points
                ):
                    context_memory.main_plot_points.append(plot_point)
                    current_scene.add_plot_point(plot_point)

        # Update character states based on interactions
        for dialogue in narrative_result.get("character_dialogues", []):
            char_id = dialogue["character_id"]
            character = context_memory.get_character(char_id)
            if character:
                # Update character emotional state based on interaction
                new_state = self._infer_character_state_from_dialogue(
                    dialogue["content"], dialogue.get("emotional_state", "neutral")
                )
                if new_state:
                    character.current_state = CharacterState(new_state)

    def _infer_character_state_from_dialogue(
        self, dialogue: str, current_state: str
    ) -> Optional[str]:
        """Infer character emotional state from dialogue content"""

        dialogue_lower = dialogue.lower()

        state_indicators = {
            "happy": ["高興", "開心", "快樂", "滿意", "愉快"],
            "angry": ["生氣", "憤怒", "不滿", "惱火", "怒"],
            "fearful": ["害怕", "恐懼", "擔心", "緊張", "不安"],
            "excited": ["興奮", "激動", "期待", "熱情", "活力"],
            "suspicious": ["懷疑", "不信", "質疑", "猜忌", "警惕"],
        }

        for state, indicators in state_indicators.items():
            if any(indicator in dialogue_lower for indicator in indicators):
                return state

        return None

    async def _create_new_scene(
        self, context_memory: StoryContextMemory, narrative_result: Dict[str, Any]
    ):
        """Create new scene based on narrative development"""

        scene_changes = narrative_result.get("scene_changes", {})
        current_scene = context_memory.get_current_scene()

        # Generate new scene ID
        scene_count = (
            len(context_memory.scenes) + 1
        )  # 使用 scenes 字典而不是不存在的屬性
        new_scene_id = f"scene_{scene_count:03d}"

        # Determine new scene properties
        new_location = current_scene.location if current_scene else "未知地點"
        new_atmosphere = SceneMood.NEUTRAL

        if scene_changes.get("location_change"):
            # TODO: Extract actual location from narrative
            new_location = "新地點"

        if narrative_result.get("mood_shift"):
            mood_map = {
                "tense": SceneMood.TENSE,
                "peaceful": SceneMood.PEACEFUL,
                "mysterious": SceneMood.MYSTERIOUS,
                "exciting": SceneMood.EXCITING,
            }
            new_atmosphere = mood_map.get(
                narrative_result["mood_shift"], SceneMood.NEUTRAL
            )

        # Create new scene
        new_scene = SceneContext(
            scene_id=new_scene_id,
            scene_type=SceneType.EXPLORATION,
            title=f"場景 {scene_count}",
            description=narrative_result.get("main_narrative", "新的場景展開...")[:200]
            + "...",
            location=new_location,
            time_of_day=current_scene.time_of_day if current_scene else "未知",
            weather=current_scene.weather if current_scene else "未知",
            atmosphere=new_atmosphere,
            previous_scene_id=context_memory.current_scene_id,
            turn_number=0,
        )

        # Add current characters to new scene
        for char in context_memory.get_characters_in_scene():
            new_scene.add_character(char.character_id)
            char.current_location = new_location

        # Update context memory
        context_memory.add_scene(new_scene)
        context_memory.current_scene_id = new_scene_id
        context_memory.current_scene = new_scene

        logger.info(f"Created new scene: {new_scene_id} at {new_location}")


class EmotionalAI:
    """Simple emotional AI for character state management"""

    def __init__(self):
        self.emotion_mappings = {
            "positive_interaction": CharacterState.HAPPY,
            "conflict": CharacterState.ANGRY,
            "mystery": CharacterState.SUSPICIOUS,
            "danger": CharacterState.FEARFUL,
            "achievement": CharacterState.EXCITED,
        }

    def process_emotional_context(
        self, event_type: str, characters: List[GameCharacter]
    ):
        """Process emotional context for characters"""
        if event_type in self.emotion_mappings:
            target_emotion = self.emotion_mappings[event_type]
            for character in characters:
                if character.role != CharacterRole.PLAYER:
                    character.current_state = target_emotion


# Factory function implementation
def create_enhanced_story_engine(
    config_dir: Optional[Path] = None, cache_root: Optional[str] = None
) -> EnhancedStoryEngine:
    """
    Factory function to create enhanced story engine

    Args:
        config_dir: Configuration directory path
        cache_root: Cache root directory (optional)

    Returns:
        EnhancedStoryEngine: Fully initialized enhanced engine
    """
    try:
        logger.info("Creating enhanced story engine...")
        engine = EnhancedStoryEngine(config_dir)
        logger.info("Enhanced story engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create enhanced story engine: {e}")
        # Fallback to creating a wrapper around basic engine
        logger.warning("Falling back to basic story engine with enhanced wrapper")

        basic_engine = StoryEngine(config_dir, enhanced_mode=True)

        # Create minimal enhanced wrapper
        enhanced_wrapper = EnhancedStoryEngine.__new__(EnhancedStoryEngine)
        enhanced_wrapper.base_engine = basic_engine
        enhanced_wrapper.context_memories = {}
        enhanced_wrapper.character_managers = {}

        return enhanced_wrapper
