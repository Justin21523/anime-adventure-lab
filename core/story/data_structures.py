# core/story/data_structures.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import uuid

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")


class RelationType(Enum):
    FRIEND = "friend"
    ENEMY = "enemy"
    ROMANTIC = "romantic"
    NEUTRAL = "neutral"
    ALLY = "ally"
    RIVAL = "rival"


@dataclass
class Persona:
    """Character persona with personality, background and speaking style"""

    name: str
    age: Optional[int] = None
    personality: List[str] = field(default_factory=list)  # ["內向", "好奇", "謹慎"]
    background: str = ""
    speaking_style: str = ""  # "正式但溫和", "活潑直接", "神秘寡言"
    appearance: str = ""
    goals: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)
    memory_preferences: Dict[str, float] = field(
        default_factory=dict
    )  # {"romance": 0.8, "adventure": 0.6}

    def to_prompt_context(self) -> str:
        """Convert persona to prompt context"""
        context = f"角色: {self.name}"
        if self.age:
            context += f" ({self.age}歲)"
        context += f"\n性格: {', '.join(self.personality)}"
        if self.background:
            context += f"\n背景: {self.background}"
        if self.speaking_style:
            context += f"\n說話風格: {self.speaking_style}"
        if self.goals:
            context += f"\n目標: {'; '.join(self.goals)}"
        return context


class EventType(Enum):
    DIALOGUE = "dialogue"
    CHOICE = "choice"
    BATTLE = "battle"
    DISCOVERY = "discovery"
    RELATIONSHIP_CHANGE = "relationship_change"
    ITEM_ACQUIRED = "item_acquired"
    LOCATION_DISCOVERED = "location_discovered"


@dataclass
class Relationship:
    character_id: str
    # character_a: str
    # character_b: str
    # strength: float  # 0.0 to 1.0
    # description: str = ""
    relation_type: RelationType
    affinity: int = 50  # 0-100, 50 is neutral
    trust: int = 50  # 0-100, 50 is neutral
    notes: str = ""
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class InventoryItem:
    item_id: str
    name: str
    quantity: int = 1
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    acquired_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorldFlag:
    flag_id: str
    value: Any  # can be bool, int, str, etc.
    description: str = ""
    set_at: datetime = field(default_factory=datetime.now)


@dataclass
class TimelineEvent:
    event_id: str
    event_type: EventType
    description: str
    location: str = ""
    characters_involved: List[str] = field(default_factory=list)
    consequences: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    citations: List[str] = field(default_factory=list)  # RAG citations


@dataclass
class LocationState:
    location_id: str
    name: str
    discovered: bool = False
    accessible: bool = True
    properties: Dict[str, Any] = field(default_factory=dict)
    characters_present: Set[str] = field(default_factory=set)
    items_present: Set[str] = field(default_factory=set)


@dataclass
class GameState:
    """Current game state tracking progress and context"""

    # Basic info
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scene_id: str = ""
    world_id: str = ""
    current_scene: str = ""
    current_location: str = ""
    turn_count: int = 0

    # Player state
    player_name: str = ""
    player_level: int = 1
    player_health: int = 100
    player_energy: int = 100

    # Relationships with NPCs
    relationships: List[Relationship] = field(default_factory=list)

    # World state flags
    world_flags: Dict[str, WorldFlag] = field(default_factory=dict)
    flags: Dict[str, Any] = field(
        default_factory=dict
    )  # {"met_alice": True, "has_key": False}

    # Inventory management
    inventory: Dict[str, InventoryItem] = field(default_factory=dict)
    max_inventory_size: int = 50

    choice_history: List[Dict[str, Any]] = field(default_factory=list)

    # Timeline and history
    timeline: List[TimelineEvent] = field(default_factory=list)

    # Location states
    locations: Dict[str, LocationState] = field(default_factory=dict)

    # Current context for RAG
    active_personas: List[str] = field(default_factory=list)
    scene_context: str = ""
    last_rag_citations: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def update_relationship(
        self,
        character_id: str,
        relation_type: RelationType = None,
        affinity_delta: int = 0,
        trust_delta: int = 0,
        notes: str = "",
    ):
        """Update relationship with a character"""
        if character_id not in self.relationships:
            self.relationships[character_id] = Relationship(
                character_id=character_id,
                relation_type=relation_type or RelationType.NEUTRAL,
            )

        rel = self.relationships[character_id]
        if relation_type:
            rel.relation_type = relation_type
        rel.affinity = max(0, min(100, rel.affinity + affinity_delta))
        rel.trust = max(0, min(100, rel.trust + trust_delta))
        if notes:
            rel.notes = notes
        rel.last_updated = datetime.now()

        # Add to timeline
        self.add_timeline_event(
            event_type=EventType.RELATIONSHIP_CHANGE,
            description=f"Relationship with {character_id} changed: {relation_type.value if relation_type else 'updated'}",
            characters_involved=[character_id],
            consequences={
                "affinity_delta": affinity_delta,
                "trust_delta": trust_delta,
                "new_affinity": rel.affinity,
                "new_trust": rel.trust,
            },
        )

    def add_item(
        self,
        item_id: str,
        name: str,
        quantity: int = 1,
        description: str = "",
        properties: Dict[str, Any] = None,
    ) -> bool:
        """Add item to inventory, returns False if inventory full"""
        if (
            len(self.inventory) >= self.max_inventory_size
            and item_id not in self.inventory
        ):
            return False

        if item_id in self.inventory:
            self.inventory[item_id].quantity += quantity
        else:
            self.inventory[item_id] = InventoryItem(
                item_id=item_id,
                name=name,
                quantity=quantity,
                description=description,
                properties=properties or {},
            )

        # Add to timeline
        self.add_timeline_event(
            event_type=EventType.ITEM_ACQUIRED,
            description=f"Acquired {quantity}x {name}",
            consequences={"item_id": item_id, "quantity": quantity},
        )
        return True

    def remove_item(self, item_id: str, quantity: int = 1) -> bool:
        """Remove item from inventory, returns False if not enough items"""
        if item_id not in self.inventory or self.inventory[item_id].quantity < quantity:
            return False

        self.inventory[item_id].quantity -= quantity
        if self.inventory[item_id].quantity <= 0:
            del self.inventory[item_id]
        return True

    def set_flag(self, flag_id: str, value: Any, description: str = ""):
        """Set a world flag"""
        self.world_flags[flag_id] = WorldFlag(
            flag_id=flag_id, value=value, description=description
        )

    def get_flag(self, flag_id: str, default: Any = None) -> Any:
        """Get world flag value"""
        return self.world_flags.get(flag_id, WorldFlag(flag_id, default)).value

    def add_timeline_event(
        self,
        event_type: EventType,
        description: str,
        location: str = "",
        characters_involved: List[str] = None,
        consequences: Dict[str, Any] = None,
        citations: List[str] = None,
    ):
        """Add event to timeline"""
        event = TimelineEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            description=description,
            location=location or self.current_location,
            characters_involved=characters_involved or [],
            consequences=consequences or {},
            citations=citations or [],
        )
        self.timeline.append(event)
        self.last_updated = datetime.now()

    def discover_location(
        self, location_id: str, name: str, properties: Dict[str, Any] = None
    ):
        """Discover a new location"""
        if location_id not in self.locations:
            self.locations[location_id] = LocationState(
                location_id=location_id,
                name=name,
                discovered=True,
                properties=properties or {},
            )

            self.add_timeline_event(
                event_type=EventType.LOCATION_DISCOVERED,
                description=f"Discovered location: {name}",
                location=location_id,
            )

    def move_to_location(self, location_id: str):
        """Move player to a location"""
        if location_id in self.locations and self.locations[location_id].accessible:
            self.current_location = location_id
            return True
        return False

    def get_relationship_summary(self) -> Dict[str, str]:
        """Get summary of all relationships"""
        summary = {}
        for char_id, rel in self.relationships.items():
            summary[char_id] = (
                f"{rel.relation_type.value} (A:{rel.affinity}, T:{rel.trust})"
            )
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "world_id": self.world_id,
            "current_scene": self.current_scene,
            "current_location": self.current_location,
            "turn_count": self.turn_count,
            "player_name": self.player_name,
            "player_level": self.player_level,
            "player_health": self.player_health,
            "player_energy": self.player_energy,
            "relationships": {
                k: {
                    "character_id": v.character_id,
                    "relation_type": v.relation_type.value,
                    "affinity": v.affinity,
                    "trust": v.trust,
                    "notes": v.notes,
                    "last_updated": v.last_updated.isoformat(),
                }
                for k, v in self.relationships.items()
            },
            "inventory": {
                k: {
                    "item_id": v.item_id,
                    "name": v.name,
                    "quantity": v.quantity,
                    "description": v.description,
                    "properties": v.properties,
                    "acquired_at": v.acquired_at.isoformat(),
                }
                for k, v in self.inventory.items()
            },
            "world_flags": {
                k: {
                    "flag_id": v.flag_id,
                    "value": v.value,
                    "description": v.description,
                    "set_at": v.set_at.isoformat(),
                }
                for k, v in self.world_flags.items()
            },
            "timeline": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "description": e.description,
                    "location": e.location,
                    "characters_involved": e.characters_involved,
                    "consequences": e.consequences,
                    "timestamp": e.timestamp.isoformat(),
                    "citations": e.citations,
                }
                for e in self.timeline
            ],
            "locations": {
                k: {
                    "location_id": v.location_id,
                    "name": v.name,
                    "discovered": v.discovered,
                    "accessible": v.accessible,
                    "properties": v.properties,
                    "characters_present": list(v.characters_present),
                    "items_present": list(v.items_present),
                }
                for k, v in self.locations.items()
            },
            "active_personas": self.active_personas,
            "scene_context": self.scene_context,
            "last_rag_citations": self.last_rag_citations,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameState":
        """Create GameState from dictionary"""
        # This would include full deserialization logic
        # Simplified version for now
        state = cls()
        state.session_id = data.get("session_id", state.session_id)
        state.world_id = data.get("world_id", "")
        state.current_scene = data.get("current_scene", "")
        # ... (full implementation would restore all fields)
        return state


@dataclass
class DialogueEntry:
    """Single dialogue entry"""

    speaker: str
    text: str
    emotion: Optional[str] = None  # "開心", "困惑", "生氣" etc


@dataclass
class Choice:
    """Player choice option"""

    id: str
    text: str
    description: str = ""
    requirements: Dict[str, Any] = field(default_factory=dict)  # {"has_key": True}
    consequences: Dict[str, Any] = field(default_factory=dict)  # {"trust_alice": +0.1}


@dataclass
class TurnRequest:
    """Request for a story turn"""

    player_input: str
    persona: Persona
    game_state: GameState
    choice_id: Optional[str] = None  # If player selected a choice


@dataclass
class TurnResponse:
    """Response from story engine"""

    narration: str
    dialogues: List[DialogueEntry] = field(default_factory=list)
    choices: List[Choice] = field(default_factory=list)
    updated_state: Optional[GameState] = None
    scene_change: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            "narration": self.narration,
            "dialogues": [
                {"speaker": d.speaker, "text": d.text, "emotion": d.emotion}
                for d in self.dialogues
            ],
            "choices": [
                {"id": c.id, "text": c.text, "description": c.description}
                for c in self.choices
            ],
            "scene_change": self.scene_change,
        }
