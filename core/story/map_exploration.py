"""
Map and Exploration System
Location management, travel, events, and discovery.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EventTrigger:
    """An event that can occur at a location."""
    event_id: str
    name: str
    description: str
    trigger_type: str = "arrival"  # arrival, random, time, quest
    chance: float = 1.0            # 0.0-1.0 for random triggers
    min_level: int = 1
    required_flags: List[str] = field(default_factory=list)
    triggered: bool = False
    is_hidden: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "name": self.name,
            "description": self.description,
            "trigger_type": self.trigger_type,
            "chance": self.chance,
            "min_level": self.min_level,
            "required_flags": self.required_flags.copy(),
            "triggered": self.triggered,
            "is_hidden": self.is_hidden,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventTrigger":
        return cls(**data)


@dataclass
class Location:
    """A place in the game world."""
    location_id: str
    name: str
    description: str
    danger_level: int = 0           # 0-100
    connections: List[str] = field(default_factory=list)  # location_ids
    events: List[EventTrigger] = field(default_factory=list)
    npcs: List[str] = field(default_factory=list)          # character_ids
    shops: List[str] = field(default_factory=list)         # shop_ids
    resources: List[str] = field(default_factory=list)     # gatherable items
    explored: bool = False
    image_prompt: str = ""          # T2I scene generation prompt
    discovered_items: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location_id": self.location_id,
            "name": self.name,
            "description": self.description,
            "danger_level": self.danger_level,
            "connections": self.connections.copy(),
            "events": [e.to_dict() for e in self.events],
            "npcs": self.npcs.copy(),
            "shops": self.shops.copy(),
            "resources": self.resources.copy(),
            "explored": self.explored,
            "image_prompt": self.image_prompt,
            "discovered_items": self.discovered_items.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Location":
        loc = cls(
            location_id=data["location_id"],
            name=data["name"],
            description=data["description"],
            danger_level=data.get("danger_level", 0),
            connections=data.get("connections", []),
            npcs=data.get("npcs", []),
            shops=data.get("shops", []),
            resources=data.get("resources", []),
            explored=data.get("explored", False),
            image_prompt=data.get("image_prompt", ""),
            discovered_items=data.get("discovered_items", []),
        )
        for edata in data.get("events", []):
            loc.events.append(EventTrigger.from_dict(edata))
        return loc


@dataclass
class TravelResult:
    """Result of a travel attempt."""
    success: bool
    from_location: str
    to_location: str
    message: str
    events_triggered: List[Dict[str, Any]] = field(default_factory=list)
    damage_taken: int = 0
    stamina_cost: int = 0


@dataclass
class Discovery:
    """A hidden discovery at a location."""
    discovery_id: str
    name: str
    description: str
    reward_type: str = "item"  # item, info, quest, event
    reward: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Map System
# ---------------------------------------------------------------------------

class MapSystem:
    """Manages the world map, locations, and travel."""

    def __init__(self):
        self.locations: Dict[str, Location] = {}
        self.current_location: Optional[str] = None
        self.visited: List[str] = []

    # ------------------------------------------------------------------
    # Location management
    # ------------------------------------------------------------------

    def add_location(self, location: Location) -> None:
        self.locations[location.location_id] = location

    def get_location(self, location_id: str) -> Optional[Location]:
        return self.locations.get(location_id)

    def get_current_location(self) -> Optional[Location]:
        return self.locations.get(self.current_location) if self.current_location else None

    def set_current_location(self, location_id: str) -> bool:
        if location_id in self.locations:
            self.current_location = location_id
            self.locations[location_id].explored = True
            if location_id not in self.visited:
                self.visited.append(location_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Travel
    # ------------------------------------------------------------------

    def travel_to(
        self,
        target_location_id: str,
        player_stats,  # CharacterStats
    ) -> TravelResult:
        """
        Attempt to travel to a location.
        """
        current = self.get_current_location()
        target = self.locations.get(target_location_id)

        if current is None:
            return TravelResult(
                success=False,
                from_location="",
                to_location=target_location_id,
                message="你還在尋找起點...",
            )

        if target is None:
            return TravelResult(
                success=False,
                from_location=current.location_id,
                to_location=target_location_id,
                message="未知的地點",
            )

        # Check connection
        if target_location_id not in current.connections:
            return TravelResult(
                success=False,
                from_location=current.location_id,
                to_location=target_location_id,
                message=f"從 {current.name} 無法直接到達 {target.name}",
            )

        # Stamina cost
        stamina_cost = max(1, target.danger_level // 10)
        if player_stats.survival.stamina < stamina_cost:
            return TravelResult(
                success=False,
                from_location=current.location_id,
                to_location=target_location_id,
                message=f"體力不足以前往 {target.name} (需要 {stamina_cost} 體力)",
            )

        # Danger check
        danger = target.danger_level
        if danger > 50 and player_stats.combat.level < danger // 20:
            # Warning but allow
            pass

        # Execute travel
        player_stats.survival.stamina -= stamina_cost
        player_stats.survival.danger_level = danger

        # Trigger arrival events
        events = []
        for event in target.events:
            if event.trigger_type == "arrival" and not event.triggered:
                if event.min_level <= player_stats.combat.level:
                    if random.random() <= event.chance:
                        event.triggered = True
                        events.append(event.to_dict())

        msg = f"🗺️ 從 {current.name} 前往 {target.name}"
        if events:
            msg += f" — 觸發事件: {events[0].get('name', '未知')}"

        # Set new location
        self.set_current_location(target_location_id)

        return TravelResult(
            success=True,
            from_location=current.location_id,
            to_location=target_location_id,
            message=msg,
            events_triggered=events,
            stamina_cost=stamina_cost,
        )

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def get_available_locations(self) -> List[Location]:
        """Get locations reachable from current position."""
        current = self.get_current_location()
        if current is None:
            return []
        return [
            self.locations[lid]
            for lid in current.connections
            if lid in self.locations
        ]

    def discover_hidden(
        self,
        location_id: str,
        perception: int = 10,
    ) -> Optional[Discovery]:
        """
        Attempt to discover hidden things at a location.
        Success chance based on perception.
        """
        loc = self.locations.get(location_id)
        if loc is None:
            return None

        # Perception check
        chance = min(0.1 + perception * 0.02, 0.8)
        if random.random() > chance:
            return None

        # Random discovery
        discoveries = [
            Discovery(
                discovery_id=f"hidden_{location_id}_{uuid.uuid4().hex[:4]}",
                name="隱藏的寶藏",
                description="你發現了一個隱藏的箱子！",
                reward_type="item",
                reward={"gold": random.randint(10, 100)},
            ),
            Discovery(
                discovery_id=f"hidden_{location_id}_{uuid.uuid4().hex[:4]}",
                name="神秘的線索",
                description="你注意到了一些不尋常的痕跡...",
                reward_type="info",
                reward={"hint": "有人在這裡住過"},
            ),
            Discovery(
                discovery_id=f"hidden_{location_id}_{uuid.uuid4().hex[:4]}",
                name="隱藏的通道",
                description="你發現了一條隱藏的密道！",
                reward_type="event",
                reward={"unlock_connection": True},
            ),
        ]

        discovery = random.choice(discoveries)
        return discovery

    def explore_area(
        self,
        location_id: str,
        player_stats,  # CharacterStats
    ) -> Dict[str, Any]:
        """
        Thoroughly explore a location. Costs stamina, may find resources.
        """
        loc = self.locations.get(location_id)
        if loc is None:
            return {"success": False, "message": "未知的地點"}

        # Stamina cost
        if player_stats.survival.stamina < 5:
            return {"success": False, "message": "體力不足，無法探索"}

        player_stats.survival.stamina -= 5
        loc.explored = True

        # Resource gathering
        found = []
        for resource in loc.resources:
            if random.random() < 0.3 + player_stats.survival.perception * 0.01:
                found.append(resource)
                if resource not in loc.discovered_items:
                    loc.discovered_items.append(resource)

        # Hidden discovery
        discovery = self.discover_hidden(location_id, player_stats.survival.perception)

        result = {
            "success": True,
            "message": f"探索了 {loc.name}",
            "resources_found": found,
            "location_explored": True,
        }

        if discovery:
            result["discovery"] = {
                "name": discovery.name,
                "description": discovery.description,
                "reward_type": discovery.reward_type,
                "reward": discovery.reward,
            }

        return result

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def format_location(self, location_id: str) -> str:
        """Generate a text description of a location."""
        loc = self.locations.get(location_id)
        if loc is None:
            return f"未知地點: {location_id}"

        lines = [
            f"📍 {loc.name}",
            f"  {loc.description}",
            f"  危險度: {'⚠️' * (loc.danger_level // 20)} ({loc.danger_level})",
        ]

        if loc.connections:
            connected = [self.locations[c].name for c in loc.connections if c in self.locations]
            lines.append(f"  可前往: {', '.join(connected)}")

        if loc.npcs:
            lines.append(f"  NPC: {len(loc.npcs)} 人")

        if loc.shops:
            lines.append(f"  商店: {len(loc.shops)} 家")

        if loc.explored:
            lines.append(f"  ✅ 已探索")
        else:
            lines.append(f"  ❓ 未探索")

        return "\n".join(lines)

    def format_map_overview(self) -> str:
        """Brief map overview."""
        if not self.locations:
            return "🗺️ 世界地圖還沒有任何地點。"

        current = self.current_location or "?"
        lines = [
            f"🗺️ 世界地圖 ({len(self.locations)} 個地點)",
            f"  📍 當前: {self.locations[current].name if current in self.locations else '未知'}",
            f"  🔍 已探索: {sum(1 for l in self.locations.values() if l.explored)}/{len(self.locations)}",
        ]

        for loc in self.locations.values():
            marker = "📍" if loc.location_id == current else ("✅" if loc.explored else "❓")
            danger = "⚠️" if loc.danger_level > 50 else "🟢" if loc.danger_level < 30 else "🟡"
            lines.append(f"  {marker} {danger} {loc.name}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "locations": {lid: l.to_dict() for lid, l in self.locations.items()},
            "current_location": self.current_location,
            "visited": self.visited.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MapSystem":
        ms = cls()
        ms.current_location = data.get("current_location")
        ms.visited = data.get("visited", [])
        for lid, ldata in data.get("locations", {}).items():
            ms.locations[lid] = Location.from_dict(ldata)
        return ms
