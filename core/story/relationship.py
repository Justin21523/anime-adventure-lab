"""
Relationship System
Tracks affection, trust, reputation, and faction alignment for each NPC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FactionStance(Enum):
    HOSTILE = "hostile"
    WARY = "wary"
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    ALLIED = "allied"

    @property
    def icon(self) -> str:
        return {
            self.HOSTILE: "⚔️",
            self.WARY: "👁️",
            self.NEUTRAL: "😐",
            self.FRIENDLY: "😊",
            self.ALLIED: "🤝",
        }[self]


class RelationshipTier(Enum):
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    FRIEND = "friend"
    CLOSE_FRIEND = "close_friend"
    ALLY = "ally"
    RIVAL = "rival"
    ENEMY = "enemy"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RelationshipEvent:
    """A single event that affected a relationship."""
    event_id: str
    description: str
    affection_delta: int = 0
    trust_delta: int = 0
    turn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "description": self.description,
            "affection_delta": self.affection_delta,
            "trust_delta": self.trust_delta,
            "turn": self.turn,
        }


@dataclass
class Relationship:
    """Relationship state with a single NPC."""
    character_id: str
    character_name: str = ""

    # Core metrics (0-100)
    affection: int = 50        # 好感度
    trust: int = 50            # 信任度
    reputation: int = 50       # 聲望

    # Faction alignment
    faction_stance: FactionStance = FactionStance.NEUTRAL

    # Interaction history
    history: List[RelationshipEvent] = field(default_factory=list)
    interaction_count: int = 0

    # Unlocked interactions (based on relationship tier)
    unlocked_interactions: List[str] = field(default_factory=list)

    @property
    def tier(self) -> RelationshipTier:
        """Calculate relationship tier from metrics."""
        score = self.affection + self.trust
        if score >= 160:
            return RelationshipTier.ALLY
        if score >= 130:
            return RelationshipTier.CLOSE_FRIEND
        if score >= 100:
            return RelationshipTier.FRIEND
        if score >= 70:
            return RelationshipTier.ACQUAINTANCE
        if score >= 40:
            return RelationshipTier.STRANGER
        if self.affection < 20:
            return RelationshipTier.ENEMY
        return RelationshipTier.RIVAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "character_id": self.character_id,
            "character_name": self.character_name,
            "affection": self.affection,
            "trust": self.trust,
            "reputation": self.reputation,
            "faction_stance": self.faction_stance.value,
            "history": [e.to_dict() for e in self.history],
            "interaction_count": self.interaction_count,
            "unlocked_interactions": self.unlocked_interactions.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        return cls(
            character_id=data["character_id"],
            character_name=data.get("character_name", ""),
            affection=data.get("affection", 50),
            trust=data.get("trust", 50),
            reputation=data.get("reputation", 50),
            faction_stance=FactionStance(data.get("faction_stance", "neutral")),
            history=[RelationshipEvent(**e) for e in data.get("history", [])],
            interaction_count=data.get("interaction_count", 0),
            unlocked_interactions=data.get("unlocked_interactions", []),
        )


# ---------------------------------------------------------------------------
# Relationship Manager
# ---------------------------------------------------------------------------

class RelationshipManager:
    """Manages all NPC relationships for the player."""

    # Interaction unlocks by affection threshold
    INTERACTION_THRESHOLDS = {
        20: ["basic_greeting"],
        40: ["small_talk", "ask_question"],
        60: ["request_help", "share_secret"],
        75: ["invite_travel", "romantic_option"],
        90: ["ultimate_sacrifice", "marriage_option"],
    }

    def __init__(self):
        self.relationships: Dict[str, Relationship] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def add_npc(self, character_id: str, name: str,
                initial_affection: int = 50,
                initial_trust: int = 50,
                faction: FactionStance = FactionStance.NEUTRAL) -> Relationship:
        """Register an NPC in the relationship system."""
        rel = Relationship(
            character_id=character_id,
            character_name=name,
            affection=initial_affection,
            trust=initial_trust,
            faction_stance=faction,
        )
        self.relationships[character_id] = rel
        self._update_unlocked(rel)
        return rel

    # ------------------------------------------------------------------
    # Modify
    # ------------------------------------------------------------------

    def modify_affection(
        self, char_id: str, delta: int, reason: str = "", turn: int = 0
    ) -> Tuple[int, str]:
        """Modify affection. Returns (new_value, message)."""
        rel = self.relationships.get(char_id)
        if rel is None:
            return 0, "未知角色"

        old = rel.affection
        rel.affection = max(0, min(100, old + delta))
        rel.interaction_count += 1

        emoji = "❤️" if delta > 0 else "💔" if delta < 0 else "😐"
        msg = f"{emoji} {rel.character_name} 的好感度 {'↑' if delta > 0 else '↓'} ({rel.affection})"
        if reason:
            msg += f" — {reason}"

        # Record event
        if delta != 0:
            rel.history.append(RelationshipEvent(
                event_id=f"aff_{rel.interaction_count}",
                description=reason or "unknown",
                affection_delta=delta,
                turn=turn,
            ))

        # Check tier change
        old_tier = None  # Would need to track this
        self._update_unlocked(rel)

        return rel.affection, msg

    def modify_trust(
        self, char_id: str, delta: int, reason: str = "", turn: int = 0
    ) -> Tuple[int, str]:
        """Modify trust. Returns (new_value, message)."""
        rel = self.relationships.get(char_id)
        if rel is None:
            return 0, "未知角色"

        old = rel.trust
        rel.trust = max(0, min(100, old + delta))
        rel.interaction_count += 1

        emoji = "🤝" if delta > 0 else "👎" if delta < 0 else "😐"
        msg = f"{emoji} {rel.character_name} 的信任度 {'↑' if delta > 0 else '↓'} ({rel.trust})"

        if delta != 0:
            rel.history.append(RelationshipEvent(
                event_id=f"tr_{rel.interaction_count}",
                description=reason or "unknown",
                trust_delta=delta,
                turn=turn,
            ))

        return rel.trust, msg

    def modify_reputation(
        self, char_id: str, delta: int, reason: str = ""
    ) -> Tuple[int, str]:
        """Modify reputation. Returns (new_value, message)."""
        rel = self.relationships.get(char_id)
        if rel is None:
            return 0, "未知角色"

        rel.reputation = max(0, min(100, rel.reputation + delta))
        emoji = "⭐" if delta > 0 else "📉" if delta < 0 else ""
        return rel.reputation, f"{emoji} {rel.character_name} 的聲望 ({rel.reputation})"

    def set_faction_stance(self, char_id: str, stance: FactionStance) -> bool:
        """Update faction stance."""
        rel = self.relationships.get(char_id)
        if rel is None:
            return False
        rel.faction_stance = stance
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, char_id: str) -> Optional[Relationship]:
        return self.relationships.get(char_id)

    def get_tier(self, char_id: str) -> Optional[RelationshipTier]:
        rel = self.relationships.get(char_id)
        return rel.tier if rel else None

    def get_stance(self, char_id: str) -> Optional[FactionStance]:
        rel = self.relationships.get(char_id)
        return rel.faction_stance if rel else None

    def is_friendly(self, char_id: str) -> bool:
        rel = self.relationships.get(char_id)
        return rel is not None and rel.faction_stance in (
            FactionStance.FRIENDLY, FactionStance.ALLIED
        )

    def is_hostile(self, char_id: str) -> bool:
        rel = self.relationships.get(char_id)
        return rel is not None and rel.faction_stance == FactionStance.HOSTILE

    def can_interact(self, char_id: str, interaction: str) -> bool:
        """Check if an interaction is unlocked."""
        rel = self.relationships.get(char_id)
        if rel is None:
            return False
        return interaction in rel.unlocked_interactions

    def get_unlocked_interactions(self, char_id: str) -> List[str]:
        rel = self.relationships.get(char_id)
        return list(rel.unlocked_interactions) if rel else []

    def list_all(self) -> List[Relationship]:
        return list(self.relationships.values())

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def format_relationship(self, char_id: str) -> str:
        """Generate a text summary of a relationship."""
        rel = self.relationships.get(char_id)
        if rel is None:
            return f"未認識的角色: {char_id}"

        tier = rel.tier
        stance = rel.faction_stance
        lines = [
            f"📋 {rel.character_name}",
            f"  關係等級: {tier.value}",
            f"  陣營立場: {stance.icon} {stance.value}",
            f"  好感度: {'█' * (rel.affection // 5):<20} {rel.affection}/100",
            f"  信任度: {'█' * (rel.trust // 5):<20} {rel.trust}/100",
            f"  聲望: {'█' * (rel.reputation // 5):<20} {rel.reputation}/100",
        ]
        return "\n".join(lines)

    def format_all_brief(self) -> str:
        """Brief overview of all relationships."""
        if not self.relationships:
            return "還沒有認識任何人。"

        lines = ["📊 人物關係總覽:"]
        for rel in sorted(self.relationships.values(), key=lambda r: r.affection, reverse=True):
            tier = rel.tier
            stance = rel.faction_stance
            lines.append(
                f"  {stance.icon} {rel.character_name} "
                f"[{tier.value}] "
                f"❤️{rel.affection} 🤝{rel.trust}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            char_id: rel.to_dict()
            for char_id, rel in self.relationships.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipManager":
        mgr = cls()
        for char_id, rel_data in data.items():
            mgr.relationships[char_id] = Relationship.from_dict(rel_data)
        return mgr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_unlocked(self, rel: Relationship) -> None:
        """Update unlocked interactions based on affection level."""
        unlocked = []
        for threshold, interactions in self.INTERACTION_THRESHOLDS.items():
            if rel.affection >= threshold:
                unlocked.extend(interactions)
        rel.unlocked_interactions = unlocked
