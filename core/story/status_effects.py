"""
Status Effects System
Buff/Debuff management with duration tracking and stack rules.
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

class EffectType(Enum):
    # Positive
    BUFF = "buff"
    SHIELD = "shield"
    REGENERATION = "regeneration"

    # Negative
    POISON = "poison"
    PARALYSIS = "paralysis"
    SILENCE = "silence"
    BLIND = "blind"
    CONFUSION = "confusion"
    BLEED = "bleed"
    FEAR = "fear"
    CURSE = "curse"

    # Environmental
    BURN = "burn"
    FREEZE = "freeze"
    SHOCK = "shock"


class EffectSource(Enum):
    COMBAT = "combat"
    SKILL = "skill"
    ENVIRONMENT = "environment"
    ITEM = "item"
    NARRATIVE = "narrative"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StatusEffect:
    """A single status effect that can be applied to a character."""
    effect_id: str
    name: str
    effect_type: EffectType
    source: EffectSource = EffectSource.NARRATIVE

    # Stat modifications: "stat_path": delta
    # e.g. {"combat.attack": -3, "combat.hit_rate": -0.1}
    stat_modifiers: Dict[str, float] = field(default_factory=dict)

    # Duration in turns; -1 = permanent until removed
    duration: int = 1
    turns_remaining: int = 1

    # Application chance (0.0-1.0)
    chance: float = 1.0

    # Damage per tick (for poison, burn, bleed)
    tick_damage: int = 0

    # Can stack? If not, replacing with same effect refreshes duration
    stackable: bool = False

    # Human-readable description
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "name": self.name,
            "effect_type": self.effect_type.value,
            "source": self.source.value,
            "stat_modifiers": self.stat_modifiers.copy(),
            "duration": self.duration,
            "turns_remaining": self.turns_remaining,
            "chance": self.chance,
            "tick_damage": self.tick_damage,
            "stackable": self.stackable,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatusEffect":
        return cls(
            effect_id=data["effect_id"],
            name=data["name"],
            effect_type=EffectType(data["effect_type"]),
            source=EffectSource(data.get("source", "narrative")),
            stat_modifiers=data.get("stat_modifiers", {}),
            duration=data.get("duration", 1),
            turns_remaining=data.get("turns_remaining", data.get("duration", 1)),
            chance=data.get("chance", 1.0),
            tick_damage=data.get("tick_damage", 0),
            stackable=data.get("stackable", False),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Common effect presets
# ---------------------------------------------------------------------------

COMMON_EFFECTS: Dict[str, StatusEffect] = {
    "poison_minor": StatusEffect(
        effect_id="poison_minor",
        name="輕微中毒",
        effect_type=EffectType.POISON,
        source=EffectSource.COMBAT,
        duration=3,
        turns_remaining=3,
        tick_damage=5,
        description="每回合受到 5 點中毒傷害，持續 3 回合",
    ),
    "poison_severe": StatusEffect(
        effect_id="poison_severe",
        name="劇毒",
        effect_type=EffectType.POISON,
        source=EffectSource.COMBAT,
        duration=5,
        turns_remaining=5,
        tick_damage=15,
        description="每回合受到 15 點劇毒傷害，持續 5 回合",
    ),
    "paralysis": StatusEffect(
        effect_id="paralysis",
        name="麻痺",
        effect_type=EffectType.PARALYSIS,
        source=EffectSource.COMBAT,
        duration=2,
        turns_remaining=2,
        stat_modifiers={"combat.speed": -99, "combat.hit_rate": -0.5},
        description="行動延遲，命中率大幅下降",
    ),
    "burn": StatusEffect(
        effect_id="burn",
        name="灼燒",
        effect_type=EffectType.BURN,
        source=EffectSource.ENVIRONMENT,
        duration=4,
        turns_remaining=4,
        tick_damage=8,
        description="每回合受到 8 點火焰傷害",
    ),
    "bleed": StatusEffect(
        effect_id="bleed",
        name="流血",
        effect_type=EffectType.BLEED,
        source=EffectSource.COMBAT,
        duration=3,
        turns_remaining=3,
        tick_damage=6,
        description="每回合流失 6 點生命值",
    ),
    "blind": StatusEffect(
        effect_id="blind",
        name="失明",
        effect_type=EffectType.BLIND,
        source=EffectSource.COMBAT,
        duration=2,
        turns_remaining=2,
        stat_modifiers={"combat.hit_rate": -0.7, "combat.evasion": -0.3},
        description="命中率大幅下降，難以閃避",
    ),
    "fear": StatusEffect(
        effect_id="fear",
        name="恐懼",
        effect_type=EffectType.FEAR,
        source=EffectSource.COMBAT,
        duration=2,
        turns_remaining=2,
        stat_modifiers={"combat.attack": -5, "combat.defense": -3},
        description="戰鬥能力下降",
    ),
    "attack_up": StatusEffect(
        effect_id="attack_up",
        name="攻擊提升",
        effect_type=EffectType.BUFF,
        source=EffectSource.SKILL,
        duration=3,
        turns_remaining=3,
        stat_modifiers={"combat.attack": 5},
        description="攻擊力提升 5",
    ),
    "defense_up": StatusEffect(
        effect_id="defense_up",
        name="防禦提升",
        effect_type=EffectType.BUFF,
        source=EffectSource.SKILL,
        duration=3,
        turns_remaining=3,
        stat_modifiers={"combat.defense": 5},
        description="防禦力提升 5",
    ),
    "speed_up": StatusEffect(
        effect_id="speed_up",
        name="敏捷提升",
        effect_type=EffectType.BUFF,
        source=EffectSource.SKILL,
        duration=3,
        turns_remaining=3,
        stat_modifiers={"combat.speed": 5, "combat.evasion": 0.1},
        description="速度和閃避提升",
    ),
    "regen": StatusEffect(
        effect_id="regen",
        name="生命回復",
        effect_type=EffectType.REGENERATION,
        source=EffectSource.ITEM,
        duration=3,
        turns_remaining=3,
        tick_damage=-10,  # Negative = heal
        description="每回合恢復 10 點生命值",
    ),
}


# ---------------------------------------------------------------------------
# Effect Manager
# ---------------------------------------------------------------------------

class EffectManager:
    """Manages status effects for a single target (character/entity)."""

    def __init__(self, target_id: str):
        self.target_id = target_id
        self.active_effects: List[StatusEffect] = []

    # ------------------------------------------------------------------
    # Apply / Remove
    # ------------------------------------------------------------------

    def apply(self, effect: StatusEffect) -> Tuple[bool, str]:
        """Apply a status effect. Returns (applied, message)."""

        # Check chance
        import random
        if random.random() > effect.chance:
            return False, f"{effect.name} 沒有命中"

        # Check for existing same-type effect
        existing = self._find_same_type(effect.effect_type)
        if existing and not effect.stackable:
            # Replace
            idx = self.active_effects.index(existing)
            self.active_effects[idx] = effect
            return True, f"{effect.name} 替換了 {existing.name}"

        self.active_effects.append(effect)
        return True, f"受到 {effect.name} 效果"

    def remove(self, effect_id: str) -> Tuple[bool, str]:
        """Remove a specific effect."""
        for i, effect in enumerate(self.active_effects):
            if effect.effect_id == effect_id:
                removed = self.active_effects.pop(i)
                return True, f"解除了 {removed.name}"
        return False, f"沒有找到 {effect_id}"

    def remove_type(self, effect_type: EffectType) -> int:
        """Remove all effects of a given type. Returns count."""
        before = len(self.active_effects)
        self.active_effects = [e for e in self.active_effects if e.effect_type != effect_type]
        return before - len(self.active_effects)

    def clear_negative(self) -> int:
        """Remove all negative effects. Returns count removed."""
        before = len(self.active_effects)
        self.active_effects = [
            e for e in self.active_effects
            if e.effect_type in (EffectType.BUFF, EffectType.SHIELD, EffectType.REGENERATION)
        ]
        return before - len(self.active_effects)

    # ------------------------------------------------------------------
    # Tick (per-turn processing)
    # ------------------------------------------------------------------

    def tick(self) -> List[Dict[str, Any]]:
        """
        Process one turn tick.
        Returns list of event dicts: {"type": "damage"/"heal"/"expire", "value": int, "effect": str}
        """
        events = []
        to_remove = []

        for effect in self.active_effects:
            # Tick damage/heal
            if effect.tick_damage != 0:
                events.append({
                    "type": "heal" if effect.tick_damage < 0 else "damage",
                    "value": abs(effect.tick_damage),
                    "effect_id": effect.effect_id,
                    "effect_name": effect.name,
                })

            # Decrement duration
            if effect.duration >= 0:
                effect.turns_remaining -= 1
                if effect.turns_remaining <= 0:
                    to_remove.append(effect)
                    events.append({
                        "type": "expire",
                        "effect_id": effect.effect_id,
                        "effect_name": effect.name,
                    })

        for effect in to_remove:
            self.active_effects.remove(effect)

        return events

    # ------------------------------------------------------------------
    # Stat modification aggregation
    # ------------------------------------------------------------------

    def get_stat_modifiers(self) -> Dict[str, float]:
        """Aggregate all active stat modifiers."""
        modifiers: Dict[str, float] = {}
        for effect in self.active_effects:
            for stat, delta in effect.stat_modifiers.items():
                modifiers[stat] = modifiers.get(stat, 0) + delta
        return modifiers

    def get_effective_mod(self, stat_path: str) -> float:
        """Get total modifier for a specific stat."""
        modifiers = self.get_stat_modifiers()
        return modifiers.get(stat_path, 0)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_effect(self, effect_id: str) -> bool:
        return any(e.effect_id == effect_id for e in self.active_effects)

    def has_type(self, effect_type: EffectType) -> bool:
        return any(e.effect_type == effect_type for e in self.active_effects)

    def get_active(self) -> List[StatusEffect]:
        return list(self.active_effects)

    def get_active_names(self) -> List[str]:
        return [e.name for e in self.active_effects]

    def get_buffs(self) -> List[StatusEffect]:
        return [e for e in self.active_effects
                if e.effect_type in (EffectType.BUFF, EffectType.SHIELD, EffectType.REGENERATION)]

    def get_debuffs(self) -> List[StatusEffect]:
        return [e for e in self.active_effects
                if e.effect_type not in (EffectType.BUFF, EffectType.SHIELD, EffectType.REGENERATION)]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "effects": [e.to_dict() for e in self.active_effects],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffectManager":
        mgr = cls(target_id=data["target_id"])
        for edata in data.get("effects", []):
            mgr.active_effects.append(StatusEffect.from_dict(edata))
        return mgr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_same_type(self, effect_type: EffectType) -> Optional[StatusEffect]:
        for e in self.active_effects:
            if e.effect_type == effect_type:
                return e
        return None

    def format_display(self) -> str:
        """Generate a text summary of active effects."""
        if not self.active_effects:
            return ""

        parts = ["✨ 當前效果:"]
        for e in self.active_effects:
            icon = "🟢" if e.effect_type in (EffectType.BUFF, EffectType.SHIELD, EffectType.REGENERATION) else "🔴"
            duration = f" ({e.turns_remaining}回合)" if e.duration >= 0 else " (永久)"
            parts.append(f"  {icon} {e.name}{duration}")
        return "\n".join(parts)
