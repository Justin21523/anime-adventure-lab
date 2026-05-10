"""
Stat System Core
Defines all numerical attributes for characters in the game.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple


# ---------------------------------------------------------------------------
# Stat categories
# ---------------------------------------------------------------------------

@dataclass
class CombatStats:
    """Combat-related numerical attributes."""

    attack: int = 10            # 攻擊力
    defense: int = 10           # 防禦力
    speed: int = 10             # 速度（決定先手順序）
    hit_rate: float = 0.85      # 命中率 (0.0-1.0)
    crit_rate: float = 0.05     # 暴擊率 (0.0-1.0)
    crit_multiplier: float = 1.5 # 暴擊倍率
    evasion: float = 0.10       # 閃避率 (0.0-1.0)
    counter_rate: float = 0.05  # 反擊率 (0.0-1.0)
    hp: int = 100               # 生命值
    max_hp: int = 100           # 最大生命值
    mp: int = 50                # 魔力值
    max_mp: int = 50            # 最大魔力值

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CombatStats":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SocialStats:
    """Social interaction attributes."""

    charm: int = 10             # 魅力
    persuasion: int = 10        # 說服力
    deception: int = 10         # 欺騙
    empathy: int = 10           # 共情
    intimidation: int = 10      # 威嚇
    knowledge: int = 10         # 知識廣度
    leadership: int = 10        # 領導力

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SocialStats":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SurvivalStats:
    """Survival and environmental attributes."""

    danger_level: int = 0       # 危險值 (0-100, 越高越危險)
    stamina: int = 100          # 體力 (0-100)
    morale: int = 100           # 士氣 (0-100)
    hunger: int = 0             # 飢餓 (0=飽, 100=極度飢餓)
    stress: int = 0             # 壓力 (0=放鬆, 100=崩潰邊緣)
    suspicion: int = 0          # 被懷疑程度 (0-100)
    stealth: int = 10           # 潛行能力
    perception: int = 10        # 觀察力

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurvivalStats":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ResourceStats:
    """Economic and resource attributes."""

    gold: int = 100             # 金幣
    food: int = 5               # 食物數量
    potions: int = 3            # 藥水數量
    antidotes: int = 1          # 解毒劑
    materials: Dict[str, int] = field(default_factory=dict)  # 各類素材

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceStats":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Combined character stats
# ---------------------------------------------------------------------------

@dataclass
class CharacterStats:
    """Full character stat sheet — combines all categories."""

    # Core attributes (used everywhere)
    strength: int = 10
    intelligence: int = 10
    wisdom: int = 10
    dexterity: int = 10
    charisma: int = 10
    luck: int = 10

    # Derived categories
    combat: CombatStats = field(default_factory=CombatStats)
    social: SocialStats = field(default_factory=SocialStats)
    survival: SurvivalStats = field(default_factory=SurvivalStats)
    resources: ResourceStats = field(default_factory=ResourceStats)

    # Progression
    experience: int = 0
    level: int = 1
    xp_to_next: int = 100

    # Flags
    is_alive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strength": self.strength,
            "intelligence": self.intelligence,
            "wisdom": self.wisdom,
            "dexterity": self.dexterity,
            "charisma": self.charisma,
            "luck": self.luck,
            "combat": self.combat.to_dict(),
            "social": self.social.to_dict(),
            "survival": self.survival.to_dict(),
            "resources": self.resources.to_dict(),
            "experience": self.experience,
            "level": self.level,
            "xp_to_next": self.xp_to_next,
            "is_alive": self.is_alive,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterStats":
        combat_data = data.get("combat", {})
        social_data = data.get("social", {})
        survival_data = data.get("survival", {})
        resource_data = data.get("resources", {})

        return cls(
            strength=data.get("strength", 10),
            intelligence=data.get("intelligence", 10),
            wisdom=data.get("wisdom", 10),
            dexterity=data.get("dexterity", 10),
            charisma=data.get("charisma", 10),
            luck=data.get("luck", 10),
            combat=CombatStats.from_dict(combat_data),
            social=SocialStats.from_dict(social_data),
            survival=SurvivalStats.from_dict(survival_data),
            resources=ResourceStats.from_dict(resource_data),
            experience=data.get("experience", 0),
            level=data.get("level", 1),
            xp_to_next=data.get("xp_to_next", 100),
            is_alive=data.get("is_alive", True),
        )

    # ------------------------------------------------------------------
    # Stat manipulation helpers
    # ------------------------------------------------------------------

    def apply_bonus(self, stat_path: str, value: int) -> None:
        """Apply a bonus to a nested stat via dot notation.

        Examples:
            apply_bonus("combat.attack", 5)
            apply_bonus("social.charm", -3)
            apply_bonus("strength", 2)
        """
        parts = stat_path.split(".")
        if len(parts) == 1:
            # Top-level attribute
            attr = parts[0]
            if hasattr(self, attr) and isinstance(getattr(self, attr), (int, float)):
                setattr(self, attr, getattr(self, attr) + value)
        elif len(parts) == 2:
            category, attr = parts
            target = getattr(self, category, None)
            if target is not None and hasattr(target, attr):
                current = getattr(target, attr)
                if isinstance(current, (int, float)):
                    setattr(target, attr, current + value)

    def get_effective(self, stat_path: str) -> float:
        """Get effective stat value (clamped to valid ranges where needed)."""
        parts = stat_path.split(".")
        if len(parts) == 1:
            val = getattr(self, parts[0], 0)
            return max(0, val)
        elif len(parts) == 2:
            category, attr = parts
            target = getattr(self, category, None)
            if target is not None:
                val = getattr(target, attr, 0)
                # Clamp rates to 0-1
                if attr in ("hit_rate", "crit_rate", "evasion", "counter_rate"):
                    return max(0.0, min(1.0, float(val)))
                return max(0, val)
        return 0

    # ------------------------------------------------------------------
    # Experience and leveling
    # ------------------------------------------------------------------

    def gain_xp(self, amount: int) -> Tuple[int, bool]:
        """Gain XP. Returns (remaining_xp, leveled_up)."""
        self.experience += amount
        leveled = False
        while self.experience >= self.xp_to_next:
            self.experience -= self.xp_to_next
            self.level_up()
            leveled = True
        return self.experience, leveled

    def level_up(self) -> Dict[str, int]:
        """Level up — increase core stats and derived caps."""
        gains = {
            "strength": 1,
            "intelligence": 1,
            "wisdom": 1,
            "dexterity": 1,
            "charisma": 1,
            "luck": 1,
        }
        for attr, delta in gains.items():
            setattr(self, attr, getattr(self, attr) + delta)

        # Derived bonuses
        self.combat.max_hp += 10
        self.combat.hp = self.combat.max_hp  # Full heal on level up
        self.combat.max_mp += 5
        self.combat.mp = self.combat.max_mp
        self.combat.attack += 1
        self.combat.defense += 1

        # Increase XP threshold
        self.xp_to_next = int(self.xp_to_next * 1.3)
        self.level += 1

        return gains

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def take_damage(self, amount: int) -> int:
        """Take damage. Returns actual damage dealt (after defense)."""
        effective = max(1, amount - self.combat.defense)
        self.combat.hp -= effective
        if self.combat.hp <= 0:
            self.combat.hp = 0
            self.is_alive = False
        return effective

    def heal(self, amount: int) -> int:
        """Heal. Returns actual HP restored."""
        before = self.combat.hp
        self.combat.hp = min(self.combat.max_hp, self.combat.hp + amount)
        if not self.is_alive and self.combat.hp > 0:
            self.is_alive = True
        return self.combat.hp - before

    def is_exhausted(self) -> bool:
        """Check if character is exhausted (low stamina + low morale)."""
        return (self.survival.stamina < 20 or self.survival.morale < 20)

    def is_starving(self) -> bool:
        """Check if character is starving."""
        return self.survival.hunger >= 90

    def is_panicking(self) -> bool:
        """Check if character is panicking."""
        return self.survival.stress >= 90
