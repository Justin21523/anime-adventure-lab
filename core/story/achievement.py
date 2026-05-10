"""
Achievement System
Tracks achievements, unlocks, and progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Achievement:
    """A single achievement definition."""
    achievement_id: str
    name: str
    description: str
    icon: str = "🏆"

    # Condition function: receives game session dict, returns bool
    # Stored as a serializable description; actual check done by AchievementManager
    condition_type: str = "manual"  # manual, combat, exploration, social, economy
    condition_params: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"type": "kill_count", "threshold": 100}

    # Rewards
    reward_gold: int = 0
    reward_xp: int = 0
    reward_stat_bonus: Dict[str, int] = field(default_factory=dict)

    # State
    unlocked: bool = False
    unlocked_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "achievement_id": self.achievement_id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "condition_type": self.condition_type,
            "condition_params": self.condition_params.copy(),
            "reward_gold": self.reward_gold,
            "reward_xp": self.reward_xp,
            "reward_stat_bonus": self.reward_stat_bonus.copy(),
            "unlocked": self.unlocked,
            "unlocked_at": self.unlocked_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Achievement":
        return cls(**data)


@dataclass
class AchievementProgress:
    """Track progress toward an achievement."""
    achievement_id: str
    current_value: float = 0.0
    target_value: float = 1.0
    is_complete: bool = False

    @property
    def percentage(self) -> float:
        if self.target_value <= 0:
            return 1.0 if self.is_complete else 0.0
        return min(1.0, self.current_value / self.target_value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "achievement_id": self.achievement_id,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "is_complete": self.is_complete,
            "percentage": round(self.percentage, 2),
        }


# ---------------------------------------------------------------------------
# Achievement Manager
# ---------------------------------------------------------------------------

class AchievementManager:
    """Manages achievement definitions and tracking."""

    def __init__(self):
        self.achievements: Dict[str, Achievement] = {}
        self.progress: Dict[str, AchievementProgress] = {}
        self.statistics: Dict[str, int] = {}  # Track counters

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, achievement: Achievement) -> None:
        self.achievements[achievement.achievement_id] = achievement
        if not achievement.unlocked:
            params = achievement.condition_params
            self.progress[achievement.achievement_id] = AchievementProgress(
                achievement_id=achievement.achievement_id,
                target_value=params.get("threshold", 1),
            )

    def register_default_achievements(self) -> None:
        """Register built-in default achievements."""
        defaults = [
            Achievement(
                achievement_id="first_step",
                name="初次冒險",
                description="完成第一個回合",
                icon="👣",
                condition_type="manual",
                condition_params={"threshold": 1},
                reward_xp=10,
            ),
            Achievement(
                achievement_id="first_battle",
                name="初戰告捷",
                description="贏得第一場戰鬥",
                icon="⚔️",
                condition_type="combat",
                condition_params={"type": "combat_wins", "threshold": 1},
                reward_gold=50,
                reward_xp=30,
            ),
            Achievement(
                achievement_id="battle_veteran",
                name="戰鬥老手",
                description="贏得 50 場戰鬥",
                icon="🛡️",
                condition_type="combat",
                condition_params={"type": "combat_wins", "threshold": 50},
                reward_gold=500,
                reward_xp=200,
                reward_stat_bonus={"combat.attack": 2},
            ),
            Achievement(
                achievement_id="explorer",
                name="探索者",
                description="探索 10 個地點",
                icon="🗺️",
                condition_type="exploration",
                condition_params={"type": "locations_explored", "threshold": 10},
                reward_gold=100,
                reward_xp=50,
            ),
            Achievement(
                achievement_id="friend_maker",
                name="社交達人",
                description="讓 5 個 NPC 的好感度達到 80+",
                icon="🤝",
                condition_type="social",
                condition_params={"type": "high_affection_count", "threshold": 5},
                reward_gold=200,
                reward_xp=100,
                reward_stat_bonus={"social.charm": 3},
            ),
            Achievement(
                achievement_id="level_10",
                name="十級達人",
                description="角色達到 10 級",
                icon="⭐",
                condition_type="manual",
                condition_params={"type": "level", "threshold": 10},
                reward_gold=300,
                reward_xp=150,
            ),
            Achievement(
                achievement_id="rich",
                name="富翁",
                description="累積 10000 金幣",
                icon="💰",
                condition_type="economy",
                condition_params={"type": "gold_earned", "threshold": 10000},
                reward_gold=1000,
                reward_xp=200,
            ),
            Achievement(
                achievement_id="collector",
                name="收藏家",
                description="背包中擁有 20 個不同的物品",
                icon="📦",
                condition_type="manual",
                condition_params={"type": "unique_items", "threshold": 20},
                reward_gold=150,
                reward_xp=80,
            ),
        ]
        for a in defaults:
            self.register(a)

    # ------------------------------------------------------------------
    # Statistics tracking
    # ------------------------------------------------------------------

    def track(self, stat_name: str, delta: int = 1) -> None:
        """Increment a tracked statistic."""
        self.statistics[stat_name] = self.statistics.get(stat_name, 0) + delta

    def get_stat(self, stat_name: str) -> int:
        return self.statistics.get(stat_name, 0)

    # ------------------------------------------------------------------
    # Progress update
    # ------------------------------------------------------------------

    def update_progress(self, achievement_id: str, value: float) -> None:
        """Manually update progress for an achievement."""
        prog = self.progress.get(achievement_id)
        if prog is None:
            return
        prog.current_value = max(prog.current_value, value)
        if prog.current_value >= prog.target_value:
            prog.is_complete = True
            self._unlock(achievement_id)

    def check_all(self) -> List[Achievement]:
        """Check all achievements against current statistics."""
        newly_unlocked = []

        for ach_id, ach in self.achievements.items():
            if ach.unlocked:
                continue

            params = ach.condition_params
            stat_name = params.get("type", "")
            threshold = params.get("threshold", 1)

            current = self.statistics.get(stat_name, 0)
            prog = self.progress.get(ach_id)
            if prog:
                prog.current_value = float(current)

            if current >= threshold:
                self._unlock(ach_id)
                newly_unlocked.append(ach)

        return newly_unlocked

    # ------------------------------------------------------------------
    # Manual unlock
    # ------------------------------------------------------------------

    def unlock(self, achievement_id: str) -> Optional[Achievement]:
        """Manually unlock an achievement."""
        return self._unlock(achievement_id)

    def _unlock(self, achievement_id: str) -> Optional[Achievement]:
        ach = self.achievements.get(achievement_id)
        if ach is None or ach.unlocked:
            return ach

        ach.unlocked = True
        ach.unlocked_at = datetime.now().isoformat()

        prog = self.progress.get(achievement_id)
        if prog:
            prog.is_complete = True

        return ach

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, achievement_id: str) -> Optional[Achievement]:
        return self.achievements.get(achievement_id)

    def get_progress(self, achievement_id: str) -> Optional[float]:
        prog = self.progress.get(achievement_id)
        return prog.percentage if prog else None

    def get_unlocked(self) -> List[Achievement]:
        return [a for a in self.achievements.values() if a.unlocked]

    def get_locked(self) -> List[Achievement]:
        return [a for a in self.achievements.values() if not a.unlocked]

    def get_total_unlocked(self) -> int:
        return sum(1 for a in self.achievements.values() if a.unlocked)

    def get_total(self) -> int:
        return len(self.achievements)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def format_summary(self) -> str:
        """Generate a text summary of achievements."""
        total = self.get_total()
        unlocked = self.get_total_unlocked()
        lines = [f"🏆 成就: {unlocked}/{total} 已解鎖"]

        # Recently unlocked (last 3)
        recent = [a for a in self.achievements.values() if a.unlocked][-3:]
        if recent:
            lines.append("  最近解鎖:")
            for a in recent:
                lines.append(f"    {a.icon} {a.name}")

        # Closest to unlocking
        close = sorted(
            [a for a in self.achievements.values() if not a.unlocked],
            key=lambda a: self.get_progress(a.achievement_id) or 0,
            reverse=True,
        )[:3]
        if close:
            lines.append("  即將解鎖:")
            for a in close:
                p = self.get_progress(a.achievement_id) or 0
                bar = '█' * int(p * 10) + '░' * (10 - int(p * 10))
                lines.append(f"    {a.icon} {a.name} [{bar}] {p*100:.0f}%")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "achievements": {aid: a.to_dict() for aid, a in self.achievements.items()},
            "progress": {aid: p.to_dict() for aid, p in self.progress.items()},
            "statistics": self.statistics.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AchievementManager":
        mgr = cls()
        mgr.statistics = data.get("statistics", {})
        for aid, adata in data.get("achievements", {}).items():
            mgr.achievements[aid] = Achievement.from_dict(adata)
        for aid, pdata in data.get("progress", {}).items():
            mgr.progress[aid] = AchievementProgress(**pdata)
        return mgr
