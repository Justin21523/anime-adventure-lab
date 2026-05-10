"""
Time System
Tracks game time (hours, days, seasons, years) and triggers time-based events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

    @property
    def icon(self) -> str:
        return {
            Season.SPRING: "🌸",
            Season.SUMMER: "☀️",
            Season.AUTUMN: "🍂",
            Season.WINTER: "❄️",
        }[self]

    @property
    def name_zh(self) -> str:
        return {
            Season.SPRING: "春",
            Season.SUMMER: "夏",
            Season.AUTUMN: "秋",
            Season.WINTER: "冬",
        }[self]


class TimeOfDay(Enum):
    MIDNIGHT = "midnight"
    EARLY_MORNING = "early_morning"
    MORNING = "morning"
    NOON = "noon"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"

    @property
    def icon(self) -> str:
        return {
            TimeOfDay.MIDNIGHT: "🌑",
            TimeOfDay.EARLY_MORNING: "🌅",
            TimeOfDay.MORNING: "🌤️",
            TimeOfDay.NOON: "☀️",
            TimeOfDay.AFTERNOON: "🌤️",
            TimeOfDay.EVENING: "🌇",
            TimeOfDay.NIGHT: "🌙",
        }[self]


@dataclass
class GameTime:
    """Current game clock."""
    turn: int = 0
    hour: int = 8             # 0-23
    day: int = 1
    season: Season = Season.SPRING
    year: int = 1

    @property
    def time_of_day(self) -> TimeOfDay:
        if self.hour == 0:
            return TimeOfDay.MIDNIGHT
        elif self.hour < 6:
            return TimeOfDay.EARLY_MORNING
        elif self.hour < 9:
            return TimeOfDay.MORNING
        elif self.hour < 12:
            return TimeOfDay.NOON
        elif self.hour < 17:
            return TimeOfDay.AFTERNOON
        elif self.hour < 20:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT

    def format_short(self) -> str:
        """Short format: '春 第3天 14:00'"""
        tod = self.time_of_day
        return f"{self.season.name_zh} 第{self.day}天 {tod.icon} {self.hour:02d}:00"

    def format_long(self) -> str:
        """Long format: '第一年 春天 第3天 下午2點'"""
        tod = self.time_of_day
        season_name = self.season.name_zh
        time_desc = {
            TimeOfDay.MIDNIGHT: "午夜",
            TimeOfDay.EARLY_MORNING: "清晨",
            TimeOfDay.MORNING: "上午",
            TimeOfDay.NOON: "中午",
            TimeOfDay.AFTERNOON: "下午",
            TimeOfDay.EVENING: "傍晚",
            TimeOfDay.NIGHT: "夜晚",
        }
        return (f"第{self.year}年 {season_name} 第{self.day}天 "
                f"{tod.icon} {time_desc[tod]}{self.hour % 12}點")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "hour": self.hour,
            "day": self.day,
            "season": self.season.value,
            "year": self.year,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameTime":
        return cls(
            turn=data.get("turn", 0),
            hour=data.get("hour", 8),
            day=data.get("day", 1),
            season=Season(data.get("season", "spring")),
            year=data.get("year", 1),
        )


@dataclass
class TimeEvent:
    """An event triggered by time conditions."""
    event_id: str
    description: str
    trigger_hour: Optional[int] = None      # Specific hour
    trigger_day: Optional[int] = None       # Specific day
    trigger_season: Optional[Season] = None # Specific season
    recurring: bool = False                 # Repeat each cycle
    triggered: bool = False

    def check(self, game_time: GameTime) -> bool:
        if self.triggered and not self.recurring:
            return False
        if self.trigger_hour is not None and game_time.hour != self.trigger_hour:
            return False
        if self.trigger_day is not None and game_time.day != self.trigger_day:
            return False
        if self.trigger_season is not None and game_time.season != self.trigger_season:
            return False
        return True

    def fire(self, recurring: bool = False) -> None:
        self.triggered = True
        if not recurring:
            return  # One-time, stays triggered


# ---------------------------------------------------------------------------
# Time System
# ---------------------------------------------------------------------------

class TimeSystem:
    """Manages game time and time-based events."""

    # Default: 1 turn = 1 hour (configurable)
    HOURS_PER_TURN = 1
    DAYS_PER_SEASON = 30
    TURNS_PER_DAY = 24  # 24 hours

    def __init__(self, hours_per_turn: int = 1):
        self.game_time = GameTime()
        self.HOURS_PER_TURN = hours_per_turn
        self.scheduled_events: List[TimeEvent] = []
        self.fired_events: List[TimeEvent] = []

    # ------------------------------------------------------------------
    # Advancing time
    # ------------------------------------------------------------------

    def advance(self, turns: int = 1) -> Dict[str, Any]:
        """Advance time by N turns. Returns time delta info."""
        hours_advanced = turns * self.HOURS_PER_TURN
        old_season = self.game_time.season
        old_day = self.game_time.day

        self.game_time.turn += turns
        self.game_time.hour += hours_advanced

        # Day rollover
        days_passed = self.game_time.hour // 24
        self.game_time.hour = self.game_time.hour % 24
        self.game_time.day += days_passed

        # Season rollover
        seasons_passed = (self.game_time.day - 1) // self.DAYS_PER_SEASON
        if seasons_passed > 0:
            self.game_time.day -= seasons_passed * self.DAYS_PER_SEASON
            season_idx = ((self.game_time.season.value in [s.value for s in Season]) and
                         list(Season).index(self.game_time.season) + seasons_passed) % 4
            self.game_time.season = list(Season)[season_idx]

        # Year rollover
        if self.game_time.season.value < old_season.value:  # Wrapped to spring
            self.game_time.year += 1

        return {
            "turns_advanced": turns,
            "hours_advanced": hours_advanced,
            "days_passed": days_passed,
            "season_changed": self.game_time.season != old_season,
            "new_season": self.game_time.season.name_zh if self.game_time.season != old_season else None,
        }

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def schedule_event(self, event: TimeEvent) -> None:
        """Schedule a time-based event."""
        self.scheduled_events.append(event)

    def check_events(self) -> List[TimeEvent]:
        """Check all scheduled events and fire matching ones."""
        fired = []
        for event in self.scheduled_events:
            if event.check(self.game_time):
                recurring = event.recurring
                event.fire(recurring=recurring)
                fired.append(event)
                self.fired_events.append(event)
        return fired

    def clear_event(self, event_id: str) -> bool:
        """Remove a scheduled event."""
        for i, event in enumerate(self.scheduled_events):
            if event.event_id == event_id:
                self.scheduled_events.pop(i)
                return True
        return False

    # ------------------------------------------------------------------
    # Seasonal effects
    # ------------------------------------------------------------------

    def get_seasonal_effects(self) -> Dict[str, Any]:
        """Return modifiers based on current season."""
        season = self.game_time.season
        effects = {
            "season": season.name_zh,
            "icon": season.icon,
        }

        if season == Season.SPRING:
            effects["description"] = "春暖花開，探索效率提升"
            effects["modifiers"] = {"survival.stamina_regen": 1.2, "exploration_bonus": 0.1}
        elif season == Season.SUMMER:
            effects["description"] = "炎炎夏日，體力消耗加快"
            effects["modifiers"] = {"survival.hunger_rate": 1.3, "combat.speed": -1}
        elif season == Season.AUTUMN:
            effects["description"] = "秋高氣爽，適合狩獵和採集"
            effects["modifiers"] = {"gathering_bonus": 0.2, "combat.attack": 1}
        elif season == Season.WINTER:
            effects["description"] = "寒冬嚴峻，生存難度提高"
            effects["modifiers"] = {"survival.hunger_rate": 1.5, "exploration_penalty": 0.2}

        return effects

    # ------------------------------------------------------------------
    # Time-of-day effects
    # ------------------------------------------------------------------

    def get_time_of_day_effects(self) -> Dict[str, Any]:
        """Return modifiers based on current time of day."""
        tod = self.game_time.time_of_day
        effects = {
            "time_of_day": tod.value,
            "icon": tod.icon,
        }

        if tod in (TimeOfDay.NIGHT, TimeOfDay.MIDNIGHT):
            effects["description"] = "夜色降臨，視野受限但適合潛行"
            effects["modifiers"] = {"survival.stealth": 3, "combat.hit_rate": -0.1}
        elif tod == TimeOfDay.EVENING:
            effects["description"] = "黃昏時分，危險開始增加"
            effects["modifiers"] = {"survival.danger_level": 5}
        elif tod == TimeOfDay.NOON:
            effects["description"] = "正午時分，視野最佳"
            effects["modifiers"] = {"survival.perception": 2}

        return effects

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_description(self) -> str:
        """Get a narrative time description."""
        return self.game_time.format_long()

    def is_night(self) -> bool:
        return self.game_time.time_of_day in (TimeOfDay.NIGHT, TimeOfDay.MIDNIGHT)

    def is_dawn(self) -> bool:
        return self.game_time.time_of_day == TimeOfDay.EARLY_MORNING

    def is_daytime(self) -> bool:
        return self.game_time.time_of_day in (
            TimeOfDay.MORNING, TimeOfDay.NOON, TimeOfDay.AFTERNOON
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_time": self.game_time.to_dict(),
            "hours_per_turn": self.HOURS_PER_TURN,
            "scheduled_events": [
                {
                    "event_id": e.event_id,
                    "description": e.description,
                    "trigger_hour": e.trigger_hour,
                    "trigger_day": e.trigger_day,
                    "trigger_season": e.trigger_season.value if e.trigger_season else None,
                    "recurring": e.recurring,
                    "triggered": e.triggered,
                }
                for e in self.scheduled_events
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSystem":
        ts = cls(hours_per_turn=data.get("hours_per_turn", 1))
        ts.game_time = GameTime.from_dict(data["game_time"])
        for edata in data.get("scheduled_events", []):
            event = TimeEvent(
                event_id=edata["event_id"],
                description=edata["description"],
                trigger_hour=edata.get("trigger_hour"),
                trigger_day=edata.get("trigger_day"),
                trigger_season=Season(edata["trigger_season"]) if edata.get("trigger_season") else None,
                recurring=edata.get("recurring", False),
                triggered=edata.get("triggered", False),
            )
            ts.scheduled_events.append(event)
        return ts
