# core/story/game_state.py
import json
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
from enum import Enum
import json
import uuid


@dataclass
class PlayerStats:
    """Player character statistics"""

    health: int = 100
    energy: int = 100
    intelligence: int = 10
    charisma: int = 10
    luck: int = 10
    experience: int = 0
    level: int = 1

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "PlayerStats":
        return cls(**data)


@dataclass
class GameState:
    """Current game state snapshot"""

    scene_id: str
    scene_description: str
    available_choices: List[Dict[str, Any]]
    story_context: Dict[str, Any]
    flags: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameState":
        return cls(**data)


@dataclass
class GameSession:
    """Complete game session data"""

    session_id: str
    player_name: str
    persona_id: str
    created_at: datetime
    updated_at: datetime
    turn_count: int
    stats: PlayerStats
    inventory: List[str]
    current_state: GameState
    history: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameSession":
        # Convert ISO strings back to datetime
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["stats"] = PlayerStats.from_dict(data["stats"])
        data["current_state"] = GameState.from_dict(data["current_state"])
        return cls(**data)

    def add_to_history(
        self, player_input: str, ai_response: str, choice_id: Optional[str] = None
    ):
        """Add turn to history"""
        self.history.append(
            {
                "turn": self.turn_count,
                "timestamp": datetime.now().isoformat(),
                "player_input": player_input,
                "ai_response": ai_response,
                "choice_id": choice_id,
                "scene_id": self.current_state.scene_id,
            }
        )
        self.turn_count += 1
        self.updated_at = datetime.now()
