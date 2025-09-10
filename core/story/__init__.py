# =============================================================================
# Phase 2a: Story Engine 核心實作
# 人格驅動的文字冒險遊戲系統
# =============================================================================

# core/story/__init__.py
"""
Text Adventure Game Engine
Story-driven interactive game system with personality and state management
"""

from .engine import StoryEngine, get_story_engine
from .game_state import GameState, GameSession, PlayerStats
from .persona import PersonaManager, GamePersona
from .narrative import NarrativeGenerator, StoryContext
from .choices import ChoiceManager, GameChoice

__all__ = [
    "StoryEngine",
    "get_story_engine",
    "GameState",
    "GameSession",
    "PlayerStats",
    "PersonaManager",
    "GamePersona",
    "NarrativeGenerator",
    "StoryContext",
    "ChoiceManager",
    "GameChoice",
]
