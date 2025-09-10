# =============================================================================
# core/story/choices.py
"""
Choice Management System
Handles player choices, branching narratives, and consequence tracking
"""

import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ChoiceType(Enum):
    """Types of game choices"""

    DIALOGUE = "dialogue"
    ACTION = "action"
    EXPLORATION = "exploration"
    COMBAT = "combat"
    PUZZLE = "puzzle"
    MORAL = "moral"


class ChoiceDifficulty(Enum):
    """Choice difficulty levels"""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


@dataclass
class GameChoice:
    """Represents a single player choice option"""

    choice_id: str
    text: str
    choice_type: ChoiceType
    difficulty: ChoiceDifficulty
    requirements: Dict[str, Any]  # stat requirements, items needed, etc.
    consequences: Dict[str, Any]  # stat changes, story flags, etc.
    success_chance: float = 1.0  # 0.0 to 1.0
    description: Optional[str] = None

    def can_choose(
        self, player_stats: Dict[str, int], inventory: List[str], flags: Dict[str, bool]
    ) -> Tuple[bool, str]:
        """Check if player can make this choice"""
        # Check stat requirements
        for stat, min_value in self.requirements.get("stats", {}).items():
            if player_stats.get(stat, 0) < min_value:
                return False, f"需要{stat} >= {min_value}"

        # Check item requirements
        required_items = self.requirements.get("items", [])
        for item in required_items:
            if item not in inventory:
                return False, f"需要物品：{item}"

        # Check flag requirements
        for flag, required_value in self.requirements.get("flags", {}).items():
            if flags.get(flag, False) != required_value:
                return False, f"不符合條件：{flag}"

        return True, ""

    def execute(
        self, player_stats: Dict[str, int], luck: int = 10
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute choice and return success status and consequences"""
        # Calculate success based on chance and luck
        luck_bonus = (luck - 10) * 0.01  # Each point above 10 adds 1% success chance
        final_chance = min(1.0, self.success_chance + luck_bonus)

        success = random.random() < final_chance
        consequences = self.consequences.copy()

        # Apply failure penalties if unsuccessful
        if not success and "failure_penalty" in self.consequences:
            consequences.update(self.consequences["failure_penalty"])

        return success, consequences


class ChoiceManager:
    """Manages choice generation and validation"""

    def __init__(self):
        self.choice_templates = self._load_choice_templates()

    def _load_choice_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load choice templates for different scenarios"""
        return {
            "exploration": {
                "investigate_sound": {
                    "text": "調查奇怪的聲音",
                    "type": ChoiceType.EXPLORATION,
                    "difficulty": ChoiceDifficulty.MEDIUM,
                    "success_chance": 0.7,
                    "requirements": {"stats": {"intelligence": 5}},
                    "consequences": {
                        "stats": {"experience": 10},
                        "flags": {"investigated_sound": True},
                    },
                },
                "avoid_danger": {
                    "text": "小心避開危險",
                    "type": ChoiceType.ACTION,
                    "difficulty": ChoiceDifficulty.EASY,
                    "success_chance": 0.9,
                    "requirements": {},
                    "consequences": {"stats": {"health": -5}},
                },
            },
            "dialogue": {
                "persuade": {
                    "text": "嘗試說服對方",
                    "type": ChoiceType.DIALOGUE,
                    "difficulty": ChoiceDifficulty.MEDIUM,
                    "success_chance": 0.6,
                    "requirements": {"stats": {"charisma": 8}},
                    "consequences": {"flags": {"npc_persuaded": True}},
                },
                "intimidate": {
                    "text": "威脅對方",
                    "type": ChoiceType.DIALOGUE,
                    "difficulty": ChoiceDifficulty.HARD,
                    "success_chance": 0.4,
                    "requirements": {"stats": {"charisma": 12}},
                    "consequences": {"flags": {"npc_intimidated": True}},
                },
            },
        }

    def generate_choices(
        self,
        context: Dict[str, Any],
        player_stats: Dict[str, int],
        inventory: List[str],
        flags: Dict[str, bool],
    ) -> List[GameChoice]:
        """Generate appropriate choices for current context"""
        choices = []
        scene_type = context.get("scene_type", "exploration")

        # Get relevant choice templates
        templates = self.choice_templates.get(scene_type, {})

        for choice_id, template in templates.items():
            choice = GameChoice(
                choice_id=choice_id,
                text=template["text"],
                choice_type=template["type"],
                difficulty=template["difficulty"],
                success_chance=template["success_chance"],
                requirements=template["requirements"],
                consequences=template["consequences"],
            )

            # Check if choice is available to player
            can_choose, reason = choice.can_choose(player_stats, inventory, flags)
            if can_choose or context.get("show_all_choices", False):
                if not can_choose:
                    choice.text += f" ({reason})"
                choices.append(choice)

        return choices

    def validate_choice(
        self, choice_id: str, available_choices: List[GameChoice]
    ) -> Optional[GameChoice]:
        """Validate that a choice is available"""
        for choice in available_choices:
            if choice.choice_id == choice_id:
                return choice
        return None
