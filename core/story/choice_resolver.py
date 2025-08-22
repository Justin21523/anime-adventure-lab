# core/story/choice_resolver.py
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from .game_state import GameState, RelationType

logger = logging.getLogger(__name__)


@dataclass
class ChoiceCondition:
    """Condition for choice availability"""

    type: str  # "flag", "relationship", "item", "stat", "location"
    target: str
    operator: str  # "==", "!=", ">", "<", ">=", "<=", "has", "not_has"
    value: Any

    def evaluate(self, game_state: GameState) -> bool:
        """Evaluate if condition is met"""
        try:
            if self.type == "flag":
                actual_value = game_state.get_flag(self.target, None)
                return self._compare_values(actual_value, self.operator, self.value)

            elif self.type == "relationship":
                if self.target not in game_state.relationships:
                    return False
                rel = game_state.relationships[self.target]
                if "affinity" in self.operator:
                    actual_value = rel.affinity
                elif "trust" in self.operator:
                    actual_value = rel.trust
                else:
                    actual_value = rel.relation_type.value
                return self._compare_values(
                    actual_value, self.operator.split("_")[-1], self.value
                )

            elif self.type == "item":
                if self.operator == "has":
                    return self.target in game_state.inventory
                elif self.operator == "not_has":
                    return self.target not in game_state.inventory
                elif self.target in game_state.inventory:
                    actual_value = game_state.inventory[self.target].quantity
                    return self._compare_values(actual_value, self.operator, self.value)
                return False

            elif self.type == "stat":
                actual_value = getattr(game_state, f"player_{self.target}", 0)
                return self._compare_values(actual_value, self.operator, self.value)

            elif self.type == "location":
                if self.operator == "==":
                    return game_state.current_location == self.value
                elif self.operator == "!=":
                    return game_state.current_location != self.value
                elif self.operator == "discovered":
                    return (
                        self.target in game_state.locations
                        and game_state.locations[self.target].discovered
                    )

            return False

        except Exception as e:
            logger.error(f"Error evaluating condition {self}: {e}")
            return False

    def _compare_values(self, actual: Any, operator: str, expected: Any) -> bool:
        """Compare two values based on operator"""
        if operator == "==":
            return actual == expected
        elif operator == "!=":
            return actual != expected
        elif operator == ">" and isinstance(actual, (int, float)):
            return actual > expected
        elif operator == "<" and isinstance(actual, (int, float)):
            return actual < expected
        elif operator == ">=" and isinstance(actual, (int, float)):
            return actual >= expected
        elif operator == "<=" and isinstance(actual, (int, float)):
            return actual <= expected
        elif operator == "has":
            return expected in actual if hasattr(actual, "__contains__") else False
        elif operator == "not_has":
            return expected not in actual if hasattr(actual, "__contains__") else True
        return False


@dataclass
class ChoiceConsequence:
    """Consequence of making a choice"""

    type: str  # "flag", "relationship", "item", "stat", "location", "dialogue"
    target: str
    action: str  # "set", "add", "remove", "change"
    value: Any
    description: str = ""


@dataclass
class Choice:
    """A choice available to the player"""

    id: str
    text: str
    description: str = ""
    conditions: List[ChoiceCondition] = None  # type: ignore
    consequences: List[ChoiceConsequence] = None  # type: ignore
    priority: int = 0  # Higher priority choices appear first
    tags: List[str] = None  # type: ignore

    def is_available(self, game_state: GameState) -> bool:
        """Check if choice is available given current game state"""
        if not self.conditions:
            return True

        return all(condition.evaluate(game_state) for condition in self.conditions)

    def apply_consequences(self, game_state: GameState) -> Dict[str, Any]:
        """Apply choice consequences to game state"""
        changes = {
            "relationships": [],
            "items": [],
            "flags": [],
            "stats": [],
            "location": None,
        }

        if not self.consequences:
            return changes

        for consequence in self.consequences:
            try:
                if consequence.type == "flag":
                    game_state.set_flag(
                        consequence.target, consequence.value, consequence.description
                    )
                    changes["flags"].append(
                        {
                            "flag_id": consequence.target,
                            "value": consequence.value,
                            "description": consequence.description,
                        }
                    )

                elif consequence.type == "relationship":
                    if consequence.action == "change":
                        game_state.update_relationship(
                            character_id=consequence.target,
                            affinity_delta=consequence.value.get("affinity_delta", 0),
                            trust_delta=consequence.value.get("trust_delta", 0),
                            notes=consequence.description,
                        )
                        changes["relationships"].append(
                            {
                                "character": consequence.target,
                                "affinity_delta": consequence.value.get(
                                    "affinity_delta", 0
                                ),
                                "trust_delta": consequence.value.get("trust_delta", 0),
                            }
                        )

                elif consequence.type == "item":
                    if consequence.action == "add":
                        game_state.add_item(
                            item_id=consequence.target,
                            name=consequence.value.get("name", consequence.target),
                            quantity=consequence.value.get("quantity", 1),
                            description=consequence.description,
                        )
                        changes["items"].append(
                            {
                                "action": "add",
                                "item_id": consequence.target,
                                "quantity": consequence.value.get("quantity", 1),
                            }
                        )
                    elif consequence.action == "remove":
                        game_state.remove_item(consequence.target, consequence.value)
                        changes["items"].append(
                            {
                                "action": "remove",
                                "item_id": consequence.target,
                                "quantity": consequence.value,
                            }
                        )

                elif consequence.type == "stat":
                    current_value = getattr(
                        game_state, f"player_{consequence.target}", 0
                    )
                    if consequence.action == "set":
                        setattr(
                            game_state,
                            f"player_{consequence.target}",
                            consequence.value,
                        )
                    elif consequence.action == "add":
                        setattr(
                            game_state,
                            f"player_{consequence.target}",
                            current_value + consequence.value,
                        )
                    changes["stats"].append(
                        {
                            "stat": consequence.target,
                            "action": consequence.action,
                            "value": consequence.value,
                        }
                    )

                elif consequence.type == "location":
                    if consequence.action == "move":
                        game_state.move_to_location(consequence.target)
                        changes["location"] = consequence.target
                    elif consequence.action == "discover":
                        game_state.discover_location(
                            location_id=consequence.target,
                            name=consequence.value.get("name", consequence.target),
                            properties=consequence.value.get("properties", {}),
                        )

            except Exception as e:
                logger.error(f"Error applying consequence {consequence}: {e}")

        return changes


class ChoiceResolver:
    """Resolves player choices and their consequences"""

    def __init__(self):
        self.choice_templates = self._load_choice_templates()
        self.dynamic_choice_generators = {}

    def _load_choice_templates(self) -> Dict[str, Choice]:
        """Load predefined choice templates"""
        return {
            "explore": Choice(
                id="explore",
                text="探索周圍",
                description="仔細查看當前環境",
                consequences=[
                    ChoiceConsequence(
                        type="flag",
                        target="area_explored",
                        action="set",
                        value=True,
                        description="已探索當前區域",
                    )
                ],
            ),
            "rest": Choice(
                id="rest",
                text="休息",
                description="恢復體力和精神",
                conditions=[ChoiceCondition("stat", "energy", "<", 50)],
                consequences=[
                    ChoiceConsequence(
                        type="stat",
                        target="energy",
                        action="set",
                        value=100,
                        description="完全恢復精力",
                    )
                ],
            ),
            "continue": Choice(id="continue", text="繼續", description="繼續當前情節"),
        }

    def get_available_choices(
        self, game_state: GameState, context_choices: List[Dict[str, Any]] = None
    ) -> List[Choice]:
        """Get all available choices for current game state"""
        available_choices = []

        # Add context-specific choices from story generation
        if context_choices:
            for choice_data in context_choices:
                choice = self._create_choice_from_data(choice_data)
                if choice and choice.is_available(game_state):
                    available_choices.append(choice)

        # Add template choices that are available
        for template_choice in self.choice_templates.values():
            if template_choice.is_available(game_state):
                available_choices.append(template_choice)

        # Generate dynamic choices based on game state
        dynamic_choices = self._generate_dynamic_choices(game_state)
        available_choices.extend(dynamic_choices)

        # Sort by priority (higher first)
        available_choices.sort(key=lambda c: c.priority, reverse=True)

        return available_choices

    def _create_choice_from_data(self, choice_data: Dict[str, Any]) -> Optional[Choice]:
        """Create Choice object from story generation data"""
        try:
            conditions = []
            if "conditions" in choice_data:
                for cond_data in choice_data["conditions"]:
                    conditions.append(
                        ChoiceCondition(
                            type=cond_data["type"],
                            target=cond_data["target"],
                            operator=cond_data["operator"],
                            value=cond_data["value"],
                        )
                    )

            consequences = []
            if "consequences" in choice_data:
                for cons_data in choice_data["consequences"]:
                    consequences.append(
                        ChoiceConsequence(
                            type=cons_data["type"],
                            target=cons_data["target"],
                            action=cons_data["action"],
                            value=cons_data["value"],
                            description=cons_data.get("description", ""),
                        )
                    )

            return Choice(
                id=choice_data["id"],
                text=choice_data["text"],
                description=choice_data.get("description", ""),
                conditions=conditions,
                consequences=consequences,
                priority=choice_data.get("priority", 0),
                tags=choice_data.get("tags", []),
            )

        except Exception as e:
            logger.error(f"Error creating choice from data {choice_data}: {e}")
            return None

    def _generate_dynamic_choices(self, game_state: GameState) -> List[Choice]:
        """Generate dynamic choices based on current state"""
        dynamic_choices = []

        # Relationship-based choices
        for char_id, relationship in game_state.relationships.items():
            if relationship.affinity < 30:
                # Offer reconciliation choice
                dynamic_choices.append(
                    Choice(
                        id=f"reconcile_{char_id}",
                        text=f"與{char_id}和解",
                        description=f"嘗試修復與{char_id}的關係",
                        consequences=[
                            ChoiceConsequence(
                                type="relationship",
                                target=char_id,
                                action="change",
                                value={"affinity_delta": 10, "trust_delta": 5},
                                description="主動和解",
                            )
                        ],
                        priority=2,
                    )
                )

        # Inventory-based choices
        if "healing_potion" in game_state.inventory and game_state.player_health < 50:
            dynamic_choices.append(
                Choice(
                    id="use_healing_potion",
                    text="使用治療藥水",
                    description="恢復生命值",
                    consequences=[
                        ChoiceConsequence(
                            type="item",
                            target="healing_potion",
                            action="remove",
                            value=1,
                        ),
                        ChoiceConsequence(
                            type="stat", target="health", action="add", value=50
                        ),
                    ],
                    priority=3,
                )
            )

        # Location-based choices
        if game_state.current_location:
            current_loc = game_state.locations.get(game_state.current_location)
            if current_loc and len(current_loc.characters_present) > 0:
                for char_id in current_loc.characters_present:
                    if char_id not in [
                        d["speaker"]
                        for d in game_state.timeline[-1:3]
                        if d.event_type.value == "dialogue"
                    ]:
                        dynamic_choices.append(
                            Choice(
                                id=f"talk_to_{char_id}",
                                text=f"與{char_id}對話",
                                description=f"主動找{char_id}交談",
                                priority=1,
                            )
                        )

        return dynamic_choices

    def resolve_choice(self, choice_id: str, game_state: GameState) -> Dict[str, Any]:
        """Resolve a choice and apply its consequences"""
        # Find the choice
        choice = self.choice_templates.get(choice_id)

        if not choice:
            # Try to find in dynamic choices
            available_choices = self.get_available_choices(game_state)
            choice = next((c for c in available_choices if c.id == choice_id), None)

        if not choice:
            logger.warning(f"Choice {choice_id} not found")
            return {}

        if not choice.is_available(game_state):
            logger.warning(f"Choice {choice_id} not available")
            return {}

        # Apply consequences
        changes = choice.apply_consequences(game_state)

        logger.info(f"Resolved choice {choice_id} with changes: {changes}")
        return changes

    def register_dynamic_choice_generator(
        self, name: str, generator: Callable[[GameState], List[Choice]]
    ):
        """Register a dynamic choice generator function"""
        self.dynamic_choice_generators[name] = generator

    def get_choice_preview(
        self, choice_id: str, game_state: GameState
    ) -> Optional[Dict[str, Any]]:
        """Get a preview of what would happen if this choice is made"""
        choice = self.choice_templates.get(choice_id)
        if not choice or not choice.is_available(game_state):
            return None

        preview = {
            "id": choice.id,
            "text": choice.text,
            "description": choice.description,
            "available": choice.is_available(game_state),
            "consequences_preview": [],
        }

        if choice.consequences:
            for consequence in choice.consequences:
                preview["consequences_preview"].append(
                    {
                        "type": consequence.type,
                        "target": consequence.target,
                        "action": consequence.action,
                        "description": consequence.description,
                    }
                )

        return preview
