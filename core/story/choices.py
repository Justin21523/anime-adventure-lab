# =============================================================================
# core/story/choices.py
"""
Choice Management System
Handles player choices, branching narratives, and consequence tracking
"""

import logging
import random
from typing import Dict, List, Optional, Any, Tuple, Callable
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
    STRATEGIC = "strategic"
    SOCIAL = "social"


class ChoiceDifficulty(Enum):
    """Choice difficulty levels"""

    TRIVIAL = "trivial"
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
    hidden_until_unlocked: bool = False

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

    def get_display_text(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Get display text with possible context modifications"""
        if not context:
            return self.text

        # Simple template substitution
        display_text = self.text
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in display_text:
                display_text = display_text.replace(placeholder, str(value))

        return display_text


class ChoiceManager:
    """Enhanced choice manager with context awareness"""

    def __init__(self):
        self.choice_templates = self._load_choice_templates()
        self.dynamic_choice_generators = self._setup_dynamic_generators()

    def _load_choice_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load choice templates for different scenarios"""
        return {
            "exploration": {
                "examine_area": {
                    "text": "仔細檢查這個區域",
                    "type": ChoiceType.EXPLORATION,
                    "difficulty": ChoiceDifficulty.EASY,
                    "success_chance": 0.9,
                    "requirements": {},
                    "consequences": {"stats": {"intelligence": 1}},
                },
                "search_for_secrets": {
                    "text": "尋找隱藏的秘密",
                    "type": ChoiceType.EXPLORATION,
                    "difficulty": ChoiceDifficulty.MEDIUM,
                    "success_chance": 0.6,
                    "requirements": {"stats": {"intelligence": 8}},
                    "consequences": {
                        "stats": {"intelligence": 2},
                        "flags": {"searched_area": True},
                    },
                },
                "investigate_sounds": {
                    "text": "調查奇怪的聲音",
                    "type": ChoiceType.EXPLORATION,
                    "difficulty": ChoiceDifficulty.MEDIUM,
                    "success_chance": 0.7,
                    "requirements": {},
                    "consequences": {"discovery": True},
                },
            },
            "dialogue": {
                "friendly_approach": {
                    "text": "友善地接近並打招呼",
                    "type": ChoiceType.DIALOGUE,
                    "difficulty": ChoiceDifficulty.EASY,
                    "success_chance": 0.8,
                    "requirements": {},
                    "consequences": {"stats": {"charisma": 1}, "relationship": 1},
                },
                "intimidate": {
                    "text": "試圖威嚇對方",
                    "type": ChoiceType.DIALOGUE,
                    "difficulty": ChoiceDifficulty.HARD,
                    "success_chance": 0.4,
                    "requirements": {"stats": {"charisma": 12}},
                    "consequences": {
                        "relationship": -2,
                        "flags": {"npc_intimidated": True},
                    },
                },
                "ask_information": {
                    "text": "禮貌地詢問資訊",
                    "type": ChoiceType.DIALOGUE,
                    "difficulty": ChoiceDifficulty.MEDIUM,
                    "success_chance": 0.75,
                    "requirements": {"stats": {"charisma": 6}},
                    "consequences": {
                        "stats": {"intelligence": 1},
                        "information_gained": True,
                    },
                },
                "offer_help": {
                    "text": "主動提供幫助",
                    "type": ChoiceType.DIALOGUE,
                    "difficulty": ChoiceDifficulty.MEDIUM,
                    "success_chance": 0.7,
                    "requirements": {},
                    "consequences": {"stats": {"charisma": 2}, "relationship": 3},
                },
            },
            "action": {
                "careful_approach": {
                    "text": "小心謹慎地行動",
                    "type": ChoiceType.ACTION,
                    "difficulty": ChoiceDifficulty.MEDIUM,
                    "success_chance": 0.8,
                    "requirements": {},
                    "consequences": {"stats": {"intelligence": 1}},
                },
                "bold_action": {
                    "text": "大膽果斷地行動",
                    "type": ChoiceType.ACTION,
                    "difficulty": ChoiceDifficulty.HARD,
                    "success_chance": 0.6,
                    "requirements": {"stats": {"charisma": 10}},
                    "consequences": {
                        "stats": {"charisma": 2},
                        "dramatic_outcome": True,
                    },
                },
                "wait_and_observe": {
                    "text": "等待並觀察情況",
                    "type": ChoiceType.ACTION,
                    "difficulty": ChoiceDifficulty.EASY,
                    "success_chance": 1.0,
                    "requirements": {},
                    "consequences": {"stats": {"intelligence": 1}, "time_passed": True},
                },
                "retreat_safely": {
                    "text": "安全地撤退",
                    "type": ChoiceType.ACTION,
                    "difficulty": ChoiceDifficulty.EASY,
                    "success_chance": 0.95,
                    "requirements": {},
                    "consequences": {"safe_retreat": True},
                },
            },
            "combat": {
                "aggressive_attack": {
                    "text": "發動猛烈攻擊",
                    "type": ChoiceType.COMBAT,
                    "difficulty": ChoiceDifficulty.HARD,
                    "success_chance": 0.5,
                    "requirements": {"stats": {"health": 50}},
                    "consequences": {"damage_dealt": "high", "stats": {"health": -10}},
                },
                "defensive_stance": {
                    "text": "採取防禦姿態",
                    "type": ChoiceType.COMBAT,
                    "difficulty": ChoiceDifficulty.MEDIUM,
                    "success_chance": 0.8,
                    "requirements": {},
                    "consequences": {"damage_reduced": True, "stats": {"health": 5}},
                },
                "strategic_maneuver": {
                    "text": "使用戰術機動",
                    "type": ChoiceType.COMBAT,
                    "difficulty": ChoiceDifficulty.MEDIUM,
                    "success_chance": 0.7,
                    "requirements": {"stats": {"intelligence": 8}},
                    "consequences": {
                        "tactical_advantage": True,
                        "stats": {"intelligence": 1},
                    },
                },
            },
            "puzzle": {
                "logical_analysis": {
                    "text": "進行邏輯分析",
                    "type": ChoiceType.PUZZLE,
                    "difficulty": ChoiceDifficulty.MEDIUM,
                    "success_chance": 0.6,
                    "requirements": {"stats": {"intelligence": 10}},
                    "consequences": {
                        "stats": {"intelligence": 2},
                        "puzzle_progress": True,
                    },
                },
                "trial_and_error": {
                    "text": "嘗試不同的方法",
                    "type": ChoiceType.PUZZLE,
                    "difficulty": ChoiceDifficulty.EASY,
                    "success_chance": 0.4,
                    "requirements": {},
                    "consequences": {"attempts": 1, "patience_required": True},
                },
                "seek_hints": {
                    "text": "尋找線索和提示",
                    "type": ChoiceType.PUZZLE,
                    "difficulty": ChoiceDifficulty.MEDIUM,
                    "success_chance": 0.8,
                    "requirements": {},
                    "consequences": {"hints_found": True, "stats": {"intelligence": 1}},
                },
            },
        }

    def _setup_dynamic_generators(self) -> Dict[str, Callable]:
        """Setup dynamic choice generators based on context"""
        return {
            "character_specific": self._generate_character_specific_choices,
            "location_specific": self._generate_location_specific_choices,
            "story_specific": self._generate_story_specific_choices,
            "relationship_based": self._generate_relationship_based_choices,
        }

    def generate_choices(
        self,
        context: Dict[str, Any],
        player_stats: Dict[str, int],
        inventory: List[str],
        flags: Dict[str, bool],
        max_choices: int = 4,
    ) -> List[GameChoice]:
        """Generate context-appropriate choices"""

        choices = []
        scene_type = context.get("scene_type", "exploration")

        # Get base choices from templates
        base_choices = self._get_base_choices_for_scene(scene_type)
        choices.extend(base_choices)

        # Add dynamic choices based on context
        dynamic_choices = self._generate_dynamic_choices(
            context, player_stats, inventory, flags
        )
        choices.extend(dynamic_choices)

        # Filter choices based on requirements
        valid_choices = []
        for choice in choices:
            can_choose, reason = choice.can_choose(player_stats, inventory, flags)
            if can_choose:
                valid_choices.append(choice)
            elif (
                len(valid_choices) < 2
            ):  # Ensure minimum choices even if requirements not met
                # Modify choice text to indicate requirement
                modified_choice = GameChoice(
                    choice_id=f"{choice.choice_id}_disabled",
                    text=f"{choice.text} ({reason})",
                    choice_type=choice.choice_type,
                    difficulty=choice.difficulty,
                    requirements=choice.requirements,
                    consequences={},
                    success_chance=0.0,
                    description=f"需要滿足條件: {reason}",
                )
                valid_choices.append(modified_choice)

        # Ensure we have at least some choices
        if not valid_choices:
            valid_choices = self._get_fallback_choices()

        # Sort by appropriateness and limit count
        sorted_choices = self._sort_choices_by_context(valid_choices, context)
        return sorted_choices[:max_choices]

    def _get_base_choices_for_scene(self, scene_type: str) -> List[GameChoice]:
        """Get base choices appropriate for scene type"""
        choices = []

        # Always include some exploration choices
        exploration_templates = self.choice_templates.get("exploration", {})
        for choice_id, template in list(exploration_templates.items())[:2]:
            choices.append(self._create_choice_from_template(choice_id, template))

        # Add scene-specific choices
        if scene_type in self.choice_templates:
            scene_templates = self.choice_templates[scene_type]
            for choice_id, template in list(scene_templates.items())[:2]:
                choices.append(self._create_choice_from_template(choice_id, template))

        # Add action choices
        action_templates = self.choice_templates.get("action", {})
        for choice_id, template in list(action_templates.items())[:1]:
            choices.append(self._create_choice_from_template(choice_id, template))

        return choices

    def _create_choice_from_template(
        self, choice_id: str, template: Dict[str, Any]
    ) -> GameChoice:
        """Create GameChoice from template data"""
        return GameChoice(
            choice_id=choice_id,
            text=template["text"],
            choice_type=template["type"],
            difficulty=template["difficulty"],
            requirements=template.get("requirements", {}),
            consequences=template.get("consequences", {}),
            success_chance=template.get("success_chance", 1.0),
            description=template.get("description"),
        )

    def _generate_dynamic_choices(
        self,
        context: Dict[str, Any],
        player_stats: Dict[str, int],
        inventory: List[str],
        flags: Dict[str, bool],
    ) -> List[GameChoice]:
        """Generate dynamic choices based on current context"""

        dynamic_choices = []

        # Character-specific choices
        if "active_characters" in context:
            char_choices = self._generate_character_specific_choices(
                context, player_stats
            )
            dynamic_choices.extend(char_choices)

        # Location-specific choices
        if "current_location" in context:
            location_choices = self._generate_location_specific_choices(
                context, inventory
            )
            dynamic_choices.extend(location_choices)

        # Story-specific choices
        if "plot_points" in context:
            story_choices = self._generate_story_specific_choices(context, flags)
            dynamic_choices.extend(story_choices)

        # Inventory-based choices
        if inventory:
            item_choices = self._generate_inventory_based_choices(inventory, context)
            dynamic_choices.extend(item_choices)

        return dynamic_choices

    def _generate_character_specific_choices(
        self, context: Dict[str, Any], player_stats: Dict[str, int]
    ) -> List[GameChoice]:
        """Generate choices specific to present characters"""

        choices = []
        active_characters = context.get("active_characters", [])

        for character in active_characters:
            if character == "player":  # Skip player character
                continue

            # Generate dialogue options for each NPC
            choices.append(
                GameChoice(
                    choice_id=f"talk_to_{character.lower().replace(' ', '_')}",
                    text=f"與{character}交談",
                    choice_type=ChoiceType.DIALOGUE,
                    difficulty=ChoiceDifficulty.EASY,
                    requirements={},
                    consequences={
                        "stats": {"charisma": 1},
                        "character_interaction": character,
                    },
                    success_chance=0.9,
                    description=f"主動與{character}開始對話",
                )
            )

            # Generate help/assist options if charisma is high enough
            if player_stats.get("charisma", 0) >= 8:
                choices.append(
                    GameChoice(
                        choice_id=f"help_{character.lower().replace(' ', '_')}",
                        text=f"主動幫助{character}",
                        choice_type=ChoiceType.ACTION,
                        difficulty=ChoiceDifficulty.MEDIUM,
                        requirements={"stats": {"charisma": 8}},
                        consequences={"stats": {"charisma": 2}, "relationship": 2},
                        success_chance=0.8,
                        description=f"提供協助給{character}",
                    )
                )

        return choices

    def _generate_location_specific_choices(
        self, context: Dict[str, Any], inventory: List[str]
    ) -> List[GameChoice]:
        """Generate choices specific to current location"""

        choices = []
        location = context.get("current_location", "").lower()

        # Forest/outdoor locations
        if any(keyword in location for keyword in ["森林", "樹林", "野外", "山"]):
            choices.extend(
                [
                    GameChoice(
                        choice_id="gather_resources",
                        text="收集天然資源",
                        choice_type=ChoiceType.EXPLORATION,
                        difficulty=ChoiceDifficulty.EASY,
                        requirements={},
                        consequences={
                            "add_items": ["樹枝", "草藥"],
                            "stats": {"intelligence": 1},
                        },
                        success_chance=0.8,
                        description="在自然環境中尋找有用的材料",
                    ),
                    GameChoice(
                        choice_id="climb_tree",
                        text="爬上高樹觀察",
                        choice_type=ChoiceType.EXPLORATION,
                        difficulty=ChoiceDifficulty.MEDIUM,
                        requirements={"stats": {"health": 70}},
                        consequences={
                            "stats": {"intelligence": 2},
                            "view_expanded": True,
                        },
                        success_chance=0.7,
                        description="攀爬到制高點獲得更好的視野",
                    ),
                ]
            )

        # Cave/underground locations
        elif any(keyword in location for keyword in ["洞穴", "地下", "隧道", "洞窟"]):
            choices.extend(
                [
                    GameChoice(
                        choice_id="examine_walls",
                        text="檢查洞壁上的痕跡",
                        choice_type=ChoiceType.EXPLORATION,
                        difficulty=ChoiceDifficulty.MEDIUM,
                        requirements={},
                        consequences={
                            "stats": {"intelligence": 1},
                            "clues_found": True,
                        },
                        success_chance=0.6,
                        description="仔細觀察洞穴中可能的線索",
                    ),
                    GameChoice(
                        choice_id="light_torch",
                        text="點燃火把照明",
                        choice_type=ChoiceType.ACTION,
                        difficulty=ChoiceDifficulty.EASY,
                        requirements={"items": ["火把"]},
                        consequences={
                            "visibility_improved": True,
                            "remove_items": ["火把"],
                        },
                        success_chance=0.95,
                        description="使用火把提高能見度",
                    ),
                ]
            )

        # Town/village locations
        elif any(keyword in location for keyword in ["城鎮", "村莊", "市集", "街道"]):
            choices.extend(
                [
                    GameChoice(
                        choice_id="ask_locals",
                        text="向當地居民打探消息",
                        choice_type=ChoiceType.DIALOGUE,
                        difficulty=ChoiceDifficulty.MEDIUM,
                        requirements={"stats": {"charisma": 6}},
                        consequences={"stats": {"intelligence": 2}, "local_info": True},
                        success_chance=0.75,
                        description="與當地人交流獲取資訊",
                    ),
                    GameChoice(
                        choice_id="visit_merchant",
                        text="拜訪商人進行交易",
                        choice_type=ChoiceType.ACTION,
                        difficulty=ChoiceDifficulty.EASY,
                        requirements={},
                        consequences={"trading_available": True},
                        success_chance=0.9,
                        description="尋找商人進行買賣",
                    ),
                ]
            )

        return choices

    def _generate_story_specific_choices(
        self, context: Dict[str, Any], flags: Dict[str, bool]
    ) -> List[GameChoice]:
        """Generate choices based on story progression"""

        choices = []
        plot_points = context.get("plot_points", [])

        # Check for specific story flags and generate relevant choices
        if "mystery_discovered" in flags and flags["mystery_discovered"]:
            choices.append(
                GameChoice(
                    choice_id="investigate_mystery",
                    text="深入調查發現的謎團",
                    choice_type=ChoiceType.PUZZLE,
                    difficulty=ChoiceDifficulty.HARD,
                    requirements={"stats": {"intelligence": 12}},
                    consequences={
                        "stats": {"intelligence": 3},
                        "mystery_progress": True,
                    },
                    success_chance=0.6,
                    description="運用智慧解開謎團",
                )
            )

        if "enemy_spotted" in flags and flags["enemy_spotted"]:
            choices.extend(
                [
                    GameChoice(
                        choice_id="prepare_ambush",
                        text="準備伏擊敵人",
                        choice_type=ChoiceType.STRATEGIC,
                        difficulty=ChoiceDifficulty.HARD,
                        requirements={"stats": {"intelligence": 10}},
                        consequences={
                            "ambush_prepared": True,
                            "stats": {"intelligence": 2},
                        },
                        success_chance=0.7,
                        description="制定戰術計劃",
                    ),
                    GameChoice(
                        choice_id="avoid_enemy",
                        text="避開敵人繼續前進",
                        choice_type=ChoiceType.ACTION,
                        difficulty=ChoiceDifficulty.MEDIUM,
                        requirements={},
                        consequences={"enemy_avoided": True, "stealth_success": True},
                        success_chance=0.8,
                        description="悄悄繞過危險",
                    ),
                ]
            )

        if "ally_in_danger" in flags and flags["ally_in_danger"]:
            choices.append(
                GameChoice(
                    choice_id="rescue_ally",
                    text="冒險救援盟友",
                    choice_type=ChoiceType.ACTION,
                    difficulty=ChoiceDifficulty.EXTREME,
                    requirements={"stats": {"health": 60, "charisma": 8}},
                    consequences={
                        "ally_saved": True,
                        "stats": {"health": -20, "charisma": 3},
                    },
                    success_chance=0.5,
                    description="不顧危險拯救同伴",
                )
            )

        return choices

    def _generate_inventory_based_choices(
        self, inventory: List[str], context: Dict[str, Any]
    ) -> List[GameChoice]:
        """Generate choices based on available items"""

        choices = []

        # Healing items
        if "治療藥水" in inventory:
            choices.append(
                GameChoice(
                    choice_id="use_healing_potion",
                    text="使用治療藥水",
                    choice_type=ChoiceType.ACTION,
                    difficulty=ChoiceDifficulty.TRIVIAL,
                    requirements={"items": ["治療藥水"]},
                    consequences={
                        "stats": {"health": 30},
                        "remove_items": ["治療藥水"],
                    },
                    success_chance=1.0,
                    description="恢復生命值",
                )
            )

        # Tools and equipment
        if "繩索" in inventory:
            choices.append(
                GameChoice(
                    choice_id="use_rope",
                    text="使用繩索",
                    choice_type=ChoiceType.ACTION,
                    difficulty=ChoiceDifficulty.MEDIUM,
                    requirements={"items": ["繩索"]},
                    consequences={"climbing_enabled": True, "remove_items": ["繩索"]},
                    success_chance=0.9,
                    description="用繩索解決高度或距離問題",
                )
            )

        if "鑰匙" in inventory:
            choices.append(
                GameChoice(
                    choice_id="use_key",
                    text="使用鑰匙開鎖",
                    choice_type=ChoiceType.ACTION,
                    difficulty=ChoiceDifficulty.EASY,
                    requirements={"items": ["鑰匙"]},
                    consequences={"door_unlocked": True, "remove_items": ["鑰匙"]},
                    success_chance=0.95,
                    description="用鑰匙打開鎖住的門或容器",
                )
            )

        # Magic items
        if "魔法卷軸" in inventory:
            choices.append(
                GameChoice(
                    choice_id="cast_spell",
                    text="施展魔法卷軸",
                    choice_type=ChoiceType.ACTION,
                    difficulty=ChoiceDifficulty.HARD,
                    requirements={"items": ["魔法卷軸"], "stats": {"intelligence": 15}},
                    consequences={
                        "magic_effect": True,
                        "remove_items": ["魔法卷軸"],
                        "stats": {"intelligence": 1},
                    },
                    success_chance=0.8,
                    description="使用魔法產生特殊效果",
                )
            )

        return choices

    def _get_fallback_choices(self) -> List[GameChoice]:
        """Get basic fallback choices when no other choices are available"""
        return [
            GameChoice(
                choice_id="observe_surroundings",
                text="觀察周圍環境",
                choice_type=ChoiceType.EXPLORATION,
                difficulty=ChoiceDifficulty.TRIVIAL,
                requirements={},
                consequences={"stats": {"intelligence": 1}},
                success_chance=1.0,
                description="仔細觀察當前環境",
            ),
            GameChoice(
                choice_id="think_carefully",
                text="仔細思考下一步",
                choice_type=ChoiceType.ACTION,
                difficulty=ChoiceDifficulty.TRIVIAL,
                requirements={},
                consequences={"stats": {"intelligence": 1}, "time_passed": True},
                success_chance=1.0,
                description="花時間制定計劃",
            ),
            GameChoice(
                choice_id="continue_journey",
                text="繼續前進",
                choice_type=ChoiceType.ACTION,
                difficulty=ChoiceDifficulty.EASY,
                requirements={},
                consequences={"progress": True},
                success_chance=0.9,
                description="朝目標繼續前進",
            ),
        ]

    def _sort_choices_by_context(
        self, choices: List[GameChoice], context: Dict[str, Any]
    ) -> List[GameChoice]:
        """Sort choices by relevance to current context"""

        def choice_relevance_score(choice: GameChoice) -> int:
            score = 0

            # Prioritize choices that match scene type
            scene_type = context.get("scene_type", "").lower()
            choice_type = choice.choice_type.value.lower()

            if scene_type == choice_type:
                score += 10
            elif scene_type == "dialogue" and choice_type in ["dialogue", "action"]:
                score += 5
            elif scene_type == "exploration" and choice_type in [
                "exploration",
                "puzzle",
            ]:
                score += 5

            # Prioritize based on difficulty matching player level
            difficulty_preference = {
                ChoiceDifficulty.TRIVIAL: 1,
                ChoiceDifficulty.EASY: 3,
                ChoiceDifficulty.MEDIUM: 5,
                ChoiceDifficulty.HARD: 3,
                ChoiceDifficulty.EXTREME: 1,
            }
            score += difficulty_preference.get(choice.difficulty, 0)

            # Prioritize choices with higher success chance
            score += int(choice.success_chance * 5)

            # Bonus for choices with meaningful consequences
            if choice.consequences:
                score += 2
                if "stats" in choice.consequences:
                    score += 1
                if any(
                    key in choice.consequences
                    for key in ["relationship", "story_progress", "discovery"]
                ):
                    score += 3

            return score

        return sorted(choices, key=choice_relevance_score, reverse=True)

    def validate_choice(
        self, choice_id: str, available_choices: List[GameChoice]
    ) -> Optional[GameChoice]:
        """Validate that a choice is available"""
        for choice in available_choices:
            if choice.choice_id == choice_id:
                return choice
        return None

    def get_choice_consequences_preview(
        self, choice: GameChoice, player_stats: Dict[str, int]
    ) -> Dict[str, Any]:
        """Get a preview of what consequences a choice might have"""

        # Estimate success chance with current stats
        estimated_success = choice.success_chance
        luck_bonus = (player_stats.get("luck", 10) - 10) * 0.01
        estimated_success = min(1.0, estimated_success + luck_bonus)

        preview = {
            "success_chance": estimated_success,
            "difficulty": choice.difficulty.value,
            "potential_benefits": [],
            "potential_risks": [],
        }

        # Analyze consequences
        consequences = choice.consequences

        if "stats" in consequences:
            for stat, change in consequences["stats"].items():
                if change > 0:
                    preview["potential_benefits"].append(f"{stat} +{change}")
                else:
                    preview["potential_risks"].append(f"{stat} {change}")

        if "add_items" in consequences:
            items = consequences["add_items"]
            preview["potential_benefits"].append(f"獲得物品: {', '.join(items)}")

        if "remove_items" in consequences:
            items = consequences["remove_items"]
            preview["potential_risks"].append(f"消耗物品: {', '.join(items)}")

        if "relationship" in consequences:
            change = consequences["relationship"]
            if change > 0:
                preview["potential_benefits"].append(f"關係改善 +{change}")
            else:
                preview["potential_risks"].append(f"關係惡化 {change}")

        # Add failure risks if success chance is low
        if estimated_success < 0.7:
            preview["potential_risks"].append("失敗風險較高")
            if "failure_penalty" in consequences:
                preview["potential_risks"].append("失敗會有額外懲罰")

        return preview

    def generate_emergency_choices(self, context: Dict[str, Any]) -> List[GameChoice]:
        """Generate emergency choices for critical situations"""

        emergency_choices = [
            GameChoice(
                choice_id="emergency_retreat",
                text="緊急撤退",
                choice_type=ChoiceType.ACTION,
                difficulty=ChoiceDifficulty.MEDIUM,
                requirements={},
                consequences={"safe_retreat": True, "stats": {"health": 10}},
                success_chance=0.9,
                description="立即脫離危險情況",
            ),
            GameChoice(
                choice_id="call_for_help",
                text="呼救求援",
                choice_type=ChoiceType.ACTION,
                difficulty=ChoiceDifficulty.EASY,
                requirements={},
                consequences={"help_requested": True, "stats": {"charisma": -1}},
                success_chance=0.8,
                description="向他人求助",
            ),
            GameChoice(
                choice_id="desperate_gamble",
                text="孤注一擲",
                choice_type=ChoiceType.ACTION,
                difficulty=ChoiceDifficulty.EXTREME,
                requirements={},
                consequences={"all_or_nothing": True, "stats": {"luck": -5}},
                success_chance=0.3,
                description="冒極大風險的最後手段",
            ),
        ]

        return emergency_choices

    def _generate_relationship_based_choices(
        self, context: Dict[str, Any], player_stats: Dict[str, int] = None  # type: ignore
    ) -> List[GameChoice]:
        """Generate choices based on character relationships"""

        choices = []
        active_characters = context.get("active_characters", [])
        relationships = context.get("relationships", {})

        for character in active_characters:
            if character == "player":
                continue

            char_relationship = relationships.get(character, 0)

            # High relationship choices
            if char_relationship > 50:
                choices.append(
                    GameChoice(
                        choice_id=f"deep_talk_{character.lower().replace(' ', '_')}",
                        text=f"與{character}進行深度交流",
                        choice_type=ChoiceType.DIALOGUE,
                        difficulty=ChoiceDifficulty.EASY,
                        requirements={},
                        consequences={
                            "stats": {"charisma": 2},
                            "relationship": {character: 5},
                            "special_dialogue": True,
                        },
                        success_chance=0.9,
                        description=f"與{character}的友好關係讓對話更順暢",
                    )
                )

            # Low relationship choices
            elif char_relationship < -20:
                choices.append(
                    GameChoice(
                        choice_id=f"reconcile_{character.lower().replace(' ', '_')}",
                        text=f"嘗試與{character}和解",
                        choice_type=ChoiceType.DIALOGUE,
                        difficulty=ChoiceDifficulty.HARD,
                        requirements={"stats": {"charisma": 12}},
                        consequences={
                            "stats": {"charisma": 1},
                            "relationship": {character: 15},
                            "reconciliation_attempt": True,
                        },
                        success_chance=0.4,
                        description=f"嘗試修復與{character}的關係",
                    )
                )

            # Neutral relationship choices
            else:
                choices.append(
                    GameChoice(
                        choice_id=f"approach_{character.lower().replace(' ', '_')}",
                        text=f"接近{character}",
                        choice_type=ChoiceType.SOCIAL,
                        difficulty=ChoiceDifficulty.MEDIUM,
                        requirements={},
                        consequences={
                            "stats": {"charisma": 1},
                            "relationship": {character: 2},
                            "social_interaction": True,
                        },
                        success_chance=0.7,
                        description=f"與{character}建立關係",
                    )
                )

        return choices[:3]  # 限制返回數量
