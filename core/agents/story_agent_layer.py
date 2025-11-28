"""
Story Agent Decision Layer

Integrates Agent autonomy into story turn processing.
Uses the safety wrapper to ensure all Agent actions are secure.

Agent can:
- Analyze the current story context
- Decide what actions to take (modify world, update stats, search memory, generate images)
- Execute actions through the safety wrapper
- Return decision context to story engine

This is the bridge between Story Engine and Agent Tool System.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentDecision:
    """Represents a decision made by the Agent"""

    def __init__(
        self,
        decision_type: str,
        tool_calls: List[Dict[str, Any]],
        reasoning: str,
        confidence: float = 1.0
    ):
        self.decision_type = decision_type
        self.tool_calls = tool_calls
        self.reasoning = reasoning
        self.confidence = confidence
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "tool_calls": self.tool_calls,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


class StoryAgentLayer:
    """
    Agent decision layer for story turns

    Determines when and how Agent should intervene in story progression.
    All tool calls go through the safety wrapper.
    """

    def __init__(self, safety_wrapper=None, llm_adapter=None):
        """
        Args:
            safety_wrapper: StorySafetyWrapper instance
            llm_adapter: LLM adapter for Agent reasoning
        """
        from .story_safety_wrapper import get_safety_wrapper
        from ..llm.adapter import get_llm_adapter

        self.safety_wrapper = safety_wrapper or get_safety_wrapper()
        self.llm_adapter = llm_adapter or get_llm_adapter()

        # Agent configuration
        self.enabled = True
        self.intervention_threshold = 0.7  # Confidence threshold for Agent intervention

    async def should_agent_intervene(
        self,
        session_id: str,
        player_input: str,
        narrative_text: str,
        context_memory: Any
    ) -> Tuple[bool, str]:
        """
        Determine if Agent should intervene in this turn

        Args:
            session_id: Game session ID
            player_input: Player's input text
            narrative_text: Generated narrative
            context_memory: StoryContextMemory instance

        Returns:
            (should_intervene: bool, reason: str)
        """
        if not self.enabled:
            return False, "Agent disabled"

        # Check for explicit Agent triggers in narrative
        agent_keywords = [
            "獲得", "失去", "發現", "達成", "解鎖",  # Chinese
            "gain", "lose", "discover", "achieve", "unlock",  # English
            "quest", "任務", "完成", "complete",
            "level", "等級", "升級", "level up",
            "damage", "傷害", "治療", "heal",
            "item", "物品", "道具"
        ]

        narrative_lower = narrative_text.lower()
        player_lower = player_input.lower()

        # Check if narrative contains agent-worthy events
        trigger_count = sum(
            1 for keyword in agent_keywords
            if keyword.lower() in narrative_lower or keyword.lower() in player_lower
        )

        if trigger_count >= 2:
            return True, f"Multiple event keywords detected ({trigger_count})"

        # Check for quest progression
        if context_memory:
            current_scene = context_memory.get_current_scene()
            if current_scene and current_scene.scene_objectives:
                return True, "Scene has active objectives"

        return False, "No Agent intervention needed"

    async def make_decision(
        self,
        session_id: str,
        player_input: str,
        narrative_text: str,
        context_memory: Any,
        session_stats: Dict[str, Any]
    ) -> Optional[AgentDecision]:
        """
        Let Agent analyze context and decide on actions

        Args:
            session_id: Game session ID
            player_input: Player's input
            narrative_text: Generated narrative
            context_memory: StoryContextMemory
            session_stats: Current player stats

        Returns:
            AgentDecision if Agent should act, None otherwise
        """
        try:
            # Build Agent prompt
            agent_prompt = self._build_agent_prompt(
                player_input,
                narrative_text,
                context_memory,
                session_stats
            )

            # Get Agent's analysis (using LLM)
            # For now, use rule-based decision making
            # In future, this could be replaced with LLM call

            tool_calls = []
            reasoning_parts = []

            # Rule 1: Quest completion detection
            if any(keyword in narrative_text.lower() for keyword in ["完成", "達成", "成功"]):
                quest_flag = self._extract_quest_from_narrative(narrative_text)
                if quest_flag:
                    tool_calls.append({
                        "tool": "modify_world_state",
                        "params": {
                            "flags": {quest_flag: True},
                            "reason": "Quest completion detected"
                        }
                    })
                    reasoning_parts.append(f"Detected quest completion: {quest_flag}")

            # Rule 2: Damage detection
            damage_amount = self._extract_damage_from_narrative(narrative_text)
            if damage_amount > 0:
                tool_calls.append({
                    "tool": "update_character_state",
                    "params": {
                        "stats": {"hp": -damage_amount},
                        "reason": "Player took damage",
                        "relative": True
                    }
                })
                reasoning_parts.append(f"Player took {damage_amount} damage")

            # Rule 3: Item acquisition detection
            item_name = self._extract_item_from_narrative(narrative_text)
            if item_name:
                tool_calls.append({
                    "tool": "add_inventory_item",
                    "params": {
                        "item": item_name,
                        "quantity": 1,
                        "reason": "Item acquired from narrative"
                    }
                })
                reasoning_parts.append(f"Player acquired item: {item_name}")

            # Rule 4: NPC encounter detection
            npc_name = self._extract_npc_from_context(context_memory)
            if npc_name:
                tool_calls.append({
                    "tool": "modify_world_state",
                    "params": {
                        "flags": {f"npc_met_{npc_name}": True},
                        "reason": "NPC encounter"
                    }
                })
                reasoning_parts.append(f"Player met NPC: {npc_name}")

            # Rule 5: Location discovery
            if any(keyword in narrative_text.lower() for keyword in ["進入", "到達", "發現", "來到"]):
                location = self._extract_location_from_context(context_memory)
                if location:
                    tool_calls.append({
                        "tool": "modify_world_state",
                        "params": {
                            "flags": {f"location_discovered_{location}": True},
                            "reason": "Location discovery"
                        }
                    })
                    reasoning_parts.append(f"Discovered location: {location}")

            if not tool_calls:
                return None

            decision = AgentDecision(
                decision_type="story_event_processing",
                tool_calls=tool_calls,
                reasoning="; ".join(reasoning_parts),
                confidence=0.8
            )

            return decision

        except Exception as e:
            logger.error(f"Failed to make Agent decision: {e}")
            return None

    async def execute_decision(
        self,
        session_id: str,
        decision: AgentDecision
    ) -> Dict[str, Any]:
        """
        Execute Agent decision through safety wrapper

        Args:
            session_id: Game session ID
            decision: AgentDecision to execute

        Returns:
            Execution results with success/failure info
        """
        results = {
            "decision_type": decision.decision_type,
            "reasoning": decision.reasoning,
            "tool_results": [],
            "overall_success": True,
            "errors": []
        }

        for tool_call in decision.tool_calls:
            tool_name = tool_call["tool"]
            params = tool_call["params"]

            try:
                # Execute through safety wrapper
                result = await self.safety_wrapper.execute_tool(
                    tool_name,
                    session_id,
                    params
                )

                results["tool_results"].append({
                    "tool": tool_name,
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                    "rollback_performed": result.rollback_performed
                })

                if not result.success:
                    results["overall_success"] = False
                    results["errors"].append(f"{tool_name}: {result.error}")

            except Exception as e:
                logger.error(f"Failed to execute tool {tool_name}: {e}")
                results["overall_success"] = False
                results["errors"].append(f"{tool_name}: {str(e)}")
                results["tool_results"].append({
                    "tool": tool_name,
                    "success": False,
                    "error": str(e)
                })

        return results

    def _build_agent_prompt(
        self,
        player_input: str,
        narrative_text: str,
        context_memory: Any,
        session_stats: Dict[str, Any]
    ) -> str:
        """Build prompt for Agent reasoning"""
        current_scene = context_memory.get_current_scene() if context_memory else None

        prompt = f"""You are a Story Agent analyzing game events.

Player Input: {player_input}
Narrative Response: {narrative_text}

Current Scene: {current_scene.title if current_scene else 'Unknown'}
Player Stats: HP {session_stats.get('hp', 0)}/{session_stats.get('max_hp', 100)}

Analyze this turn and determine what game state changes should occur.
Consider: quest progression, damage, item acquisition, NPC encounters, location discovery.
"""
        return prompt

    def _extract_quest_from_narrative(self, narrative: str) -> Optional[str]:
        """Extract quest flag from narrative"""
        # Simple pattern matching - could be improved with LLM
        if "龍" in narrative or "dragon" in narrative.lower():
            return "quest_dragon_complete"
        if "森林" in narrative or "forest" in narrative.lower():
            return "quest_forest_complete"
        return None

    def _extract_damage_from_narrative(self, narrative: str) -> int:
        """Extract damage amount from narrative"""
        import re
        # Look for damage patterns like "受到 20 傷害" or "took 20 damage"
        patterns = [
            r'受到\s*(\d+)\s*傷害',
            r'took\s*(\d+)\s*damage',
            r'(\d+)\s*點傷害',
            r'(\d+)\s*damage'
        ]

        for pattern in patterns:
            match = re.search(pattern, narrative, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return 0

    def _extract_item_from_narrative(self, narrative: str) -> Optional[str]:
        """Extract item name from narrative"""
        # Simple pattern matching
        if "劍" in narrative or "sword" in narrative.lower():
            return "sword"
        if "藥水" in narrative or "potion" in narrative.lower():
            return "health_potion"
        if "鑰匙" in narrative or "key" in narrative.lower():
            return "key"
        return None

    def _extract_npc_from_context(self, context_memory: Any) -> Optional[str]:
        """Extract NPC name from context"""
        if not context_memory:
            return None

        current_scene = context_memory.get_current_scene()
        if current_scene and current_scene.primary_npc:
            # Sanitize NPC name for flag
            npc_name = current_scene.primary_npc.lower().replace(" ", "_")
            return npc_name

        return None

    def _extract_location_from_context(self, context_memory: Any) -> Optional[str]:
        """Extract location name from context"""
        if not context_memory:
            return None

        current_scene = context_memory.get_current_scene()
        if current_scene and current_scene.location:
            # Sanitize location name for flag
            location = current_scene.location.lower().replace(" ", "_")
            return location

        return None


# Singleton instance
_agent_layer_instance: Optional[StoryAgentLayer] = None


def get_agent_layer(safety_wrapper=None, llm_adapter=None) -> StoryAgentLayer:
    """Get or create singleton Agent layer instance"""
    global _agent_layer_instance
    if _agent_layer_instance is None:
        _agent_layer_instance = StoryAgentLayer(safety_wrapper, llm_adapter)
    return _agent_layer_instance


def reset_agent_layer():
    """Reset singleton (for testing)"""
    global _agent_layer_instance
    _agent_layer_instance = None
