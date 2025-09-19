# core/agent/story_integration.py
"""
Agent-Story System Integration
Enables agents to interact with story/game systems for dynamic narrative generation
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import asyncio

from .base_agent import BaseAgent, SimpleReasoningAgent
from .multi_step_processor import MultiStepProcessor

logger = logging.getLogger(__name__)


@dataclass
class StoryContext:
    """Context for story-driven agent interactions"""

    story_id: str
    character_name: str
    current_scene: str
    character_state: Dict[str, Any]
    story_history: List[Dict[str, Any]]
    available_actions: List[str]
    narrative_style: str = "adventure"

    def get_context_summary(self) -> str:
        """Get formatted context for agent processing"""
        summary = f"Story: {self.story_id}\n"
        summary += f"Character: {self.character_name}\n"
        summary += f"Current Scene: {self.current_scene}\n"
        summary += f"Character State: {self.character_state}\n"

        if self.story_history:
            recent_events = self.story_history[-3:]  # Last 3 events
            summary += "Recent Events:\n"
            for event in recent_events:
                summary += (
                    f"- {event.get('action', 'Unknown')}: {event.get('result', '')}\n"
                )

        if self.available_actions:
            summary += f"Available Actions: {', '.join(self.available_actions)}\n"

        return summary


class StoryAgent(SimpleReasoningAgent):
    """
    Specialized agent for story/narrative interactions
    Integrates with game systems to provide intelligent story responses
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="story_agent",
            description="Intelligent story and narrative agent",
            **kwargs,
        )
        self.story_contexts: Dict[str, StoryContext] = {}

    async def process_story_action(
        self,
        story_context: StoryContext,
        player_action: str,
        action_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a player action within story context

        Args:
            story_context: Current story state and context
            player_action: Action requested by player
            action_parameters: Additional parameters for the action

        Returns:
            Story response with narrative, consequences, and state changes
        """
        try:
            # Store context for this story
            self.story_contexts[story_context.story_id] = story_context

            # Prepare task description for agent processing
            task_description = f"""
            Process story action in {story_context.narrative_style} style:

            Player Action: {player_action}
            {story_context.get_context_summary()}

            Generate an appropriate story response including:
            1. Narrative description of what happens
            2. Any consequences or state changes
            3. New available actions for the player
            4. Updated scene description if needed
            """

            # Add story-specific context
            story_context_data = {
                "story_mode": True,
                "narrative_style": story_context.narrative_style,
                "character_name": story_context.character_name,
                "current_scene": story_context.current_scene,
                "action_parameters": action_parameters or {},
            }

            # Execute the story processing task
            result = await self.execute_task(
                task_description=task_description, context=story_context_data
            )

            if result["success"]:
                # Parse the story response
                story_response = self._parse_story_response(
                    result["result"], story_context, player_action
                )

                # Update story history
                story_context.story_history.append(
                    {
                        "action": player_action,
                        "result": story_response.get("narrative", ""),
                        "parameters": action_parameters,
                        "timestamp": time.time(),
                    }
                )

                return {
                    "success": True,
                    "story_response": story_response,
                    "updated_context": story_context,
                    "agent_steps": result.get("steps_taken", 0),
                    "tools_used": result.get("tools_used", []),
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Story processing failed"),
                    "fallback_response": self._generate_fallback_response(
                        story_context, player_action
                    ),
                }

        except Exception as e:
            logger.error(f"Story action processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_response": self._generate_fallback_response(
                    story_context, player_action
                ),
            }

    def _parse_story_response(
        self, agent_result: str, story_context: StoryContext, player_action: str
    ) -> Dict[str, Any]:
        """Parse agent result into structured story response"""

        # Simple parsing - in production, use more sophisticated NLP
        lines = agent_result.split("\n")

        narrative = ""
        consequences = []
        new_actions = []
        scene_update = ""

        current_section = "narrative"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Section headers
            if "consequence" in line.lower() or "result" in line.lower():
                current_section = "consequences"
                continue
            elif "action" in line.lower() and (
                "available" in line.lower() or "next" in line.lower()
            ):
                current_section = "actions"
                continue
            elif "scene" in line.lower() and "update" in line.lower():
                current_section = "scene"
                continue

            # Content parsing
            if current_section == "narrative":
                narrative += line + " "
            elif current_section == "consequences":
                if line.startswith("-") or line.startswith("*"):
                    consequences.append(line[1:].strip())
                else:
                    consequences.append(line)
            elif current_section == "actions":
                if line.startswith("-") or line.startswith("*"):
                    new_actions.append(line[1:].strip())
                elif "," in line:
                    new_actions.extend([a.strip() for a in line.split(",")])
                else:
                    new_actions.append(line)
            elif current_section == "scene":
                scene_update += line + " "

        # Clean up and validate
        narrative = narrative.strip() or f"You {player_action}."

        if not new_actions:
            new_actions = ["look around", "continue", "wait"]

        return {
            "narrative": narrative,
            "consequences": consequences,
            "available_actions": new_actions[:5],  # Limit to 5 actions
            "scene_update": scene_update.strip(),
            "action_processed": player_action,
        }

    def _generate_fallback_response(
        self, story_context: StoryContext, player_action: str
    ) -> Dict[str, Any]:
        """Generate a simple fallback response when agent processing fails"""

        fallback_narratives = [
            f"You {player_action} in the {story_context.current_scene}.",
            f"You attempt to {player_action}. The result is uncertain.",
            f"You {player_action}, and the story continues to unfold.",
        ]

        narrative = fallback_narratives[hash(player_action) % len(fallback_narratives)]

        return {
            "narrative": narrative,
            "consequences": [],
            "available_actions": ["look around", "continue", "try something else"],
            "scene_update": "",
            "action_processed": player_action,
            "is_fallback": True,
        }

    async def generate_scene_description(
        self, story_context: StoryContext, scene_type: str = "descriptive"
    ) -> Dict[str, Any]:
        """Generate detailed scene description using agent capabilities"""

        task_description = f"""
        Generate a {scene_type} scene description for a {story_context.narrative_style} story:

        Scene: {story_context.current_scene}
        Character: {story_context.character_name}
        Character State: {story_context.character_state}

        Create an immersive description that:
        1. Sets the mood and atmosphere
        2. Describes key visual and sensory details
        3. Hints at potential interactions or discoveries
        4. Matches the {story_context.narrative_style} style
        """

        result = await self.execute_task(
            task_description=task_description,
            context={"story_mode": True, "scene_generation": True},
        )

        if result["success"]:
            return {
                "success": True,
                "scene_description": result["result"],
                "scene_type": scene_type,
                "generated_for": story_context.current_scene,
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Scene generation failed"),
                "fallback_description": f"You find yourself in {story_context.current_scene}.",
            }


class StoryAgentManager:
    """
    Manager for story agent instances and story-driven tasks
    Coordinates multiple agents for complex narrative scenarios
    """

    def __init__(self):
        self.agents: Dict[str, StoryAgent] = {}
        self.multi_step_processor = MultiStepProcessor()
        self.active_stories: Dict[str, StoryContext] = {}

        # Register story agent types
        self._initialize_story_agents()

    def _initialize_story_agents(self):
        """Initialize different types of story agents"""

        # Main story agent
        self.agents["narrative"] = StoryAgent(
            name="narrative_agent",
            description="Primary narrative and story processing agent",
        )

        # Dialogue specialist
        self.agents["dialogue"] = StoryAgent(
            name="dialogue_agent",
            description="Specialized in character dialogue and conversations",
        )

        # World building agent
        self.agents["world"] = StoryAgent(
            name="world_agent",
            description="Handles world building, locations, and environmental details",
        )

        # Combat/action agent
        self.agents["action"] = StoryAgent(
            name="action_agent",
            description="Processes combat, action sequences, and skill checks",
        )

    async def process_complex_story_scenario(
        self,
        story_context: StoryContext,
        scenario_type: str,
        scenario_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process complex story scenarios using multiple specialized agents

        Args:
            story_context: Current story state
            scenario_type: Type of scenario (dialogue, combat, exploration, etc.)
            scenario_data: Scenario-specific data and parameters

        Returns:
            Comprehensive scenario result with multi-agent contributions
        """
        try:
            # Create multi-step task for the scenario
            task_id = f"{story_context.story_id}_{scenario_type}_{int(time.time())}"

            scenario_task = await self.multi_step_processor.create_task(
                task_id=task_id,
                description=f"Process {scenario_type} scenario in story {story_context.story_id}",
                context={
                    "story_context": story_context,
                    "scenario_type": scenario_type,
                    "scenario_data": scenario_data,
                },
            )

            # Plan scenario steps based on type
            if scenario_type == "dialogue":
                await self._plan_dialogue_scenario(
                    scenario_task, story_context, scenario_data
                )
            elif scenario_type == "combat":
                await self._plan_combat_scenario(
                    scenario_task, story_context, scenario_data
                )
            elif scenario_type == "exploration":
                await self._plan_exploration_scenario(
                    scenario_task, story_context, scenario_data
                )
            else:
                # Generic scenario planning
                await self._plan_generic_scenario(
                    scenario_task, story_context, scenario_data
                )

            # Execute the multi-step scenario
            result = await self.multi_step_processor.execute_task(task_id)

            return {
                "success": True,
                "scenario_type": scenario_type,
                "scenario_result": result,
                "story_context": story_context,
            }

        except Exception as e:
            logger.error(f"Complex story scenario processing failed: {e}")
            return {"success": False, "error": str(e), "scenario_type": scenario_type}

    async def _plan_dialogue_scenario(self, task, story_context, scenario_data):
        """Plan steps for dialogue scenario"""
        task.add_step(
            description=f"Generate character dialogue for {scenario_data.get('npc_name', 'NPC')}",
            agent_name="dialogue",
        )

        task.add_step(
            description="Process player dialogue options and responses",
            agent_name="dialogue",
            dependencies=[0],
        )

    async def _plan_combat_scenario(self, task, story_context, scenario_data):
        """Plan steps for combat scenario"""
        task.add_step(
            description="Assess combat situation and initialize encounter",
            agent_name="action",
        )

        task.add_step(
            description="Process combat actions and determine outcomes",
            agent_name="action",
            dependencies=[0],
        )

        task.add_step(
            description="Generate narrative description of combat results",
            agent_name="narrative",
            dependencies=[1],
        )

    async def _plan_exploration_scenario(self, task, story_context, scenario_data):
        """Plan steps for exploration scenario"""
        task.add_step(
            description="Generate detailed environment description", agent_name="world"
        )

        task.add_step(
            description="Identify points of interest and interactive elements",
            agent_name="world",
            dependencies=[0],
        )

        task.add_step(
            description="Create exploration narrative and available actions",
            agent_name="narrative",
            dependencies=[1],
        )

    async def _plan_generic_scenario(self, task, story_context, scenario_data):
        """Plan steps for generic scenario"""
        task.add_step(
            description=f"Process {scenario_data.get('action', 'unknown action')} in current context",
            agent_name="narrative",
        )

    def get_story_agent(self, agent_type: str = "narrative") -> Optional[StoryAgent]:
        """Get specific story agent by type"""
        return self.agents.get(agent_type)

    def register_story_context(self, story_context: StoryContext):
        """Register story context for tracking"""
        self.active_stories[story_context.story_id] = story_context

    def get_active_stories(self) -> List[str]:
        """Get list of active story IDs"""
        return list(self.active_stories.keys())


import time
