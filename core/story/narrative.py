# =============================================================================
# core/story/narrative.py
"""
Narrative Generation System
Handles story generation, context management, and scene transitions
"""

import logging
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..llm.adapter import get_llm_adapter, ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class StoryContext:
    """Context for story generation"""

    current_scene: str
    previous_scenes: List[str]
    active_characters: List[str]
    current_location: str
    time_of_day: str
    weather: str
    mood: str
    plot_points: List[str]
    world_state: Dict[str, Any]


class NarrativeGenerator:
    """Generates narrative content using LLM"""

    def __init__(self):
        self.llm = get_llm_adapter()
        self.scene_templates = self._load_scene_templates()

    def _load_scene_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load scene templates for different story types"""
        return {
            "opening": {
                "template": "開始一個新的冒險故事，玩家名叫{player_name}。設定：{setting}",
                "mood": "adventurous",
                "length": "medium",
            },
            "exploration": {
                "template": "玩家正在探索{location}，描述他們發現的有趣事物",
                "mood": "curious",
                "length": "medium",
            },
            "encounter": {
                "template": "玩家遇到了{character}，描述這次相遇",
                "mood": "tense",
                "length": "long",
            },
            "resolution": {
                "template": "根據玩家的選擇{choice}，描述後果和故事發展",
                "mood": "consequential",
                "length": "medium",
            },
        }

    async def generate_scene(
        self,
        context: StoryContext,
        persona_prompt: str,
        player_input: str,
        scene_type: str = "exploration",
    ) -> str:
        """Generate a new scene based on context and player input"""

        template = self.scene_templates.get(
            scene_type, self.scene_templates["exploration"]
        )

        # Build context prompt
        context_prompt = f"""
場景背景：
- 當前位置：{context.current_location}
- 時間：{context.time_of_day}
- 天氣：{context.weather}
- 氣氛：{context.mood}
- 活躍角色：{', '.join(context.active_characters)}

故事發展要點：
{chr(10).join(f'- {point}' for point in context.plot_points)}

世界狀態：
{self._format_world_state(context.world_state)}
        """

        # Create messages for LLM
        messages = [
            ChatMessage(role="system", content=persona_prompt),
            ChatMessage(role="system", content=context_prompt),
            ChatMessage(role="user", content=f"玩家動作：{player_input}"),
        ]

        try:
            response = await self.llm.chat(messages)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate scene: {e}")
            return self._generate_fallback_scene(player_input, context)

    def _format_world_state(self, world_state: Dict[str, Any]) -> str:
        """Format world state for prompt"""
        if not world_state:
            return "無特殊狀態"

        formatted = []
        for key, value in world_state.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)

    def _generate_fallback_scene(self, player_input: str, context: StoryContext) -> str:
        """Generate fallback scene when LLM fails"""
        fallback_responses = [
            f"你決定{player_input}。周圍的環境似乎對你的行動有所回應...",
            f"當你{player_input}時，你注意到{context.current_location}中有些微妙的變化。",
            f"你的行動{player_input}在{context.current_location}中引起了一些有趣的反應。",
        ]
        return random.choice(fallback_responses)

    async def generate_opening_scene(
        self, player_name: str, setting: str, persona_prompt: str
    ) -> str:
        """Generate opening scene for new game"""
        opening_prompt = f"""
創造一個引人入勝的開場場景：
- 玩家名字：{player_name}
- 遊戲設定：{setting}
- 要求：設定背景、介紹環境、提供初始目標
- 長度：2-3段落
- 語調：神秘而吸引人
        """

        messages = [
            ChatMessage(role="system", content=persona_prompt),
            ChatMessage(role="user", content=opening_prompt),
        ]

        try:
            response = await self.llm.chat(messages)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate opening scene: {e}")
            return f"""
歡迎，{player_name}！

你發現自己身處在{setting}的神秘世界中。四周的景象既陌生又充滿可能性，
空氣中瀰漫著冒險的氣息。你知道，一個偉大的旅程即將開始...

你準備好踏出第一步了嗎？
            """.strip()
