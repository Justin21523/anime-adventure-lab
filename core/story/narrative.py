# =============================================================================
# core/story/narrative.py
"""
Narrative Generation System
Handles story generation, context management, and scene transitions
"""

import logging
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from ..llm import ChatMessage, LLMResponse, get_llm_adapter

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
        try:
            self.llm = get_llm_adapter()
            if not self.llm.is_available():
                logger.warning("LLM adapter not available, will use fallback responses")
        except Exception as e:
            logger.error(f"Failed to initialize LLM adapter: {e}")
            self.llm = None
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
        """Generate narrative scene based on context"""

        if self.llm is None:
            return self._generate_fallback_scene(player_input, context)

        # Build context prompt
        context_prompt = self._build_context_prompt(context)

        # Get scene template
        template = self.scene_templates.get(
            scene_type, self.scene_templates["exploration"]
        )

        # Create generation prompt
        generation_prompt = f"""
根據以下情境生成故事場景：

{context_prompt}

玩家行動：{player_input}
場景類型：{scene_type}
場景模板：{template['template']}
預期情緒：{template['mood']}
預期長度：{template['length']}

生成要求：
1. 描述生動的場景和環境
2. 反映玩家行動的結果
3. 保持與之前情節的連貫性
4. 為下一步行動設置懸念
5. 字數控制在 150-300 字之間
        """

        messages = [
            ChatMessage(role="system", content=persona_prompt),
            ChatMessage(role="user", content=generation_prompt),
        ]

        try:
            response = await self.llm.chat(messages)  # type: ignore

            # 統一處理響應類型 - 修正 LLMResponse 問題
            if hasattr(response, "content"):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()

        except Exception as e:
            logger.error(f"Failed to generate scene: {e}")
            return self._generate_fallback_scene(player_input, context)

    def _build_context_prompt(self, context: StoryContext) -> str:
        """Build context description from StoryContext"""
        context_parts = [
            f"當前場景：{context.current_scene}",
            f"當前位置：{context.current_location}",
            f"時間：{context.time_of_day}",
            f"天氣：{context.weather}",
            f"情緒氛圍：{context.mood}",
        ]

        if context.previous_scenes:
            context_parts.append(
                f"之前場景：{' -> '.join(context.previous_scenes[-3:])}"
            )

        if context.active_characters:
            context_parts.append(f"在場角色：{', '.join(context.active_characters)}")

        if context.plot_points:
            context_parts.append(f"劇情要點：{'; '.join(context.plot_points[-3:])}")

        if context.world_state:
            world_state_desc = self._format_world_state(context.world_state)
            context_parts.append(f"世界狀態：\n{world_state_desc}")

        return "\n".join(context_parts)

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

        if self.llm is None:
            return f"""
歡迎，{player_name}！

你發現自己身處在{setting}的神秘世界中。四周的景象既陌生又充滿可能性，
空氣中瀰漫著冒險的氣息。你知道，一個偉大的旅程即將開始...

你準備好踏出第一步了嗎？
            """.strip()

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
            response = await self.llm.chat(messages)  # type: ignore

            # 統一處理響應類型
            if hasattr(response, "content"):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()

        except Exception as e:
            logger.error(f"Failed to generate opening scene: {e}")
            return f"""
歡迎，{player_name}！

你發現自己身處在{setting}的神秘世界中。四周的景象既陌生又充滿可能性，
空氣中瀰漫著冒險的氣息。你知道，一個偉大的旅程即將開始...

你準備好踏出第一步了嗎？
            """.strip()

    async def generate_character_dialogue(
        self,
        character_name: str,
        dialogue_context: str,
        personality_traits: List[str],
        current_mood: str = "neutral",
    ) -> str:
        """Generate character dialogue based on personality"""

        if self.llm is None:
            return f"[{character_name}沉默地看著你，似乎在思考著什麼...]"

        dialogue_prompt = f"""
為角色 {character_name} 生成對話：

角色特徵：{', '.join(personality_traits)}
當前情緒：{current_mood}
對話情境：{dialogue_context}

要求：
1. 體現角色的個性特徵
2. 符合當前情緒狀態
3. 推進劇情發展
4. 長度控制在50-150字
        """

        messages = [
            ChatMessage(role="system", content="你是一個專業的角色對話創作者"),
            ChatMessage(role="user", content=dialogue_prompt),
        ]

        try:
            response = await self.llm.chat(messages)  # type: ignore

            if hasattr(response, "content"):
                dialogue = response.content.strip()
            elif isinstance(response, str):
                dialogue = response.strip()
            else:
                dialogue = str(response).strip()

            # Clean up dialogue (remove quotes if present)
            dialogue = dialogue.strip('"').strip("'").strip()

            return dialogue if dialogue else f"[{character_name}沉默地看著你...]"

        except Exception as e:
            logger.error(f"Failed to generate character dialogue: {e}")
            return f"[{character_name}沉默地看著你，似乎在思考著什麼...]"

    async def generate_scene_description(
        self,
        location: str,
        atmosphere: str,
        time_of_day: str,
        weather: str,
        focus_elements: List[str] = None,  # type: ignore
    ) -> str:
        """Generate detailed scene description"""

        if self.llm is None:
            return f"你身處在{location}中，{atmosphere}的氣氛讓這個{time_of_day}顯得格外特別。"

        focus_text = ""
        if focus_elements:
            focus_text = f"特別描述：{', '.join(focus_elements)}"

        scene_prompt = f"""
生成詳細的場景描述：

地點：{location}
氣氛：{atmosphere}
時間：{time_of_day}
天氣：{weather}
{focus_text}

要求：
1. 生動的視覺描述
2. 感官細節（聲音、氣味、觸感）
3. 營造相應的氣氛
4. 長度控制在100-200字
        """

        messages = [
            ChatMessage(role="system", content="你是一個專業的場景描述創作者"),
            ChatMessage(role="user", content=scene_prompt),
        ]

        try:
            response = await self.llm.chat(messages)  # type: ignore

            if hasattr(response, "content"):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()

        except Exception as e:
            logger.error(f"Failed to generate scene description: {e}")
            return f"你身處在{location}中，{atmosphere}的氣氛讓這個{time_of_day}顯得格外特別。"

    def generate_narrative(
        self, context: Dict[str, Any], choice_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate basic narrative (獨立實作，不依賴父類)"""
        try:
            player_input = context.get("player_input", "")
            location = context.get("current_location", "未知地點")
            characters = context.get("characters", [])

            narrative_parts = []

            # 位置設定
            if location != "未知地點":
                narrative_parts.append(f"在{location}")

            # 角色情境
            if characters and isinstance(characters, list):
                char_names = [
                    str(char) for char in characters if str(char) != "player"
                ][:2]
                if char_names:
                    narrative_parts.append(f"{', '.join(char_names)}也在這裡")

            # 行動回應
            if player_input:
                if choice_result and choice_result.get("success"):
                    narrative_parts.append("你的行動取得了積極的結果")
                else:
                    narrative_parts.append("你採取了行動")

            if narrative_parts:
                return (
                    "，".join(narrative_parts)
                    + "。故事繼續展開，每個選擇都可能帶來新的可能性。"
                )
            else:
                return "你的冒險故事在這裡繼續，充滿了無限的可能性。"

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Narrative generation error: {e}")
            return "你的冒險故事在這裡繼續，充滿了無限的可能性。"

    def generate_scene_transitions(
        self, from_location: str, to_location: str, transition_method: str = "移動"
    ) -> str:
        """Generate scene transition narrative"""

        transition_templates = {
            "移動": f"你離開了{from_location}，朝著{to_location}前進。",
            "傳送": f"一陣光芒包圍了你，當光芒消散時，你發現自己已經到了{to_location}。",
            "時間": f"時間流逝，場景從{from_location}轉換到了{to_location}。",
            "夢境": f"恍惚間，{from_location}的景象消失了，取而代之的是{to_location}的畫面。",
        }

        base_template = transition_templates.get(
            transition_method, transition_templates["移動"]
        )

        # Add some randomized details
        detail_options = [
            "路上發生的事情讓你印象深刻。",
            "這次轉換讓你對接下來的冒險充滿期待。",
            "新的環境帶來了新的可能性。",
            "你感受到了變化的氣息。",
        ]

        detail = random.choice(detail_options)

        return f"{base_template} {detail}"

    def get_narrative_suggestions(
        self, player_input: str, context: StoryContext
    ) -> List[str]:
        """Get narrative direction suggestions based on input"""

        input_lower = player_input.lower()
        suggestions = []

        # Action-based suggestions
        if any(word in input_lower for word in ["探索", "查看", "觀察"]):
            suggestions.append("詳細描述環境和可發現的物品")
            suggestions.append("引入新的謎題或線索")

        elif any(word in input_lower for word in ["說話", "對話", "交談"]):
            suggestions.append("深入發展角色關係")
            suggestions.append("透過對話推進主要劇情")

        elif any(word in input_lower for word in ["戰鬥", "攻擊", "戰鬥"]):
            suggestions.append("描述動作序列和戰術")
            suggestions.append("展現角色技能和成長")

        elif any(word in input_lower for word in ["離開", "前往", "移動"]):
            suggestions.append("場景轉換和新環境介紹")
            suggestions.append("路途中的遭遇和發現")

        # Context-based suggestions
        if context.mood == "tense":
            suggestions.append("維持緊張氣氛但提供喘息機會")
        elif context.mood == "peaceful":
            suggestions.append("利用寧靜時刻進行角色發展")

        # Default suggestions if none matched
        if not suggestions:
            suggestions = ["推進主要故事線", "引入新的角色或元素", "深化當前場景的描述"]

        return suggestions[:3]  # Return top 3 suggestions


# =============================================================================
# Enhanced Narrative Generator (from story_system.py integration)
# =============================================================================


class EnhancedNarrativeGenerator(NarrativeGenerator):
    """Enhanced narrative generator with full context awareness"""

    def __init__(self):
        super().__init__()
        self.character_voice_cache = {}
        self.narrative_templates = self._load_enhanced_templates()
        self.context_analyzers = self._setup_context_analyzers()

    def _setup_context_analyzers(self) -> Dict[str, Callable]:
        """Setup context analysis functions"""
        return {
            "mood_analyzer": self._analyze_mood_context,
            "character_analyzer": self._analyze_character_context,
            "location_analyzer": self._analyze_location_context,
            "progression_analyzer": self._analyze_story_progression,
        }

    # 補充缺失的上下文分析方法
    def _analyze_mood_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mood and emotional context"""
        mood_indicators = {
            "positive_keywords": ["成功", "快樂", "勝利", "發現", "友好"],
            "negative_keywords": ["失敗", "危險", "恐懼", "憤怒", "悲傷"],
            "neutral_keywords": ["探索", "觀察", "思考", "等待", "準備"],
        }

        text_content = " ".join(
            [
                str(context.get("player_input", "")),
                str(context.get("scene_description", "")),
                str(context.get("recent_events", [])),
            ]
        )

        mood_scores = {}
        for mood, keywords in mood_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_content)
            mood_scores[mood] = score

        dominant_mood = max(mood_scores.items(), key=lambda x: x[1])[0]

        return {
            "dominant_mood": dominant_mood.replace("_keywords", ""),
            "mood_scores": mood_scores,
            "emotional_intensity": max(mood_scores.values())
            / max(len(mood_indicators["positive_keywords"]), 1),
        }

    def _analyze_character_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze character interaction context"""
        characters = context.get("present_characters", [])
        relationships = context.get("relationships", {})

        analysis = {
            "character_count": len(characters),
            "social_complexity": "low",
            "relationship_dynamics": {},
            "interaction_potential": 0.5,
        }

        if len(characters) > 3:
            analysis["social_complexity"] = "high"
            analysis["interaction_potential"] = 0.8
        elif len(characters) > 1:
            analysis["social_complexity"] = "medium"
            analysis["interaction_potential"] = 0.7

        # Analyze relationships
        for char in characters:
            if char in relationships:
                rel_value = relationships[char]
                if rel_value > 20:
                    analysis["relationship_dynamics"][char] = "positive"
                elif rel_value < -10:
                    analysis["relationship_dynamics"][char] = "negative"
                else:
                    analysis["relationship_dynamics"][char] = "neutral"

        return analysis

    def _analyze_location_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze location and environmental context"""
        location = context.get("current_location", "未知地點")
        weather = context.get("weather", "")
        time_of_day = context.get("time_of_day", "")

        location_types = {
            "urban": ["城市", "城鎮", "市場", "商店", "街道"],
            "natural": ["森林", "山脈", "河流", "海邊", "草原"],
            "mystical": ["神殿", "遺跡", "洞穴", "魔法", "神秘"],
            "domestic": ["家", "房屋", "村莊", "農場", "客棧"],
        }

        location_type = "unknown"
        for ltype, keywords in location_types.items():
            if any(keyword in location for keyword in keywords):
                location_type = ltype
                break

        return {
            "location": location,
            "location_type": location_type,
            "environmental_mood": self._determine_environmental_mood(
                location, weather, time_of_day
            ),
            "exploration_potential": self._calculate_exploration_potential(
                location_type
            ),
            "safety_level": self._assess_location_safety(location_type),
        }

    def _analyze_story_progression(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze story progression and narrative flow"""
        recent_events = context.get("recent_events", [])
        turn_count = context.get("turn_count", 0)
        completed_objectives = context.get("completed_objectives", [])

        progression_stage = "beginning"
        if turn_count > 50:
            progression_stage = "advanced"
        elif turn_count > 20:
            progression_stage = "middle"
        elif turn_count > 5:
            progression_stage = "developing"

        story_momentum = "steady"
        if len(recent_events) > 5:
            story_momentum = "fast"
        elif len(recent_events) < 2:
            story_momentum = "slow"

        return {
            "progression_stage": progression_stage,
            "story_momentum": story_momentum,
            "narrative_complexity": len(completed_objectives) + len(recent_events),
            "plot_development_suggestions": self._generate_plot_suggestions(
                progression_stage, story_momentum
            ),
        }

    def _load_enhanced_templates(self) -> Dict[str, Any]:
        """Load enhanced narrative templates"""
        return {
            "scene_transitions": {
                "location_change": "隨著{player_name}離開{old_location}，前往{new_location}，環境發生了顯著的變化...",
                "time_passage": "時間悄悄流逝，{time_description}，為接下來的事件蒙上了新的色彩...",
                "mood_shift": "氣氛突然{mood_change}，讓在場的每個人都感受到了這種轉變...",
            },
            "character_interactions": {
                "first_meeting": "{character_name}首次出現時，{appearance_description}。你能感受到{personality_hint}...",
                "relationship_change": "{character_name}看你的眼神{relationship_description}，你們之間的關係似乎{change_direction}...",
                "dialogue_context": "在{location}的{atmosphere}氛圍中，{character_name}{emotion_state}地說道...",
            },
            "plot_development": {
                "revelation": "突然間，一個重要的真相浮出水面: {revelation_content}。這改變了一切...",
                "conflict_escalation": "情況變得更加複雜，{conflict_description}讓事態向{direction}發展...",
                "objective_completion": "隨著{objective}的完成，新的可能性展現在眼前...",
            },
        }

    def generate_narrative(
        self, context: Dict[str, Any], choice_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate basic narrative (獨立實作，不依賴父類)"""
        try:
            # Extract key context elements
            player_input = context.get("player_input", "")
            location = context.get("current_location", "未知地點")
            characters = context.get("characters", [])
            scene_description = context.get("scene_description", "")

            # Build narrative components
            narrative_parts = []

            # Location setting
            if location and location != "未知地點":
                narrative_parts.append(f"在{location}")

            # Character context
            if characters and isinstance(characters, list):
                char_names = [
                    str(char) for char in characters if str(char) != "player"
                ][:2]
                if char_names:
                    if len(char_names) == 1:
                        narrative_parts.append(f"{char_names[0]}也在這裡")
                    else:
                        narrative_parts.append(f"{', '.join(char_names)}都在場")

            # Action context
            if player_input:
                action_response = self._generate_simple_action_response(
                    player_input, choice_result
                )
                narrative_parts.append(action_response)

            # Choice outcome
            if choice_result:
                outcome_text = self._generate_simple_outcome_text(choice_result)
                if outcome_text:
                    narrative_parts.append(outcome_text)

            # Combine parts
            if narrative_parts:
                base_narrative = "，".join(narrative_parts)
                return (
                    base_narrative
                    + "。故事繼續展開，每個選擇都可能帶來意想不到的結果。"
                )
            else:
                return f"你在{location if location != '未知地點' else '這個神秘的地方'}繼續著冒險，思考著下一步的行動。"

        except Exception as e:
            logger.error(f"Narrative generation error: {e}")
            return "你的冒險故事在這裡繼續，充滿了無限的可能性。"

    # 輔助方法
    def _determine_environmental_mood(
        self, location: str, weather: str, time_of_day: str
    ) -> str:
        """Determine environmental mood based on location factors"""
        mood_factors = []

        if "黑暗" in location or "夜晚" in time_of_day:
            mood_factors.append("mysterious")
        if "陽光" in weather or "明亮" in location:
            mood_factors.append("cheerful")
        if "暴風" in weather or "危險" in location:
            mood_factors.append("tense")

        return mood_factors[0] if mood_factors else "neutral"

    def _calculate_exploration_potential(self, location_type: str) -> float:
        """Calculate exploration potential based on location type"""
        potential_map = {
            "natural": 0.8,
            "mystical": 0.9,
            "urban": 0.6,
            "domestic": 0.4,
            "unknown": 0.5,
        }
        return potential_map.get(location_type, 0.5)

    def _assess_location_safety(self, location_type: str) -> str:
        """Assess safety level of location"""
        safety_map = {
            "domestic": "safe",
            "urban": "moderate",
            "natural": "moderate",
            "mystical": "dangerous",
            "unknown": "unknown",
        }
        return safety_map.get(location_type, "unknown")

    def _generate_plot_suggestions(self, stage: str, momentum: str) -> List[str]:
        """Generate plot development suggestions"""
        suggestions_map = {
            ("beginning", "slow"): ["引入新角色", "建立世界觀", "設定初始目標"],
            ("beginning", "steady"): ["發展角色關係", "探索環境", "建立衝突"],
            ("beginning", "fast"): ["穩定節奏", "深化背景", "建立情感連結"],
            ("developing", "slow"): ["加入轉折", "引入新挑戰", "發展子情節"],
            ("developing", "steady"): ["推進主線", "發展角色弧線", "建立高潮"],
            ("developing", "fast"): ["整合情節線", "準備重大事件", "解決次要衝突"],
            ("middle", "slow"): ["重大揭露", "角色成長時刻", "改變現狀"],
            ("middle", "steady"): ["準備高潮", "解決核心衝突", "角色面對選擇"],
            ("middle", "fast"): ["高潮事件", "重大決戰", "故事轉折點"],
            ("advanced", "slow"): ["解決餘留問題", "角色反思", "準備結局"],
            ("advanced", "steady"): ["故事收尾", "角色命運確定", "主題昇華"],
            ("advanced", "fast"): ["快速解決", "開放式結局", "後續可能性"],
        }

        return suggestions_map.get((stage, momentum), ["繼續發展當前情節"])

    def generate_contextual_narrative(
        self,
        context_memory: Any,
        player_input: str,
        choice_result: Optional[Dict[str, Any]] = None,
        forced_scene_type: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate narrative with full context awareness (完整實作)"""

        try:
            # 基本上下文分析
            current_scene = (
                context_memory.get_current_scene()
                if hasattr(context_memory, "get_current_scene")
                else None
            )
            present_characters = (
                context_memory.get_characters_in_scene()
                if hasattr(context_memory, "get_characters_in_scene")
                else []
            )

            # 建構分析上下文
            analysis_context = {
                "player_input": player_input,
                "current_location": (
                    current_scene.location if current_scene else "未知地點"
                ),
                "present_characters": [
                    char.name for char in present_characters if hasattr(char, "name")
                ],
                "scene_description": current_scene.description if current_scene else "",
                "choice_result": choice_result,
                "recent_events": (
                    getattr(context_memory, "narrative_memory", [])[-3:]
                    if hasattr(context_memory, "narrative_memory")
                    else []
                ),
            }

            # 進行上下文分析
            mood_analysis = self._analyze_mood_context(analysis_context)
            character_analysis = self._analyze_character_context(analysis_context)
            location_analysis = self._analyze_location_context(analysis_context)
            progression_analysis = self._analyze_story_progression(analysis_context)

            # 生成主要敘事
            main_narrative = self._generate_enhanced_narrative(
                analysis_context, mood_analysis, character_analysis, location_analysis
            )

            return {
                "main_narrative": main_narrative,
                "character_reactions": self._generate_character_reactions_enhanced(
                    present_characters, mood_analysis
                ),
                "scene_developments": self._generate_scene_developments_enhanced(
                    current_scene, choice_result
                ),
                "narrative_suggestions": progression_analysis[
                    "plot_development_suggestions"
                ],
                "mood_indicators": [mood_analysis["dominant_mood"]],
                "context_analysis": {
                    "mood": mood_analysis,
                    "characters": character_analysis,
                    "location": location_analysis,
                    "progression": progression_analysis,
                },
            }

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Enhanced narrative generation failed: {e}")

            # 降級到基本生成
            return {
                "main_narrative": self.generate_narrative(
                    {"player_input": player_input}, choice_result
                ),
                "character_reactions": {},
                "scene_developments": {},
                "narrative_suggestions": ["繼續探索", "與角色交談", "觀察環境"],
                "mood_indicators": ["neutral"],
                "context_analysis": {},
            }

    def _generate_enhanced_narrative(
        self,
        context: Dict[str, Any],
        mood_analysis: Dict[str, Any],
        character_analysis: Dict[str, Any],
        location_analysis: Dict[str, Any],
    ) -> str:
        """Generate enhanced narrative based on comprehensive analysis"""

        # 基本敘事框架
        location = context.get("current_location", "未知地點")
        player_input = context.get("player_input", "")
        choice_result = context.get("choice_result", {})

        # 根據分析結果調整敘事風格
        mood = mood_analysis.get("dominant_mood", "neutral")
        social_complexity = character_analysis.get("social_complexity", "low")
        environmental_mood = location_analysis.get("environmental_mood", "neutral")

        narrative_parts = []

        # 環境設定
        if location != "未知地點":
            env_desc = self._get_enhanced_environment_description(
                location, environmental_mood
            )
            narrative_parts.append(f"在{location}的{env_desc}中")

        # 社交情境
        if social_complexity != "low":
            present_chars = context.get("present_characters", [])
            if present_chars:
                char_context = self._generate_character_context_narrative(
                    present_chars, character_analysis
                )
                narrative_parts.append(char_context)

        # 行動回應
        if player_input:
            action_response = self._generate_mood_adjusted_action_response(
                player_input, choice_result, mood
            )
            narrative_parts.append(action_response)

        # 合併並添加情緒調節
        base_narrative = (
            "，".join(narrative_parts) if narrative_parts else "故事在這裡繼續發展"
        )

        # 根據情緒強度調整結尾
        emotional_intensity = mood_analysis.get("emotional_intensity", 0.5)
        ending = self._generate_mood_appropriate_ending(mood, emotional_intensity)

        return base_narrative + ending

    def _get_enhanced_environment_description(self, location: str, mood: str) -> str:
        """Get enhanced environment description based on mood"""
        mood_descriptors = {
            "cheerful": "明亮而溫暖的氛圍",
            "mysterious": "神秘而引人入勝的環境",
            "tense": "緊張而充滿未知的氣氛",
            "neutral": "平靜的環境",
        }
        return mood_descriptors.get(mood, "特殊的氛圍")

    def _generate_character_context_narrative(
        self, characters: List[str], analysis: Dict[str, Any]
    ) -> str:
        """Generate narrative about character interactions"""
        char_count = len(characters)
        relationships = analysis.get("relationship_dynamics", {})

        if char_count == 1:
            char = characters[0]
            rel_type = relationships.get(char, "neutral")
            rel_desc = {"positive": "友好地", "negative": "謹慎地", "neutral": ""}[
                rel_type
            ]
            return f"{char}{rel_desc}在場"
        else:
            char_list = "、".join(characters[:2])
            return f"{char_list}等人也在這裡，氣氛顯得複雜"

    def _generate_mood_adjusted_action_response(
        self, player_input: str, choice_result: Dict[str, Any], mood: str
    ) -> str:
        """Generate action response adjusted for mood"""
        success = choice_result.get("success", True)

        mood_adjustments = {
            "positive": {
                "success": "取得了出色的成果",
                "failure": "雖然遇到挫折，但保持樂觀",
            },
            "negative": {
                "success": "在困境中找到了突破",
                "failure": "情況變得更加艱難",
            },
            "neutral": {"success": "獲得了預期的結果", "failure": "遇到了一些阻礙"},
        }

        outcome = "success" if success else "failure"
        adjustment = mood_adjustments.get(mood, mood_adjustments["neutral"])

        return f"你的行動{adjustment[outcome]}"

    def _generate_mood_appropriate_ending(self, mood: str, intensity: float) -> str:
        """Generate mood-appropriate narrative ending"""
        endings = {
            "positive": "。故事朝著光明的方向發展，充滿希望的可能性在前方等待。",
            "negative": "。陰霾籠罩著前路，但挑戰中往往隱藏著成長的機會。",
            "neutral": "。故事平穩地向前推進，每個選擇都蘊含著改變的力量。",
        }

        base_ending = endings.get(mood, endings["neutral"])

        # 根據情緒強度調整
        if intensity > 0.7:
            return base_ending.replace("。", "！")
        elif intensity < 0.3:
            return base_ending.replace("。", "...")
        else:
            return base_ending

    def _generate_character_reactions_enhanced(
        self, characters: List[Any], mood_analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate enhanced character reactions"""
        reactions = {}
        dominant_mood = mood_analysis.get("dominant_mood", "neutral")

        for char in characters:
            if hasattr(char, "character_id") and char.character_id != "player":
                reaction = self._generate_individual_character_reaction(
                    char, dominant_mood
                )
                if reaction:
                    reactions[
                        char.name if hasattr(char, "name") else char.character_id
                    ] = reaction

        return reactions

    def _generate_individual_character_reaction(
        self, character: Any, mood: str
    ) -> Optional[str]:
        """Generate individual character reaction based on mood"""
        personality_traits = getattr(character, "personality_traits", [])

        # 基於角色特質和情緒的反應
        trait_reactions = {
            "智慧": {
                "positive": "若有所思地點頭表示贊同",
                "negative": "皺眉思考應對之策",
                "neutral": "冷靜地觀察情況發展",
            },
            "友善": {
                "positive": "露出了真心的微笑",
                "negative": "關切地詢問你的狀況",
                "neutral": "友好地注視著你",
            },
            "神秘": {
                "positive": "眼中閃過一絲難以捉摸的光芒",
                "negative": "更加深沉地保持著沉默",
                "neutral": "保持著一貫的神秘表情",
            },
        }

        # 找到匹配的特質
        for trait in personality_traits:
            if trait in trait_reactions:
                return trait_reactions[trait].get(
                    mood, trait_reactions[trait]["neutral"]
                )

        # 預設反應
        default_reactions = {
            "positive": "顯得心情不錯",
            "negative": "看起來有些擔心",
            "neutral": "保持著平常的表情",
        }

        return default_reactions.get(mood, default_reactions["neutral"])

    def _generate_scene_developments_enhanced(
        self, current_scene: Any, choice_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate enhanced scene developments"""
        developments = {}

        if not choice_result:
            return developments

        consequences = choice_result.get("consequences", {})

        # 場景轉換檢查
        if "scene_transition" in consequences or "location_change" in consequences:
            developments["scene_transition"] = {
                "triggered": True,
                "reason": "player choice consequences",
            }

        # 氣氛變化檢查
        success = choice_result.get("success", True)
        if current_scene and hasattr(current_scene, "atmosphere"):
            new_atmosphere = self._determine_new_atmosphere(
                current_scene, success, consequences
            )
            if new_atmosphere != getattr(current_scene, "atmosphere", None):
                developments["atmosphere_change"] = {
                    "from": str(getattr(current_scene, "atmosphere", "unknown")),
                    "to": new_atmosphere,
                    "reason": "choice outcome influence",
                }

        return developments

    def _determine_new_atmosphere(
        self, scene: Any, success: bool, consequences: Dict[str, Any]
    ) -> str:
        """Determine new scene atmosphere based on outcomes"""
        if not success:
            return "tense"
        elif "positive_outcome" in consequences:
            return "peaceful"
        elif "mystery" in str(consequences).lower():
            return "mysterious"
        elif "combat" in str(consequences).lower():
            return "exciting"
        else:
            return "neutral"

    def _generate_simple_action_response(
        self, player_input: str, choice_result: Optional[Dict[str, Any]]
    ) -> str:
        """Generate simple action response"""
        success = choice_result.get("success", True) if choice_result else True

        action_keywords = {
            "探索": "探索" if success else "搜尋",
            "交談": "對話" if success else "交流嘗試",
            "戰鬥": "戰鬥" if success else "衝突",
            "觀察": "觀察" if success else "查看",
            "休息": "休息" if success else "稍作停留",
        }

        for keyword, response in action_keywords.items():
            if keyword in player_input:
                if success:
                    return f"你的{response}取得了良好的效果"
                else:
                    return f"雖然{response}遇到了困難，但仍有收穫"

        return "你採取了行動" if success else "你的嘗試遇到了一些挑戰"

    def _generate_simple_outcome_text(
        self, choice_result: Dict[str, Any]
    ) -> Optional[str]:
        """Generate simple outcome text"""
        consequences = choice_result.get("consequences", {})
        outcomes = []

        if "stats" in consequences:
            positive_changes = [
                stat for stat, change in consequences["stats"].items() if change > 0
            ]
            if positive_changes:
                outcomes.append("你感到自己有所成長")

        if "relationship" in consequences or "relationships" in consequences:
            outcomes.append("人際關係發生了微妙的變化")

        if "discovery" in consequences or "discovery_chance" in consequences:
            outcomes.append("你有了新的發現")

        return "，".join(outcomes) if outcomes else None

    async def _generate_main_narrative(self, context: Dict[str, Any]) -> str:
        """Generate the main narrative text"""

        player_input = context.get("player_input", "")
        location = context.get("current_location", "未知地點")
        atmosphere = context.get("scene_atmosphere", "平靜")
        choice_result = context.get("choice_result", {})

        # Basic narrative framework
        narrative_parts = []

        # Location and atmosphere setting
        if location != "未知地點":
            atmosphere_desc = self._get_atmosphere_description(atmosphere)
            narrative_parts.append(f"在{location}的{atmosphere_desc}中")

        # Player action response
        if player_input:
            action_response = self._generate_action_response(
                player_input, choice_result
            )
            narrative_parts.append(action_response)

        # Choice consequence narration
        if choice_result:
            consequence_narration = self._generate_consequence_narration(choice_result)
            if consequence_narration:
                narrative_parts.append(consequence_narration)

        # Environmental details
        environmental_details = self._generate_environmental_details(context)
        if environmental_details:
            narrative_parts.append(environmental_details)

        # Combine narrative parts
        if narrative_parts:
            return "，".join(narrative_parts) + "。"
        else:
            return "故事在這裡繼續展開，充滿了無限的可能性。"

    def _get_atmosphere_description(self, atmosphere: str) -> str:
        """Get description for scene atmosphere"""
        atmosphere_map = {
            "peaceful": "寧靜祥和的氛圍",
            "tense": "緊張的氣氛",
            "mysterious": "神秘的環境",
            "exciting": "令人興奮的場面",
            "neutral": "平靜的環境",
        }
        return atmosphere_map.get(atmosphere.lower(), "特殊的氛圍")

    def _generate_action_response(
        self, player_input: str, choice_result: Dict[str, Any]
    ) -> str:
        """Generate response to player action"""

        success = choice_result.get("success", True)

        if "探索" in player_input or "investigate" in player_input.lower():
            if success:
                return "你的探索獲得了有意義的發現"
            else:
                return "儘管仔細探索，但這次沒有明顯的收穫"

        elif "交談" in player_input or "talk" in player_input.lower():
            if success:
                return "對話進行得很順利，你獲得了有用的信息"
            else:
                return "交談遇到了一些困難，但仍有收穫"

        elif "戰鬥" in player_input or "fight" in player_input.lower():
            if success:
                return "你在這次衝突中佔據了上風"
            else:
                return "戰況激烈，結果不如預期"

        else:
            if success:
                return "你的行動取得了積極的結果"
            else:
                return "事情的發展並不完全如你所願"

    def _generate_consequence_narration(
        self, choice_result: Dict[str, Any]
    ) -> Optional[str]:
        """Generate narration for choice consequences"""

        consequences = choice_result.get("consequences", {})
        narration_parts = []

        # Stat changes
        if "stats" in consequences:
            stat_changes = consequences["stats"]
            positive_changes = [
                stat for stat, change in stat_changes.items() if change > 0
            ]
            negative_changes = [
                stat for stat, change in stat_changes.items() if change < 0
            ]

            if positive_changes:
                narration_parts.append("你感到自己在某些方面有所成長")
            if negative_changes:
                narration_parts.append("這次經歷消耗了你的一些精力")

        # Relationship changes
        if "relationships" in consequences or "relationship" in consequences:
            narration_parts.append("你與周圍人物的關係發生了微妙的變化")

        # Item changes
        if "add_items" in consequences:
            items = consequences["add_items"]
            if items:
                narration_parts.append(f"你獲得了一些有用的物品")

        if "remove_items" in consequences:
            items = consequences["remove_items"]
            if items:
                narration_parts.append("你消耗了一些資源")

        return "，".join(narration_parts) if narration_parts else None

    def _generate_environmental_details(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate environmental details based on context"""

        location = context.get("current_location", "")
        characters = context.get("present_characters", [])

        details = []

        # Character presence
        if characters:
            if len(characters) == 1:
                details.append(f"{characters[0]}也在這裡")
            elif len(characters) > 1:
                char_list = "、".join(characters[:2])
                if len(characters) > 2:
                    char_list += "等人"
                details.append(f"{char_list}也在場")

        # Location-based details
        location_details = self._get_location_details(location)
        if location_details:
            details.append(location_details)

        return "，".join(details) if details else None

    def _get_location_details(self, location: str) -> Optional[str]:
        """Get environmental details for specific locations"""

        location_map = {
            "森林": "樹葉在微風中輕柔地搖擺",
            "城鎮": "遠處傳來熙熙攘攘的人聲",
            "山洞": "空氣中帶著潮濕的味道",
            "海邊": "海浪輕拍著岸邊的沙灘",
            "城堡": "古老的石牆透露著歷史的痕跡",
        }

        for key, detail in location_map.items():
            if key in location:
                return detail

        return None

    async def _generate_character_reactions(
        self, context: Dict[str, Any], present_characters: List[Any]
    ) -> Dict[str, str]:
        """Generate character reactions to the narrative"""

        reactions = {}

        for character in present_characters:
            if character.character_id == "player":
                continue

            # Generate basic reaction based on character personality
            reaction = self._generate_character_reaction(character, context)
            if reaction:
                reactions[character.name] = reaction

        return reactions

    def _generate_character_reaction(
        self, character: Any, context: Dict[str, Any]
    ) -> Optional[str]:
        """Generate individual character reaction"""

        personality_traits = getattr(character, "personality_traits", [])
        current_state = getattr(character, "current_state", None)

        # Simple reaction based on personality and state
        if "智慧" in personality_traits:
            return "若有所思地觀察著情況的發展"
        elif "友善" in personality_traits:
            return "露出了理解和支持的表情"
        elif "神秘" in personality_traits:
            return "保持著難以捉摸的表情"

        # State-based reactions
        if current_state:
            state_value = (
                current_state.value
                if hasattr(current_state, "value")
                else str(current_state)
            )
            if state_value == "happy":
                return "顯得心情愉悅"
            elif state_value == "suspicious":
                return "用審視的眼光打量著周圍"
            elif state_value == "fearful":
                return "顯得有些緊張和警戒"

        return None

    async def _generate_scene_developments(
        self, context: Dict[str, Any], current_scene: Any
    ) -> Dict[str, Any]:
        """Generate scene developments and changes"""

        developments = {}

        # Check for potential scene transitions
        if self._should_trigger_scene_transition(context):
            developments["scene_transition"] = {
                "suggested": True,
                "reason": "narrative flow suggests a location change",
            }

        # Check for mood changes
        new_mood = self._calculate_scene_mood(context)
        if current_scene and hasattr(current_scene, "atmosphere"):
            current_mood = (
                current_scene.atmosphere.value
                if hasattr(current_scene.atmosphere, "value")
                else str(current_scene.atmosphere)
            )
            if new_mood != current_mood:
                developments["mood_change"] = {
                    "from": current_mood,
                    "to": new_mood,
                    "reason": "narrative events have shifted the atmosphere",
                }

        # Check for new character introductions
        if self._should_introduce_characters(context):
            developments["character_introduction"] = {
                "suggested": True,
                "type": "contextual_character",
            }

        return developments

    def _should_trigger_scene_transition(self, context: Dict[str, Any]) -> bool:
        """Determine if a scene transition should be triggered"""

        choice_result = context.get("choice_result", {})
        consequences = choice_result.get("consequences", {})

        # Check for explicit scene transition flags
        if "scene_transition" in consequences or "location_change" in consequences:
            return True

        # Check for narrative cues
        player_input = context.get("player_input", "").lower()
        transition_keywords = ["離開", "前往", "探索新", "move to", "go to", "leave"]

        return any(keyword in player_input for keyword in transition_keywords)

    def _calculate_scene_mood(self, context: Dict[str, Any]) -> str:
        """Calculate appropriate scene mood based on context"""

        choice_result = context.get("choice_result", {})
        success = choice_result.get("success", True)
        consequences = choice_result.get("consequences", {})

        # Determine mood based on events
        if not success:
            return "tense"
        elif "positive_outcome" in consequences:
            return "peaceful"
        elif "mystery" in consequences or "unknown" in str(consequences):
            return "mysterious"
        elif "combat" in consequences or "conflict" in consequences:
            return "exciting"
        else:
            return "neutral"

    def _should_introduce_characters(self, context: Dict[str, Any]) -> bool:
        """Determine if new characters should be introduced"""

        present_characters = context.get("present_characters", [])
        choice_result = context.get("choice_result", {})

        # Don't introduce too many characters at once
        if len(present_characters) >= 3:
            return False

        # Check if narrative suggests character introduction
        consequences = choice_result.get("consequences", {})
        if "meet_character" in consequences or "character_encounter" in consequences:
            return True

        # Random chance for character introduction in certain contexts
        import random

        return random.random() < 0.2  # 20% chance

    def _generate_narrative_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """Generate narrative suggestions for the current context"""

        suggestions = []
        present_characters = context.get("present_characters", [])
        location = context.get("current_location", "")

        # Character-based suggestions
        if present_characters:
            suggestions.append(f"與{present_characters[0]}進行更深入的交流")

        # Location-based suggestions
        if "森林" in location:
            suggestions.extend(["探索更深處的森林", "尋找隱藏的路徑"])
        elif "城鎮" in location:
            suggestions.extend(["訪問當地商店", "了解當地傳說"])
        elif "城堡" in location:
            suggestions.extend(["探索古老的房間", "尋找歷史文獻"])

        # General suggestions
        suggestions.extend(["仔細觀察周圍環境", "準備應對未知挑戰", "回顧已獲得的信息"])

        return suggestions[:5]  # Return top 5 suggestions

    def _extract_mood_indicators(self, narrative: str) -> List[str]:
        """Extract mood indicators from narrative text"""

        mood_keywords = {
            "peaceful": ["寧靜", "祥和", "平靜", "輕柔"],
            "tense": ["緊張", "警戒", "危險", "困難"],
            "mysterious": ["神秘", "未知", "隱藏", "秘密"],
            "exciting": ["興奮", "激烈", "刺激", "驚人"],
            "melancholy": ["憂鬱", "悲傷", "沉重", "孤獨"],
        }

        detected_moods = []

        for mood, keywords in mood_keywords.items():
            if any(keyword in narrative for keyword in keywords):
                detected_moods.append(mood)

        return detected_moods if detected_moods else ["neutral"]

    def _extract_context_flags(
        self, context: Dict[str, Any], choice_result: Optional[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Extract context flags from narrative generation"""

        flags = {}

        # Character interaction flags
        if context.get("present_characters"):
            flags["character_interaction"] = True

        # Combat flags
        if choice_result and "combat" in str(choice_result.get("consequences", {})):
            flags["combat_occurred"] = True

        # Discovery flags
        if (
            choice_result
            and choice_result.get("success")
            and "探索" in context.get("player_input", "")
        ):
            flags["discovery_made"] = True

        # Relationship flags
        if choice_result and "relationship" in choice_result.get("consequences", {}):
            flags["relationship_changed"] = True

        return flags

    # 額外的輔助方法
    def get_character_voice(self, character_id: str, context: Dict[str, Any]) -> str:
        """Get character-specific voice/style for dialogue generation"""

        if character_id in self.character_voice_cache:
            return self.character_voice_cache[character_id]

        # Default voice styles based on character ID patterns
        voice_patterns = {
            "wise": "充滿智慧且語重心長的語調",
            "merchant": "熱情且具說服力的商業語調",
            "guard": "正式且略帶權威的語調",
            "child": "天真且充滿好奇的語調",
            "elder": "慈祥且經驗豐富的語調",
        }

        character_voice = "平穩且友善的語調"  # Default

        for pattern, voice in voice_patterns.items():
            if pattern in character_id.lower():
                character_voice = voice
                break

        self.character_voice_cache[character_id] = character_voice
        return character_voice

    def generate_dialogue(
        self,
        character_id: str,
        content: str,
        context: Dict[str, Any],
        emotional_state: Optional[str] = None,
    ) -> str:
        """Generate character dialogue with appropriate voice and emotion"""

        voice_style = self.get_character_voice(character_id, context)

        # Adjust content based on emotional state
        if emotional_state:
            if emotional_state == "angry":
                content = f"(憤怒地) {content}"
            elif emotional_state == "happy":
                content = f"(愉快地) {content}"
            elif emotional_state == "sad":
                content = f"(憂鬱地) {content}"
            elif emotional_state == "excited":
                content = f"(興奮地) {content}"

        return f"以{voice_style}說道：「{content}」"

    def create_narrative_summary(self, session_events: List[Dict[str, Any]]) -> str:
        """Create a summary of session events for context"""

        if not session_events:
            return "這是一個全新的冒險開始。"

        summary_parts = []

        # Count different event types
        choices_made = len(
            [e for e in session_events if e.get("type") == "choice_executed"]
        )
        characters_met = len(
            set([e.get("character") for e in session_events if e.get("character")])
        )
        locations_visited = len(
            set([e.get("location") for e in session_events if e.get("location")])
        )

        if choices_made > 0:
            summary_parts.append(f"已做出{choices_made}個重要決定")

        if characters_met > 0:
            summary_parts.append(f"遇見了{characters_met}個角色")

        if locations_visited > 1:
            summary_parts.append(f"探索了{locations_visited}個不同地點")

        if summary_parts:
            return f"在這次冒險中，你{('、'.join(summary_parts))}。"
        else:
            return "你的冒險之旅剛剛開始。"

    def _determine_narrative_focus(
        self,
        player_input: str,
        current_scene: Any,
        present_characters: List[Any],
        choice_result: Optional[Dict[str, Any]],
    ) -> str:
        """Determine what the narrative should focus on"""

        input_lower = player_input.lower()

        if any(word in input_lower for word in ["說話", "對話", "交談", "問"]):
            if present_characters:
                return "character_dialogue"

        if any(word in input_lower for word in ["探索", "查看", "觀察", "尋找"]):
            return "environment_exploration"

        if any(word in input_lower for word in ["戰鬥", "攻擊", "防禦", "逃跑"]):
            return "combat_action"

        if any(word in input_lower for word in ["移動", "前往", "離開", "進入"]):
            return "location_transition"

        if choice_result and choice_result.get("dramatic_weight", 0) > 5:
            return "dramatic_consequence"

        # Default focus based on scene type
        if current_scene:
            scene_type_str = (
                str(current_scene.scene_type).lower()
                if hasattr(current_scene, "scene_type")
                else ""
            )
            if "dialogue" in scene_type_str:
                return "character_interaction"
            elif "combat" in scene_type_str:
                return "action_sequence"
            elif "exploration" in scene_type_str:
                return "environment_discovery"

        return "general_progression"

    def _build_comprehensive_context_prompt(
        self, context_memory: Any, player_input: str, narrative_focus: str
    ) -> str:
        """Build comprehensive context prompt for narrative generation"""

        current_scene = context_memory.get_current_scene()
        present_characters = context_memory.get_characters_in_scene()

        # Recent story context (last 3 scenes)
        recent_scenes = []
        for scene_id in context_memory.scene_sequence[-3:]:
            scene = context_memory.scenes.get(scene_id)
            if scene:
                recent_scenes.append(f"- {scene.title}: {scene.description[:100]}...")

        # Character context
        character_context = []
        for char in present_characters:
            recent_dialogue = (
                char.get_recent_dialogue(2)
                if hasattr(char, "get_recent_dialogue")
                else []
            )
            dialogue_summary = ""
            if recent_dialogue:
                last_dialogue = recent_dialogue[-1]
                dialogue_summary = f"最近說過: '{last_dialogue['content'][:50]}...'"

            character_context.append(
                f"- {char.name} ({char.role.value if hasattr(char.role, 'value') else char.role}): "
                f"{char.current_state.value if hasattr(char.current_state, 'value') else char.current_state}狀態, "
                f"關係: {context_memory.player_relationships.get(char.character_id, 0)}/10, "
                f"{dialogue_summary}"
            )

        # World state context
        world_context = []
        if context_memory.world_flags:
            active_flags = [
                f"{k}: {v}" for k, v in context_memory.world_flags.items() if v
            ]
            if active_flags:
                world_context.append(f"世界狀態: {', '.join(active_flags[:5])}")

        if context_memory.main_plot_points:
            world_context.append(
                f"主要劇情: {'; '.join(context_memory.main_plot_points[-3:])}"
            )

        # Recent player decisions context
        recent_decisions = []
        for decision in context_memory.player_decisions[-3:]:
            recent_decisions.append(f"- 第{decision['turn']}回: {decision['decision']}")

        context_prompt = f"""
故事上下文資訊：

【當前場景】
場景: {current_scene.title if current_scene else '未知'}
位置: {current_scene.location if current_scene else '未知'}
時間: {current_scene.time_of_day if current_scene else '未知'}
氣氛: {current_scene.atmosphere.value if current_scene and hasattr(current_scene.atmosphere, 'value') else '中性'}
場景目標: {', '.join(current_scene.scene_objectives) if current_scene and current_scene.scene_objectives else '探索'}

【在場角色】
{chr(10).join(character_context) if character_context else '無其他角色'}

【最近場景歷史】
{chr(10).join(recent_scenes) if recent_scenes else '故事剛開始'}

【世界狀態】
{chr(10).join(world_context) if world_context else '無特殊狀態'}

【最近玩家決定】
{chr(10).join(recent_decisions) if recent_decisions else '尚無重要決定'}

【當前玩家行動】
玩家輸入: {player_input}
敘述重點: {narrative_focus}

【生成要求】
1. 根據上下文生成連貫的故事敘述
2. 體現角色個性和當前狀態
3. 推進劇情但保持適度懸念
4. 字數控制在200-400字之間
5. 考慮場景氣氛和時間流逝
        """

        return context_prompt

    async def _generate_character_dialogues(
        self,
        context_memory: Any,
        present_characters: List[Any],
        player_input: str,
        main_narrative: str,
    ) -> List[Dict[str, str]]:
        """Generate contextual character dialogues"""

        dialogues = []

        for character in present_characters:
            # Skip player character
            if hasattr(character, "role") and str(character.role).lower() == "player":
                continue

            # Check if character should speak
            should_speak = self._should_character_speak(
                character, player_input, main_narrative, context_memory
            )

            if should_speak:
                dialogue = await self._generate_single_character_dialogue(
                    character, context_memory, player_input, main_narrative
                )

                if dialogue:
                    dialogues.append(
                        {
                            "character_id": character.character_id,
                            "character_name": character.name,
                            "content": dialogue,
                            "emotional_state": (
                                character.current_state.value
                                if hasattr(character.current_state, "value")
                                else str(character.current_state)
                            ),
                        }
                    )

                    # Update character dialogue history if method exists
                    if hasattr(character, "add_dialogue"):
                        current_scene = context_memory.get_current_scene()
                        turn_number = current_scene.turn_number if current_scene else 0
                        character.add_dialogue(dialogue, turn_number, player_input)

        return dialogues

    def _should_character_speak(
        self,
        character: Any,
        player_input: str,
        main_narrative: str,
        context_memory: Any,
    ) -> bool:
        """Determine if character should speak in current context"""

        input_lower = player_input.lower()
        narrative_lower = main_narrative.lower()
        char_name_lower = character.name.lower()

        # Character is directly addressed
        if char_name_lower in input_lower:
            return True

        # Character is mentioned in narrative
        if char_name_lower in narrative_lower:
            return True

        # Dialogue-focused scene
        current_scene = context_memory.get_current_scene()
        if current_scene and hasattr(current_scene, "scene_type"):
            scene_type_str = str(current_scene.scene_type).lower()
            if "dialogue" in scene_type_str:
                return True

        # Character is primary NPC and hasn't spoken recently
        if (
            hasattr(current_scene, "primary_npc")
            and character.character_id == current_scene.primary_npc
        ):
            if getattr(character, "interaction_count", 0) < 2:
                return True

        # Random chance for ambient dialogue
        role_str = str(character.role).lower() if hasattr(character, "role") else ""
        ambient_chance = 0.3 if "companion" in role_str else 0.1
        return random.random() < ambient_chance

    async def _generate_single_character_dialogue(
        self,
        character: Any,
        context_memory: Any,
        player_input: str,
        main_narrative: str,
    ) -> Optional[str]:
        """Generate dialogue for a single character"""

        if self.llm is None:
            return None

        # Get relationship context
        relationship_score = context_memory.player_relationships.get(
            character.character_id, 0
        )

        # Build character-specific context
        personality_traits = getattr(character, "personality_traits", ["友善"])
        speaking_style = getattr(character, "speaking_style", "直接")
        current_state = (
            character.current_state.value
            if hasattr(character.current_state, "value")
            else str(character.current_state)
        )

        # Get recent dialogue if available
        recent_dialogue = []
        if hasattr(character, "get_recent_dialogue"):
            recent_dialogue = character.get_recent_dialogue(2)

        character_prompt = f"""
角色資訊：
- 姓名：{character.name}
- 角色：{character.role.value if hasattr(character.role, 'value') else character.role}
- 個性特徵：{', '.join(personality_traits)}
- 說話風格：{speaking_style}
- 當前狀態：{current_state}
- 與玩家關係評分：{relationship_score}/10

最近互動記錄：
{chr(10).join([f"- {d['content']}" for d in recent_dialogue]) if recent_dialogue else "無"}

當前情境：
- 玩家行動：{player_input}
- 場景描述：{main_narrative[:100]}...

請生成一段符合角色個性的對話，要求：
1. 體現角色的說話風格和個性特徵
2. 反映當前的情緒狀態
3. 考慮與玩家的關係
4. 推進對話或劇情
5. 長度控制在30-80字之間
        """

        persona_prompt = getattr(character, "persona_prompt", "你是一個故事角色")

        messages = [
            ChatMessage(role="system", content=persona_prompt),
            ChatMessage(role="user", content=character_prompt),
        ]

        try:
            response = await self.llm.chat(messages)  # type: ignore

            if hasattr(response, "content"):
                dialogue = response.content.strip()
            elif isinstance(response, str):
                dialogue = response.strip()
            else:
                dialogue = str(response).strip()

            # Clean up dialogue
            dialogue = dialogue.strip('"').strip("'").strip()

            return dialogue if dialogue else None

        except Exception as e:
            logger.error(f"Failed to generate dialogue for {character.name}: {e}")
            return None

    def _analyze_scene_changes(
        self,
        context_memory: Any,
        player_input: str,
        choice_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze if scene changes are needed"""

        changes = {
            "location_change": False,
            "time_passage": False,
            "mood_shift": False,
            "character_changes": [],
            "new_scene_needed": False,
        }

        input_lower = player_input.lower()

        # Check for location change keywords
        if any(word in input_lower for word in ["離開", "前往", "進入", "回到"]):
            changes["location_change"] = True
            changes["new_scene_needed"] = True

        # Check for time passage indicators
        if any(word in input_lower for word in ["等待", "休息", "睡覺", "過了"]):
            changes["time_passage"] = True

        # Check choice consequences
        if choice_result:
            if choice_result.get("scene_transitions"):
                changes["new_scene_needed"] = True

            if choice_result.get("dramatic_weight", 0) > 3:
                changes["mood_shift"] = True

        return changes

    def _detect_mood_shift(self, context_memory: Any, narrative: str) -> Optional[str]:
        """Detect mood shifts from narrative content"""

        narrative_lower = narrative.lower()

        mood_indicators = {
            "tense": ["緊張", "危險", "威脅", "小心", "警戒"],
            "peaceful": ["平靜", "寧靜", "安全", "放鬆", "舒適"],
            "mysterious": ["神秘", "奇怪", "詭異", "未知", "隱藏"],
            "exciting": ["興奮", "激動", "刺激", "驚喜", "發現"],
            "melancholy": ["憂鬱", "悲傷", "失落", "懷念", "遺憾"],
        }

        for mood, indicators in mood_indicators.items():
            if any(indicator in narrative_lower for indicator in indicators):
                return mood

        return None
