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
import json
import os
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentDecision:
    """Represents a decision made by the Agent"""

    def __init__(
        self,
        decision_type: str,
        tool_calls: List[Dict[str, Any]],
        reasoning: str,
        confidence: float = 1.0,
        contributors: Optional[List[Dict[str, Any]]] = None,
    ):
        self.decision_type = decision_type
        self.tool_calls = tool_calls
        self.reasoning = reasoning
        self.confidence = confidence
        self.contributors = contributors or []
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "tool_calls": self.tool_calls,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "contributors": self.contributors,
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
        self.decision_mode = os.getenv("STORY_AGENT_DECISION_MODE", "multi_agent_hybrid")
        self.max_tool_calls = int(os.getenv("STORY_AGENT_MAX_TOOL_CALLS", "6"))
        self.max_llm_calls = int(os.getenv("STORY_AGENT_MAX_LLM_CALLS", "1"))
        try:
            from core.agents.catalog import STORY_ALLOWED_TOOLS

            self.allowed_tools = {str(t.get("id") or "").strip() for t in STORY_ALLOWED_TOOLS if isinstance(t, dict)}
            self.allowed_tools = {t for t in self.allowed_tools if t}
        except Exception:
            self.allowed_tools = {
                "modify_world_state",
                "update_character_state",
                "add_inventory_item",
                "update_relationship_state",
                "rag_search",
                "generate_scene_image",
            }
        from .story_orchestrator import StoryOrchestrator

        self.orchestrator = StoryOrchestrator(
            llm_adapter=self.llm_adapter,
            allowed_tools=self.allowed_tools,
            max_tool_calls=self.max_tool_calls,
            max_llm_calls=self.max_llm_calls,
        )

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

        # Multi-agent director mode: evaluate every turn (actual tool_calls may still be empty)
        if str(self.decision_mode).startswith("multi_agent"):
            try:
                profile = self._load_agent_profile(session_id)
                if profile is not None and not bool(getattr(profile, "enabled", True)):
                    return False, "Multi-agent director disabled by agent_profile"
            except Exception:
                pass
            return True, "Multi-agent director enabled"

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

    def _load_agent_profile(self, session_id: str):
        """Load WorldPack.agent_profile snapshot from session.story_context (best-effort)."""
        try:
            from schemas.world import WorldAgentProfile
        except Exception:
            return None

        try:
            from core.story.engine import get_story_engine

            engine = get_story_engine()
            session = engine.get_session(session_id)
            story_ctx = getattr(getattr(session, "current_state", None), "story_context", {}) or {}
            raw = story_ctx.get("agent_profile") or {}
            if isinstance(raw, WorldAgentProfile):
                return raw
            if isinstance(raw, dict):
                return WorldAgentProfile(**raw)
            return WorldAgentProfile()
        except Exception:
            return WorldAgentProfile()

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON object extraction from LLM output."""
        if not text or not str(text).strip():
            return None
        raw = str(text).strip()
        # If already JSON
        if raw.startswith("{") and raw.endswith("}"):
            try:
                return json.loads(raw)
            except Exception:
                return None

        # Try to find the first {...} block
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _sanitize_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop invalid tool calls and cap total size."""
        sanitized: List[Dict[str, Any]] = []
        for call in tool_calls or []:
            if not isinstance(call, dict):
                continue
            tool = call.get("tool") or call.get("tool_name")
            params = call.get("params") or call.get("parameters") or {}
            agent = call.get("agent")
            reason = call.get("reason")

            if tool not in self.allowed_tools:
                continue
            if not isinstance(params, dict):
                continue

            sanitized.append(
                {
                    "tool": tool,
                    "params": params,
                    **({"agent": agent} if agent else {}),
                    **({"reason": reason} if reason else {}),
                }
            )

            if len(sanitized) >= self.max_tool_calls:
                break

        # Deduplicate by (tool + params JSON)
        seen = set()
        unique: List[Dict[str, Any]] = []
        for call in sanitized:
            key = (call["tool"], json.dumps(call.get("params", {}), sort_keys=True, ensure_ascii=False))
            if key in seen:
                continue
            seen.add(key)
            unique.append(call)
        return unique[: self.max_tool_calls]

    def _heuristic_subdecisions(
        self,
        session_id: str,
        player_input: str,
        narrative_text: str,
        context_memory: Any,
        session_stats: Dict[str, Any],
        world_id: str,
        flags: Dict[str, Any],
        story_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return a list of sub-decisions (heuristics) with tool_calls."""
        subs: List[Dict[str, Any]] = []

        # --- NPC tracker agent ------------------------------------------------
        npc_tool_calls: List[Dict[str, Any]] = []
        try:
            present = context_memory.get_characters_in_scene() if context_memory else []
            for char in present:
                char_id = getattr(char, "character_id", None)
                role = getattr(getattr(char, "role", None), "value", None) or getattr(char, "role", None)
                if not char_id or char_id in {"player", "narrator"}:
                    continue
                if str(role) in {"npc", "companion", "antagonist"}:
                    flag_name = f"npc_met_{char_id}"
                    if not flags.get(flag_name):
                        npc_tool_calls.append(
                            {
                                "tool": "modify_world_state",
                                "params": {"flags": {flag_name: True}, "reason": "NPC/角色出現在場景中"},
                                "agent": "npc_tracker",
                            }
                        )
        except Exception:
            pass
        if npc_tool_calls:
            subs.append(
                {
                    "agent": "npc_tracker",
                    "reasoning": f"標記已遇見角色：{len(npc_tool_calls)}",
                    "tool_calls": npc_tool_calls,
                }
            )

        # --- Location tracker agent ------------------------------------------
        loc_tool_calls: List[Dict[str, Any]] = []
        try:
            current_scene = context_memory.get_current_scene() if context_memory else None
            location = getattr(current_scene, "location", None) or story_context.get("current_location")
            if location:
                location_key = str(location).strip().replace(" ", "_")
                flag_name = f"location_discovered_{location_key}"
                if not flags.get(flag_name):
                    loc_tool_calls.append(
                        {
                            "tool": "modify_world_state",
                            "params": {"flags": {flag_name: True}, "reason": "到達/位於此地點"},
                            "agent": "location_tracker",
                        }
                    )
        except Exception:
            pass
        if loc_tool_calls:
            subs.append(
                {
                    "agent": "location_tracker",
                    "reasoning": "標記已發現地點",
                    "tool_calls": loc_tool_calls,
                }
            )

        # --- Event extractor (damage/item/quest) -----------------------------
        evt_calls: List[Dict[str, Any]] = []
        evt_reasons: List[str] = []

        # Damage
        damage_amount = self._extract_damage_from_narrative(narrative_text)
        if damage_amount > 0:
            evt_calls.append(
                {
                    "tool": "update_character_state",
                    "params": {"stats": {"health": -damage_amount}, "reason": "敘事中檢測到傷害", "relative": True},
                    "agent": "event_extractor",
                }
            )
            evt_reasons.append(f"傷害 {damage_amount}")

        # Item
        item_name = self._extract_item_from_narrative(narrative_text)
        if item_name:
            evt_calls.append(
                {
                    "tool": "add_inventory_item",
                    "params": {"item": item_name, "quantity": 1, "reason": "敘事中獲得物品"},
                    "agent": "event_extractor",
                }
            )
            # Also flag
            evt_calls.append(
                {
                    "tool": "modify_world_state",
                    "params": {"flags": {f"item_acquired_{item_name}": True}, "reason": "敘事中獲得物品"},
                    "agent": "event_extractor",
                }
            )
            evt_reasons.append(f"物品 {item_name}")

        # Quest (very simple)
        quest_flag = self._extract_quest_from_narrative(narrative_text)
        if quest_flag:
            evt_calls.append(
                {
                    "tool": "modify_world_state",
                    "params": {"flags": {quest_flag: True}, "reason": "敘事中檢測到任務進度"},
                    "agent": "event_extractor",
                }
            )
            evt_reasons.append(f"任務 {quest_flag}")

        if evt_calls:
            subs.append(
                {
                    "agent": "event_extractor",
                    "reasoning": " / ".join(evt_reasons) if evt_reasons else "事件抽取",
                    "tool_calls": evt_calls,
                }
            )

        # --- Lore retriever agent (optional) ---------------------------------
        lore_calls: List[Dict[str, Any]] = []
        try:
            rag_enabled = bool(story_context.get("enrich_with_rag")) or bool(story_context.get("rag_auto"))
            if rag_enabled:
                query = (
                    (story_context.get("rag_query") or "").strip()
                    or player_input.strip()
                )
                if query:
                    lore_calls.append(
                        {
                            "tool": "rag_search",
                            "params": {"query": query, "top_k": 3, "world_id": world_id},
                            "agent": "lore_retriever",
                        }
                    )
        except Exception:
            pass
        if lore_calls:
            subs.append(
                {
                    "agent": "lore_retriever",
                    "reasoning": "根據玩家輸入檢索世界知識庫（RAG）",
                    "tool_calls": lore_calls,
                }
            )

        return subs

    def _llm_director_subdecision(
        self,
        player_input: str,
        narrative_text: str,
        context_memory: Any,
        session_stats: Dict[str, Any],
        inventory: List[str],
        flags: Dict[str, Any],
        world_id: str,
        worldpack_summary: str,
    ) -> Optional[Dict[str, Any]]:
        """One optional LLM director sub-agent that proposes tool calls as JSON."""
        if not self.llm_adapter or self.max_llm_calls <= 0:
            return None

        try:
            current_scene = context_memory.get_current_scene() if context_memory else None
            present = context_memory.get_characters_in_scene() if context_memory else []
            present_compact = [
                {
                    "character_id": getattr(c, "character_id", ""),
                    "name": getattr(c, "name", ""),
                    "role": getattr(getattr(c, "role", None), "value", None) or getattr(c, "role", ""),
                }
                for c in present
            ]

            prompt = f"""你是一個「Story Director」AI 代理，負責把敘事事件轉成安全的遊戲狀態變更（tool_calls）。

世界(world_id): {world_id}
世界摘要:
{worldpack_summary}

玩家輸入:
{player_input}

敘事回應:
{narrative_text}

當前場景:
{getattr(current_scene, 'title', '')} / {getattr(current_scene, 'location', '')}

在場角色:
{json.dumps(present_compact, ensure_ascii=False)}

玩家狀態:
{json.dumps(session_stats, ensure_ascii=False)}

背包:
{json.dumps(inventory, ensure_ascii=False)}

已知 flags（僅供參考）:
{json.dumps(list(flags.keys())[:50], ensure_ascii=False)}

你只能使用以下工具（tool）：
{sorted(self.allowed_tools)}

規則：
1) `modify_world_state` 的 flags key 必須以 `quest_` / `npc_met_` / `location_discovered_` / `item_acquired_` / `event_` / `achievement_` 開頭。
2) `update_character_state` 的 stats key 只能是：health, energy, intelligence, charisma, luck, experience, level（可用 relative=true 表示增量）。
3) 若不需要任何修改，回傳空陣列。

輸出格式：只輸出 JSON（不要多餘文字）
{{
  "reasoning": "一句話說明你為什麼要這些 tool_calls",
  "tool_calls": [
    {{"tool": "modify_world_state", "params": {{"flags": {{"quest_example": true}}, "reason": "..."}}}},
    {{"tool": "update_character_state", "params": {{"stats": {{"health": -5}}, "relative": true, "reason": "..."}}}}
  ]
}}
"""

            raw = self.llm_adapter.generate_text(prompt, max_tokens=350, temperature=0.2)
            payload = self._extract_json_object(raw)
            if not payload:
                return None

            tool_calls = payload.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                return None
            sanitized = self._sanitize_tool_calls(
                [
                    {**c, "agent": "llm_director"}
                    for c in tool_calls
                    if isinstance(c, dict)
                ]
            )
            if not sanitized:
                return None
            return {
                "agent": "llm_director",
                "reasoning": str(payload.get("reasoning") or "LLM director proposal"),
                "tool_calls": sanitized,
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM director skipped: %s", exc)
            return None

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
            # Multi-agent director
            if str(self.decision_mode).startswith("multi_agent"):
                profile = None
                try:
                    profile = self._load_agent_profile(session_id)
                    if profile is not None and not bool(getattr(profile, "enabled", True)):
                        return None
                except Exception:
                    profile = None

                max_tool_calls = self.max_tool_calls
                max_llm_calls = self.max_llm_calls
                allowed_tools: Set[str] = set(self.allowed_tools)
                enabled_agents: Optional[Set[str]] = None

                try:
                    if profile is not None:
                        max_tool_calls = int(getattr(profile, "max_tool_calls", max_tool_calls))
                        max_llm_calls = int(getattr(profile, "max_llm_calls", max_llm_calls))

                        configured_tools = getattr(profile, "allowed_tools", None) or []
                        if isinstance(configured_tools, list) and configured_tools:
                            configured_set = {str(t).strip() for t in configured_tools if str(t).strip()}
                            allowed_tools = allowed_tools.intersection(configured_set)

                        configured_agents = getattr(profile, "enabled_agents", None) or []
                        if isinstance(configured_agents, list) and configured_agents:
                            enabled_agents = {str(a).strip() for a in configured_agents if str(a).strip()}
                except Exception:
                    pass

                max_tool_calls = max(0, min(20, int(max_tool_calls)))
                max_llm_calls = max(0, min(10, int(max_llm_calls)))

                from .story_orchestrator import StoryOrchestrator

                orchestrator = StoryOrchestrator(
                    llm_adapter=self.llm_adapter,
                    allowed_tools=allowed_tools,
                    max_tool_calls=max_tool_calls,
                    max_llm_calls=max_llm_calls,
                )

                plan = orchestrator.plan_post_turn(
                    session_id=session_id,
                    player_input=player_input,
                    narrative_text=narrative_text,
                    context_memory=context_memory,
                    session_stats=session_stats,
                    enabled_agents=enabled_agents,
                )
                if not plan:
                    return None

                return AgentDecision(
                    decision_type=str(plan.get("decision_type") or "multi_agent_story_director"),
                    tool_calls=list(plan.get("tool_calls") or []),
                    reasoning=str(plan.get("reasoning") or "multi-agent decision"),
                    confidence=float(plan.get("confidence", 0.8) or 0.8),
                    contributors=list(plan.get("contributors") or []),
                )

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
                        "stats": {"health": -damage_amount},
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
            "contributors": getattr(decision, "contributors", []),
            "tool_results": [],
            "overall_success": True,
            "errors": []
        }

        for tool_call in decision.tool_calls:
            tool_name = tool_call["tool"]
            params = tool_call["params"]
            origin_agent = tool_call.get("agent")

            try:
                # Execute through safety wrapper
                result = await self.safety_wrapper.execute_tool(
                    tool_name,
                    session_id,
                    params
                )

                results["tool_results"].append({
                    "tool": tool_name,
                    **({"agent": origin_agent} if origin_agent else {}),
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
Player Stats: Health {session_stats.get('health', 0)} / Energy {session_stats.get('energy', 0)}

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
