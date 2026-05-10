"""
Story Orchestrator (Multi-Agent Director)

將多個專職子代理（sub-agents）的提案整合成同一個可執行的 tool_calls 計畫，
並提供 contributors 供前端顯示每個子代理的貢獻與推理。

注意：此 orchestrator 只負責「決策與編排」；實際執行仍由 StoryAgentLayer
透過 StorySafetyWrapper 來完成，避免繞過安全與 rollback。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class StoryOrchestrator:
    def __init__(
        self,
        llm_adapter: Any,
        allowed_tools: Set[str],
        max_tool_calls: int = 6,
        max_llm_calls: int = 1,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.allowed_tools = set(allowed_tools or set())
        self.max_tool_calls = int(max_tool_calls or 0)
        self.max_llm_calls = int(max_llm_calls or 0)

    # ---------------------------------------------------------------------
    # Utils
    # ---------------------------------------------------------------------
    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON object extraction from LLM output."""
        if not text or not str(text).strip():
            return None
        raw = str(text).strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                return json.loads(raw)
            except Exception:
                return None

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

    def _merge_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate tool calls to preserve budget when many sub-agents propose actions."""
        world_flags: Dict[str, bool] = {}
        world_agents: Set[str] = set()
        world_reasons: List[str] = []

        stats_abs: Dict[str, Any] = {}
        stats_rel: Dict[str, float] = {}
        stats_abs_agents: Set[str] = set()
        stats_rel_agents: Set[str] = set()
        stats_abs_reasons: List[str] = []
        stats_rel_reasons: List[str] = []

        rel_abs: Dict[str, int] = {}
        rel_rel: Dict[str, int] = {}
        rel_agents: Set[str] = set()
        rel_reasons: List[str] = []

        items: Dict[str, int] = {}
        item_agents: Set[str] = set()
        item_reasons: List[str] = []

        rag_queries: Dict[tuple[str, str], int] = {}
        rag_agents: Set[str] = set()
        rag_reasons: List[str] = []

        scene_image: Optional[Dict[str, Any]] = None
        scene_agents: Set[str] = set()
        scene_reasons: List[str] = []

        passthrough: List[Dict[str, Any]] = []

        def _note_reason(bucket: List[str], agent: Optional[str], reason: Optional[str]) -> None:
            if not reason:
                return
            reason_text = str(reason).strip()
            if not reason_text:
                return
            if agent:
                bucket.append(f"{agent}: {reason_text}")
            else:
                bucket.append(reason_text)

        for call in tool_calls or []:
            if not isinstance(call, dict):
                continue
            tool = call.get("tool")
            params = call.get("params") or {}
            if not tool or not isinstance(params, dict):
                continue
            agent = str(call.get("agent") or "").strip() or None
            reason = call.get("reason") or params.get("reason")

            if tool == "modify_world_state":
                flags = params.get("flags") or {}
                if isinstance(flags, dict):
                    for k, v in flags.items():
                        key = str(k or "").strip()
                        if not key:
                            continue
                        world_flags[key] = bool(v)
                if agent:
                    world_agents.add(agent)
                _note_reason(world_reasons, agent, reason)
                continue

            if tool == "update_character_state":
                stats = params.get("stats") or {}
                relative = bool(params.get("relative", False))
                if isinstance(stats, dict):
                    if relative:
                        for k, v in stats.items():
                            key = str(k or "").strip()
                            if not key:
                                continue
                            try:
                                delta = float(v)
                            except Exception:
                                continue
                            stats_rel[key] = stats_rel.get(key, 0.0) + delta
                        if agent:
                            stats_rel_agents.add(agent)
                        _note_reason(stats_rel_reasons, agent, reason)
                    else:
                        for k, v in stats.items():
                            key = str(k or "").strip()
                            if not key:
                                continue
                            stats_abs[key] = v
                        if agent:
                            stats_abs_agents.add(agent)
                        _note_reason(stats_abs_reasons, agent, reason)
                continue

            if tool == "add_inventory_item":
                item = params.get("item")
                if item is not None and str(item).strip():
                    item_key = str(item).strip()
                    try:
                        qty = int(params.get("quantity", 1))
                    except Exception:
                        qty = 1
                    qty = max(1, qty)
                    items[item_key] = items.get(item_key, 0) + qty
                    if agent:
                        item_agents.add(agent)
                    _note_reason(item_reasons, agent, reason)
                continue

            if tool == "update_relationship_state":
                rels = params.get("relationships") or {}
                relative = bool(params.get("relative", True))
                if isinstance(rels, dict):
                    if relative:
                        for k, v in rels.items():
                            key = str(k or "").strip()
                            if not key:
                                continue
                            try:
                                delta = int(v)
                            except Exception:
                                continue
                            rel_rel[key] = rel_rel.get(key, 0) + delta
                    else:
                        for k, v in rels.items():
                            key = str(k or "").strip()
                            if not key:
                                continue
                            try:
                                rel_abs[key] = int(v)
                            except Exception:
                                continue
                    if agent:
                        rel_agents.add(agent)
                    _note_reason(rel_reasons, agent, reason)
                continue

            if tool == "rag_search":
                query = str(params.get("query") or "").strip()
                world_id = str(params.get("world_id") or "").strip()
                if query:
                    try:
                        top_k = int(params.get("top_k", 3))
                    except Exception:
                        top_k = 3
                    key = (query, world_id)
                    rag_queries[key] = max(rag_queries.get(key, 0), max(1, top_k))
                    if agent:
                        rag_agents.add(agent)
                    _note_reason(rag_reasons, agent, reason)
                continue

            if tool == "generate_scene_image":
                if scene_image is None:
                    scene_image = call
                else:
                    # Prefer force=True if any sub-agent requested it.
                    try:
                        cur_force = bool((scene_image.get("params") or {}).get("force", False))
                        new_force = bool((params or {}).get("force", False))
                        if new_force and not cur_force:
                            scene_image = call
                    except Exception:
                        pass
                if agent:
                    scene_agents.add(agent)
                _note_reason(scene_reasons, agent, reason)
                continue

            passthrough.append(call)

        merged: List[Dict[str, Any]] = []

        if world_flags:
            merged.append(
                {
                    "tool": "modify_world_state",
                    "params": {
                        "flags": world_flags,
                        "reason": " | ".join(world_reasons[:3]) if world_reasons else "orchestrator merge",
                    },
                    "agent": "|".join(sorted(world_agents)) if world_agents else "orchestrator",
                }
            )

        if stats_abs:
            merged.append(
                {
                    "tool": "update_character_state",
                    "params": {
                        "stats": stats_abs,
                        "relative": False,
                        "reason": " | ".join(stats_abs_reasons[:3]) if stats_abs_reasons else "orchestrator merge",
                    },
                    "agent": "|".join(sorted(stats_abs_agents)) if stats_abs_agents else "orchestrator",
                }
            )

        if stats_rel:
            merged.append(
                {
                    "tool": "update_character_state",
                    "params": {
                        "stats": stats_rel,
                        "relative": True,
                        "reason": " | ".join(stats_rel_reasons[:3]) if stats_rel_reasons else "orchestrator merge",
                    },
                    "agent": "|".join(sorted(stats_rel_agents)) if stats_rel_agents else "orchestrator",
                }
            )

        if rel_abs:
            rel_abs_clamped = {k: max(-10, min(10, int(v))) for k, v in rel_abs.items()}
            merged.append(
                {
                    "tool": "update_relationship_state",
                    "params": {
                        "relationships": rel_abs_clamped,
                        "relative": False,
                        "reason": " | ".join(rel_reasons[:3]) if rel_reasons else "orchestrator merge",
                    },
                    "agent": "|".join(sorted(rel_agents)) if rel_agents else "orchestrator",
                }
            )

        if rel_rel:
            rel_rel_clamped = {
                k: max(-10, min(10, int(v)))
                for k, v in rel_rel.items()
            }
            merged.append(
                {
                    "tool": "update_relationship_state",
                    "params": {
                        "relationships": rel_rel_clamped,
                        "relative": True,
                        "reason": " | ".join(rel_reasons[:3]) if rel_reasons else "orchestrator merge",
                    },
                    "agent": "|".join(sorted(rel_agents)) if rel_agents else "orchestrator",
                }
            )

        for item_name, qty in sorted(items.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
            merged.append(
                {
                    "tool": "add_inventory_item",
                    "params": {
                        "item": item_name,
                        "quantity": min(99, int(qty)),
                        "reason": " | ".join(item_reasons[:3]) if item_reasons else "orchestrator merge",
                    },
                    "agent": "|".join(sorted(item_agents)) if item_agents else "orchestrator",
                }
            )

        for (query, world_id), top_k in list(rag_queries.items())[:2]:
            merged.append(
                {
                    "tool": "rag_search",
                    "params": {"query": query, "top_k": top_k, "world_id": world_id},
                    "agent": "|".join(sorted(rag_agents)) if rag_agents else "orchestrator",
                }
            )

        if scene_image is not None:
            merged.append(
                {
                    "tool": "generate_scene_image",
                    "params": dict(scene_image.get("params") or {}),
                    "agent": "|".join(sorted(scene_agents)) if scene_agents else (scene_image.get("agent") or "orchestrator"),
                }
            )

        merged.extend(passthrough)
        return merged

    # ---------------------------------------------------------------------
    # Heuristics: sub-agents
    # ---------------------------------------------------------------------
    def _extract_quest_from_narrative(self, narrative: str) -> Optional[str]:
        if "龍" in narrative or "dragon" in narrative.lower():
            return "quest_dragon_complete"
        if "森林" in narrative or "forest" in narrative.lower():
            return "quest_forest_complete"
        return None

    def _extract_damage_from_narrative(self, narrative: str) -> int:
        patterns = [
            r"受到\s*(\d+)\s*傷害",
            r"took\s*(\d+)\s*damage",
            r"(\d+)\s*點傷害",
            r"(\d+)\s*damage",
        ]
        for pattern in patterns:
            match = re.search(pattern, narrative, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except Exception:
                    return 0
        return 0

    def _extract_item_from_narrative(self, narrative: str) -> Optional[str]:
        if "劍" in narrative or "sword" in narrative.lower():
            return "sword"
        if "藥水" in narrative or "potion" in narrative.lower():
            return "health_potion"
        if "鑰匙" in narrative or "key" in narrative.lower():
            return "key"
        return None

    def _heuristic_subdecisions(
        self,
        *,
        session_id: str,
        player_input: str,
        narrative_text: str,
        context_memory: Any,
        session_stats: Dict[str, Any],
        inventory: List[str],
        world_id: str,
        flags: Dict[str, Any],
        story_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return a list of sub-decisions (heuristics) with tool_calls."""
        subs: List[Dict[str, Any]] = []

        # --- NPC tracker agent ---------------------------------------------
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

        # --- Character director agent (relationships) ----------------------
        rel_tool_calls: List[Dict[str, Any]] = []
        try:
            present = context_memory.get_characters_in_scene() if context_memory else []
            present_ids: List[str] = []
            for char in present:
                char_id = getattr(char, "character_id", None)
                role = getattr(getattr(char, "role", None), "value", None) or getattr(char, "role", None)
                if not char_id or char_id in {"player", "narrator"}:
                    continue
                if str(role) in {"npc", "companion", "antagonist"}:
                    present_ids.append(str(char_id))

            if present_ids:
                text = f"{player_input}\n{narrative_text}".lower()
                positive = [
                    "感謝",
                    "謝謝",
                    "微笑",
                    "友好",
                    "信任",
                    "欣賞",
                    "稱讚",
                    "幫助",
                    "救",
                    "贈送",
                ]
                negative = [
                    "憤怒",
                    "厭惡",
                    "威脅",
                    "辱罵",
                    "攻擊",
                    "討厭",
                    "背叛",
                    "欺騙",
                    "拒絕",
                    "嘲笑",
                ]
                score = 0
                score += sum(1 for w in positive if w.lower() in text)
                score -= sum(1 for w in negative if w.lower() in text)

                delta = 0
                if score >= 2:
                    delta = 2
                elif score == 1:
                    delta = 1
                elif score == -1:
                    delta = -1
                elif score <= -2:
                    delta = -2

                if delta != 0:
                    rel_tool_calls.append(
                        {
                            "tool": "update_relationship_state",
                            "params": {
                                "relationships": {cid: delta for cid in present_ids},
                                "relative": True,
                                "reason": "敘事/互動語氣推測關係變化",
                            },
                            "agent": "character_director",
                        }
                    )
        except Exception:
            pass
        if rel_tool_calls:
            subs.append(
                {
                    "agent": "character_director",
                    "reasoning": "根據互動語氣微調與在場角色的關係分數",
                    "tool_calls": rel_tool_calls,
                }
            )

        # --- Location tracker agent ---------------------------------------
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

        # --- Scene graph agent (scene node markers) -----------------------
        scene_calls: List[Dict[str, Any]] = []
        try:
            current_scene = context_memory.get_current_scene() if context_memory else None
            scene_id = getattr(current_scene, "scene_id", None) or getattr(current_scene, "id", None)
            if scene_id:
                raw = str(scene_id).strip()
                key = re.sub(r"\s+", "_", raw)
                key = re.sub(r"[^\w\-\u4e00-\u9fff]", "", key)
                if key:
                    flag_name = f"event_scene_{key}"
                    if not flags.get(flag_name):
                        scene_calls.append(
                            {
                                "tool": "modify_world_state",
                                "params": {"flags": {flag_name: True}, "reason": "場景節點標記（scene graph）"},
                                "agent": "scene_graph",
                            }
                        )
        except Exception:
            pass
        if scene_calls:
            subs.append(
                {
                    "agent": "scene_graph",
                    "reasoning": "標記已進入的場景節點（event_scene_*）",
                    "tool_calls": scene_calls,
                }
            )

        # --- Event extractor agent (damage/item/quest) ---------------------
        evt_calls: List[Dict[str, Any]] = []
        evt_reasons: List[str] = []

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

        item_name = self._extract_item_from_narrative(narrative_text)
        if item_name:
            evt_calls.append(
                {
                    "tool": "add_inventory_item",
                    "params": {"item": item_name, "quantity": 1, "reason": "敘事中獲得物品"},
                    "agent": "event_extractor",
                }
            )
            evt_calls.append(
                {
                    "tool": "modify_world_state",
                    "params": {"flags": {f"item_acquired_{item_name}": True}, "reason": "敘事中獲得物品"},
                    "agent": "event_extractor",
                }
            )
            evt_reasons.append(f"物品 {item_name}")

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

        # --- Combat tactician agent --------------------------------------
        combat_calls: List[Dict[str, Any]] = []
        try:
            text = f"{player_input}\n{narrative_text}".lower()
            combat_keywords = [
                "戰鬥",
                "交戰",
                "攻擊",
                "揮劍",
                "砍",
                "刺",
                "射擊",
                "格擋",
                "閃避",
                "fight",
                "battle",
                "attack",
                "combat",
                "block",
                "dodge",
                "slash",
                "stab",
                "shoot",
            ]
            win_keywords = [
                "勝利",
                "獲勝",
                "打敗",
                "擊敗",
                "斬殺",
                "victory",
                "won",
                "defeated",
                "slain",
            ]
            lose_keywords = [
                "失敗",
                "撤退",
                "逃跑",
                "落敗",
                "defeated",
                "lost",
                "retreat",
                "fled",
            ]

            flag_name: Optional[str] = None
            if any(k in text for k in win_keywords):
                flag_name = "event_combat_won"
            elif any(k in text for k in lose_keywords):
                flag_name = "event_combat_lost"
            elif any(k in text for k in combat_keywords):
                flag_name = "event_combat_active"

            if flag_name and not flags.get(flag_name):
                combat_calls.append(
                    {
                        "tool": "modify_world_state",
                        "params": {"flags": {flag_name: True}, "reason": "偵測到戰鬥狀態/結果"},
                        "agent": "combat_tactician",
                    }
                )
        except Exception:
            pass
        if combat_calls:
            subs.append(
                {
                    "agent": "combat_tactician",
                    "reasoning": "將戰鬥節奏轉成 event_combat_* flags（供任務/連貫性使用）",
                    "tool_calls": combat_calls,
                }
            )

        # --- Item economy agent ------------------------------------------
        economy_calls: List[Dict[str, Any]] = []
        try:
            text = f"{player_input}\n{narrative_text}".lower()
            trade_keywords = [
                "交易",
                "購買",
                "買",
                "賣",
                "商人",
                "店",
                "市場",
                "金幣",
                "錢",
                "trade",
                "buy",
                "sell",
                "merchant",
                "shop",
                "market",
                "gold",
                "coin",
            ]
            if any(k in text for k in trade_keywords):
                flag_name = "event_trade"
                if not flags.get(flag_name):
                    economy_calls.append(
                        {
                            "tool": "modify_world_state",
                            "params": {"flags": {flag_name: True}, "reason": "偵測到交易/經濟事件"},
                            "agent": "item_economy",
                        }
                    )
        except Exception:
            pass
        if economy_calls:
            subs.append(
                {
                    "agent": "item_economy",
                    "reasoning": "標記交易/經濟事件（event_trade）",
                    "tool_calls": economy_calls,
                }
            )

        # --- Quest master agent -------------------------------------------
        quest_calls: List[Dict[str, Any]] = []
        reward_calls: List[Dict[str, Any]] = []
        try:
            text = f"{player_input}\n{narrative_text}".lower()
            topics: List[str] = []
            if "dragon" in text or "龍" in narrative_text:
                topics.append("dragon")
            if "forest" in text or "森林" in narrative_text:
                topics.append("forest")
            if "cave" in text or "洞窟" in narrative_text or "洞穴" in narrative_text or "洞" in narrative_text:
                topics.append("cave")
            if not topics and ("quest" in text or "任務" in narrative_text):
                topics.append("generic")

            started_markers = ["接受", "接下", "開始", "出發", "委託", "任務開始", "accepted", "accept", "start quest"]
            complete_markers = ["完成", "達成", "成功", "擊敗", "解決", "結束", "complete", "completed"]
            failed_markers = ["失敗", "放棄", "撤退", "failed", "give up"]

            stage: Optional[str] = None
            if any(m.lower() in text for m in complete_markers):
                stage = "complete"
            elif any(m.lower() in text for m in failed_markers):
                stage = "failed"
            elif any(m.lower() in text for m in started_markers) or ("quest" in text or "任務" in narrative_text):
                stage = "started"

            if stage and topics:
                flags_to_set: Dict[str, bool] = {}
                for topic in topics:
                    flag_name = (
                        f"quest_{topic}_{stage}" if topic != "generic" else f"quest_generic_{stage}"
                    )
                    if not flags.get(flag_name):
                        flags_to_set[flag_name] = True

                if flags_to_set:
                    quest_calls.append(
                        {
                            "tool": "modify_world_state",
                            "params": {"flags": flags_to_set, "reason": "任務脈絡/關鍵字推測"},
                            "agent": "quest_master",
                        }
                    )

                if stage == "complete":
                    reward_map = {"dragon": 120, "forest": 60, "cave": 80, "generic": 40}
                    exp_delta = 0
                    for topic in topics:
                        flag_name = (
                            f"quest_{topic}_complete" if topic != "generic" else "quest_generic_complete"
                        )
                        if flags.get(flag_name):
                            continue
                        exp_delta += int(reward_map.get(topic, 40))
                    if exp_delta > 0:
                        reward_calls.append(
                            {
                                "tool": "update_character_state",
                                "params": {
                                    "stats": {"experience": exp_delta},
                                    "relative": True,
                                    "reason": "完成任務獎勵（經驗值）",
                                },
                                "agent": "reward_balancer",
                            }
                        )
        except Exception:
            pass
        if quest_calls:
            subs.append(
                {
                    "agent": "quest_master",
                    "reasoning": "補強任務開始/完成/失敗的旗標追蹤",
                    "tool_calls": quest_calls,
                }
            )
        if reward_calls:
            subs.append(
                {
                    "agent": "reward_balancer",
                    "reasoning": "在任務完成時給予合理獎勵（避免重複）",
                    "tool_calls": reward_calls,
                }
            )

        # --- Inventory curator agent --------------------------------------
        inv_flag_calls: List[Dict[str, Any]] = []
        try:
            if inventory:
                prefixes = ("item_acquired_",)
                to_set: Dict[str, bool] = {}
                seen: Set[str] = set()
                for item in inventory:
                    raw = str(item or "").strip()
                    if not raw:
                        continue
                    key = re.sub(r"\s+", "_", raw)
                    key = re.sub(r"[^\w\-\u4e00-\u9fff]", "", key)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    flag_name = f"{prefixes[0]}{key}"
                    if not flags.get(flag_name):
                        to_set[flag_name] = True
                    if len(to_set) >= 10:
                        break
                if to_set:
                    inv_flag_calls.append(
                        {
                            "tool": "modify_world_state",
                            "params": {"flags": to_set, "reason": "背包內容與 item_acquired_* flags 同步"},
                            "agent": "inventory_curator",
                        }
                    )
        except Exception:
            pass
        if inv_flag_calls:
            subs.append(
                {
                    "agent": "inventory_curator",
                    "reasoning": "確保背包物品對應的 item_acquired_* flags 存在",
                    "tool_calls": inv_flag_calls,
                }
            )

        # --- Continuity critic agent --------------------------------------
        continuity_calls: List[Dict[str, Any]] = []
        continuity_notes: List[str] = []
        try:
            prefixes = (
                "quest_",
                "npc_met_",
                "location_discovered_",
                "item_acquired_",
                "event_",
                "achievement_",
            )
            context_flags = getattr(context_memory, "world_flags", {}) if context_memory else {}
            truthy_session = {k for k, v in (flags or {}).items() if bool(v)}
            truthy_context = {k for k, v in (context_flags or {}).items() if bool(v)}
            missing_session = truthy_context - truthy_session
            missing_context = truthy_session - truthy_context
            to_sync = {
                k
                for k in (missing_session | missing_context)
                if any(str(k).startswith(p) for p in prefixes)
            }
            if to_sync:
                continuity_calls.append(
                    {
                        "tool": "modify_world_state",
                        "params": {"flags": {k: True for k in sorted(to_sync)}, "reason": "修正 session.flags / context.world_flags 不一致"},
                        "agent": "continuity_critic",
                    }
                )
                continuity_notes.append(f"同步 flags {len(to_sync)}")

            bounds = {
                "health": (0, 9999),
                "energy": (0, 9999),
                "intelligence": (0, 999),
                "charisma": (0, 999),
                "luck": (0, 999),
                "experience": (0, 999999),
                "level": (1, 100),
            }
            stat_fix: Dict[str, Any] = {}
            for stat, (mn, mx) in bounds.items():
                if stat not in session_stats:
                    continue
                try:
                    val = float(session_stats.get(stat) or 0)
                except Exception:
                    continue
                if val < mn:
                    stat_fix[stat] = mn
                elif val > mx:
                    stat_fix[stat] = mx
            if stat_fix:
                continuity_calls.append(
                    {
                        "tool": "update_character_state",
                        "params": {"stats": stat_fix, "relative": False, "reason": "修正越界的角色狀態"},
                        "agent": "continuity_critic",
                    }
                )
                continuity_notes.append("修正越界狀態")
        except Exception:
            pass
        subs.append(
            {
                "agent": "continuity_critic",
                "reasoning": " / ".join(continuity_notes) if continuity_notes else "未檢測到需修正的矛盾或越界狀態",
                "tool_calls": continuity_calls,
            }
        )

        # --- Pacing director agent (analysis only) -------------------------
        try:
            text = f"{player_input}\n{narrative_text}".lower()
            combat_words = ["戰鬥", "攻擊", "防禦", "追逐", "爆炸", "blood", "boss", "combat"]
            dialogue_words = ["對話", "說", "問", "回答", "交談", "協商", "聊天"]
            explore_words = ["探索", "查看", "搜尋", "發現", "進入", "到達", "來到", "discover", "enter"]
            combat_score = sum(1 for w in combat_words if w.lower() in text)
            dialogue_score = sum(1 for w in dialogue_words if w.lower() in text)
            explore_score = sum(1 for w in explore_words if w.lower() in text)
            focus = "general"
            if combat_score >= max(dialogue_score, explore_score) and combat_score > 0:
                focus = "combat"
            elif dialogue_score >= max(combat_score, explore_score) and dialogue_score > 0:
                focus = "dialogue"
            elif explore_score >= max(combat_score, dialogue_score) and explore_score > 0:
                focus = "exploration"

            subs.append(
                {
                    "agent": "pacing_director",
                    "reasoning": f"節奏焦點判定: {focus}（combat={combat_score}, dialogue={dialogue_score}, explore={explore_score}）",
                    "tool_calls": [],
                }
            )
        except Exception:
            subs.append(
                {
                    "agent": "pacing_director",
                    "reasoning": "節奏焦點判定：略過（解析失敗）",
                    "tool_calls": [],
                }
            )

        # --- Lore retriever agent (optional) -------------------------------
        lore_calls: List[Dict[str, Any]] = []
        try:
            rag_auto = bool(story_context.get("rag_auto"))
            rag_available = story_context.get("rag_available")
            rag_available_ok = True if rag_available is None else bool(rag_available)
            rag_enabled = bool(story_context.get("enrich_with_rag")) or (rag_auto and rag_available_ok)
            if rag_enabled:
                query = ((story_context.get("rag_query") or "").strip() or player_input.strip())
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
        *,
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
3) `update_relationship_state` 的 relationships value 必須在 -10..10（relative=true 表示增量，relative=false 表示絕對值；可用 key="any" 代表對在場 NPC 套用）。
4) 若不需要任何修改，回傳空陣列。

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

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def plan_post_turn(
        self,
        *,
        session_id: str,
        player_input: str,
        narrative_text: str,
        context_memory: Any,
        session_stats: Dict[str, Any],
        enabled_agents: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return a multi-agent plan for the current turn (post narrative).

        Output shape aligns with StoryAgentLayer.AgentDecision fields.
        """
        from core.story.engine import get_story_engine

        engine = get_story_engine()
        session = engine.get_session(session_id)
        world_id = getattr(session, "world_id", "default")
        flags = dict(getattr(session.current_state, "flags", {}) or {})
        inventory = list(getattr(session, "inventory", []) or [])
        story_context = dict(getattr(session.current_state, "story_context", {}) or {})

        # worldpack summary (best-effort)
        worldpack_summary = ""
        try:
            from core.worldpacks import get_worldpack_manager

            wpm = get_worldpack_manager()
            wp = wpm.get_worldpack(world_id)
            if wp:
                worldpack_summary = (
                    f"- name: {wp.name}\n"
                    f"- description: {wp.description}\n"
                    f"- setting: {wp.setting}\n"
                    f"- difficulty: {wp.difficulty}\n"
                    f"- characters: {len(wp.characters)}\n"
                    f"- player_templates: {len(wp.player_templates)}\n"
                    f"- default_loras: {len(wp.visual.default_loras)}"
                )
        except Exception:
            worldpack_summary = ""

        contributors: List[Dict[str, Any]] = []
        merged_calls: List[Dict[str, Any]] = []
        enabled_set = {str(x).strip() for x in (enabled_agents or set()) if str(x).strip()}

        # Heuristic sub-agents
        subs = self._heuristic_subdecisions(
            session_id=session_id,
            player_input=player_input,
            narrative_text=narrative_text,
            context_memory=context_memory,
            session_stats=session_stats,
            inventory=inventory,
            world_id=world_id,
            flags=flags,
            story_context=story_context,
        )
        for sub in subs:
            agent_name = str(sub.get("agent") or "").strip()
            if enabled_set and agent_name not in enabled_set:
                continue
            contributors.append(
                {
                    "agent": sub.get("agent"),
                    "reasoning": sub.get("reasoning"),
                    "tool_calls": sub.get("tool_calls", []),
                }
            )
            merged_calls.extend(sub.get("tool_calls", []))

        # Optional single LLM director (capped)
        llm_sub = None
        if not enabled_set or "llm_director" in enabled_set:
            llm_sub = self._llm_director_subdecision(
                player_input=player_input,
                narrative_text=narrative_text,
                context_memory=context_memory,
                session_stats=session_stats,
                inventory=inventory,
                flags=flags,
                world_id=world_id,
                worldpack_summary=worldpack_summary,
            )
        if llm_sub:
            contributors.append(
                {
                    "agent": llm_sub.get("agent"),
                    "reasoning": llm_sub.get("reasoning"),
                    "tool_calls": llm_sub.get("tool_calls", []),
                }
            )
            merged_calls.extend(llm_sub.get("tool_calls", []))

        raw_count = len(merged_calls)
        merged_calls = self._merge_tool_calls(merged_calls)
        merged_calls = self._sanitize_tool_calls(merged_calls)

        if not merged_calls:
            return None

        contributors.append(
            {
                "agent": "orchestrator",
                "reasoning": f"合併/去重 tool_calls：{raw_count} -> {len(merged_calls)}（max={self.max_tool_calls}）",
                "tool_calls": merged_calls,
            }
        )

        decision_reasoning = " | ".join(
            [f"{c.get('agent')}: {c.get('reasoning')}" for c in contributors if c.get("agent")]
        )

        return {
            "decision_type": "multi_agent_story_director",
            "tool_calls": merged_calls,
            "reasoning": decision_reasoning or "multi-agent decision",
            "confidence": 0.8,
            "contributors": contributors,
        }
