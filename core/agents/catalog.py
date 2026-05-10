"""
Agent Catalog (Single Source of Truth)

提供前端 UI 與後端一致的：
- Story orchestrator 子代理列表
- Story 允許工具列表
- 預設 agent_profile

目的：避免前端各處 hardcode（WorldStudio / Story AgentProfile 等），之後加新 agent 只改這裡。
"""

from __future__ import annotations

from typing import Any, Dict, List

from schemas.world import WorldAgentProfile


STORY_SUB_AGENTS: List[Dict[str, str]] = [
    {"id": "npc_tracker", "name": "NPC 追蹤", "description": "標記 npc_met_*（遇見/出現的角色）"},
    {"id": "character_director", "name": "關係導演", "description": "依互動語氣推測關係值變化"},
    {"id": "location_tracker", "name": "地點追蹤", "description": "標記 location_discovered_*（已發現地點）"},
    {"id": "scene_graph", "name": "場景圖", "description": "標記 event_scene_*（已進入場景節點）"},
    {"id": "event_extractor", "name": "事件抽取", "description": "抽取傷害/物品/任務進度等事件"},
    {"id": "combat_tactician", "name": "戰鬥戰術", "description": "標記 event_combat_*（戰鬥狀態/結果）"},
    {"id": "item_economy", "name": "經濟/交易", "description": "標記 event_trade（交易/經濟事件）"},
    {"id": "quest_master", "name": "任務管理", "description": "補強 quest_* flags（開始/完成/失敗）"},
    {"id": "reward_balancer", "name": "獎勵平衡", "description": "任務完成時給合理獎勵（例如經驗值）"},
    {"id": "inventory_curator", "name": "背包策展", "description": "背包物品與 item_acquired_* flags 對齊"},
    {"id": "continuity_critic", "name": "連貫性審稿", "description": "修正越界狀態與矛盾（例如血量範圍）"},
    {"id": "pacing_director", "name": "節奏判定", "description": "判斷戰鬥/對話/探索焦點（分析用）"},
    {"id": "lore_retriever", "name": "知識檢索", "description": "RAG 啟用時自動 rag_search（依 world_id 分域）"},
    {"id": "llm_director", "name": "LLM Director", "description": "可選：用 LLM 產生更細緻的 tool_calls（吃 token）"},
]


STORY_ALLOWED_TOOLS: List[Dict[str, str]] = [
    {"id": "modify_world_state", "description": "更新世界 flags（quest_/npc_met_/location_discovered_/...）"},
    {"id": "update_character_state", "description": "更新角色數值（health/energy/exp/level...）"},
    {"id": "add_inventory_item", "description": "新增背包物品"},
    {"id": "update_relationship_state", "description": "更新與 NPC 的關係值"},
    {"id": "rag_search", "description": "查詢 world_id 知識庫（RAG）"},
    {"id": "generate_scene_image", "description": "生成場景圖（若有接好 T2I）"},
]


DEFAULT_STORY_AGENT_PROFILE = WorldAgentProfile(
    enabled=True,
    enabled_agents=[],
    max_tool_calls=6,
    max_llm_calls=1,
    allowed_tools=[],
)


def get_story_agent_catalog() -> Dict[str, Any]:
    return {
        "sub_agents": STORY_SUB_AGENTS,
        "allowed_tools": STORY_ALLOWED_TOOLS,
        "default_agent_profile": DEFAULT_STORY_AGENT_PROFILE,
    }

