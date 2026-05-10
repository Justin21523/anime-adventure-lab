"""
World Studio Orchestrator (Multi-Agent)

目標：
- 讓「世界工作室」可以用多個子代理，根據 RAG/現有 worldpack/可用 LoRA，產生可套用的 worldpack patch。
- 在沒有可用 LLM（或 mock 模式）時，仍會提供可用的 fallback patch，避免 UI 完全無法運作。
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorldStudioOrchestrator:
    def __init__(self, llm_adapter: Any = None, max_llm_calls: int = 5) -> None:
        if llm_adapter is None:
            try:
                from core.llm.adapter import get_llm_adapter

                llm_adapter = get_llm_adapter()
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM adapter unavailable for WorldStudioOrchestrator: %s", exc)
                llm_adapter = None

        self.llm_adapter = llm_adapter
        self.max_llm_calls = int(max_llm_calls or 0)
        self._llm_calls = 0

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

    def _llm_json(
        self, *, prompt: str, max_tokens: int = 700, temperature: float = 0.3
    ) -> Optional[Dict[str, Any]]:
        if not self.llm_adapter or self.max_llm_calls <= 0:
            return None
        if self._llm_calls >= self.max_llm_calls:
            return None
        self._llm_calls += 1
        try:
            raw = self.llm_adapter.generate_text(
                prompt, max_tokens=max_tokens, temperature=temperature
            )
            return self._extract_json_object(raw)
        except Exception as exc:  # noqa: BLE001
            logger.debug("WorldStudioOrchestrator LLM call skipped: %s", exc)
            return None

    def _merge_patch(self, base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        """Shallow merge for top-level keys + dict merge for common nested keys."""
        out = dict(base or {})
        inc = dict(incoming or {})

        for key in ["name", "description", "setting", "difficulty"]:
            value = inc.get(key)
            if isinstance(value, str) and value.strip():
                out[key] = value.strip()

        if isinstance(inc.get("world_flags"), dict):
            out.setdefault("world_flags", {})
            if isinstance(out.get("world_flags"), dict):
                for k, v in inc["world_flags"].items():
                    kk = str(k or "").strip()
                    if kk:
                        out["world_flags"][kk] = bool(v)

        if isinstance(inc.get("visual"), dict):
            out.setdefault("visual", {})
            if isinstance(out.get("visual"), dict):
                for k, v in inc["visual"].items():
                    if k in {"prompt_prefix", "negative_prompt"} and isinstance(v, str):
                        out["visual"][k] = v
                    elif k == "base_model":
                        if v is None or (isinstance(v, str) and v.strip()):
                            out["visual"][k] = v
                    elif k == "default_loras" and isinstance(v, list):
                        out["visual"][k] = v

        for key in ["characters", "player_templates"]:
            if isinstance(inc.get(key), list):
                out.setdefault(key, [])
                if isinstance(out.get(key), list):
                    out[key] = out[key] + inc[key]

        return out

    def _fallback_patch(
        self,
        *,
        world_id: str,
        instruction: str,
        include_visual: bool,
        available_loras: List[Dict[str, Any]],
        max_new_characters: int,
        max_new_player_templates: int,
    ) -> Dict[str, Any]:
        now = int(time.time())
        patch: Dict[str, Any] = {}

        # Minimal NPC
        characters: List[Dict[str, Any]] = []
        if max_new_characters > 0:
            characters.append(
                {
                    "character_id": f"npc_{now}",
                    "name": "新NPC",
                    "role": "npc",
                    "personality_traits": ["好奇", "務實", "友善"],
                    "speaking_style": "簡潔、具引導性",
                    "background_story": f"由世界工作室（fallback）根據指令生成：{instruction[:80]}",
                    "motivations": ["協助玩家", "推進劇情"],
                    "relationships": {"player": "unknown"},
                    "persona_prompt": f"你是世界({world_id})中的 NPC，請以一致的人格與世界觀與玩家互動。",
                    "content_restrictions": [],
                    "start_in_opening": False,
                }
            )
        if characters:
            patch["characters"] = characters[: max_new_characters]

        # Minimal player template
        templates: List[Dict[str, Any]] = []
        if max_new_player_templates > 0:
            templates.append(
                {
                    "template_id": f"template_{now}",
                    "name": "自訂主角風格",
                    "description": "由世界工作室（fallback）生成的玩家風格模板",
                    "personality_traits": ["冷靜", "善良", "果斷"],
                    "speaking_style": "直接但尊重他人",
                    "background_story": "一位帶著目的踏上旅程的旅人。",
                    "motivations": ["探索", "解謎", "守護同伴"],
                    "persona_prompt": f"你是玩家主角；請依照世界({world_id})的規則與氛圍做決策。",
                }
            )
        if templates:
            patch["player_templates"] = templates[: max_new_player_templates]

        if include_visual:
            visual: Dict[str, Any] = {
                "prompt_prefix": "anime style, cinematic lighting, high detail",
                "negative_prompt": "lowres, blurry, bad anatomy, extra fingers",
            }
            if available_loras:
                first = available_loras[0]
                lora_id = first.get("lora_id") or first.get("id") or first.get("name")
                if lora_id:
                    visual["default_loras"] = [{"lora_id": str(lora_id), "weight": 0.8}]
            patch["visual"] = visual

        patch.setdefault("world_flags", {})
        if isinstance(patch.get("world_flags"), dict):
            patch["world_flags"]["worldstudio_fallback_used"] = True

        return patch

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def suggest_worldpack_patch(
        self,
        *,
        world_id: str,
        instruction: str,
        worldpack_summary: str,
        rag_snippets: List[Dict[str, Any]],
        available_loras: List[Dict[str, Any]],
        include_visual: bool = True,
        max_new_characters: int = 3,
        max_new_player_templates: int = 1,
    ) -> Dict[str, Any]:
        """
        Return:
        {
          "patch": {...},
          "contributors": [{"agent": "...", "reasoning": "...", "patch": {...}}],
          "errors": [...]
        }
        """
        self._llm_calls = 0
        instruction = str(instruction or "").strip()
        if not instruction:
            return {
                "patch": {},
                "contributors": [],
                "errors": ["instruction cannot be empty"],
            }

        rag_text = "\n".join(
            [
                f"- {str(s.get('content') or '')[:400]}"
                for s in (rag_snippets or [])[:10]
                if isinstance(s, dict) and str(s.get("content") or "").strip()
            ]
        )
        lora_catalog = "\n".join(
            [
                f"- {l.get('lora_id') or l.get('id') or l.get('name')} | tags={l.get('tags', [])}"
                for l in (available_loras or [])[:30]
                if isinstance(l, dict)
            ]
        )

        contributors: List[Dict[str, Any]] = []
        errors: List[str] = []
        patch: Dict[str, Any] = {}

        # 1) world_designer -------------------------------------------------
        world_prompt = f"""你是一個子代理 world_designer，負責提出 WorldPack 的世界設定 patch（不需要全量 worldpack）。

world_id: {world_id}

現有 worldpack 摘要（僅供參考）：
{worldpack_summary}

RAG 片段（世界知識庫）：
{rag_text or "(無)"}

使用者指令：
{instruction}

輸出格式：只輸出 JSON（不要多餘文字）
{{
  "reasoning": "一句話說明你的設計重點",
  "patch": {{
    "name": "可選",
    "description": "可選",
    "setting": "可選",
    "difficulty": "可選",
    "world_flags": {{"flag_key": true}}
  }}
}}
"""
        payload = self._llm_json(prompt=world_prompt, max_tokens=550, temperature=0.2)
        if payload and isinstance(payload.get("patch"), dict):
            sub_patch = payload["patch"]
            patch = self._merge_patch(patch, sub_patch)
            contributors.append(
                {
                    "agent": "world_designer",
                    "reasoning": str(payload.get("reasoning") or ""),
                    "patch": sub_patch,
                }
            )

        # 2) npc_designer ---------------------------------------------------
        npc_prompt = f"""你是一個子代理 npc_designer，負責提出 NPC/角色模板（WorldPack.characters）。

world_id: {world_id}
worldpack 摘要：
{worldpack_summary}

RAG 片段：
{rag_text or "(無)"}

指令：
{instruction}

限制：
- 最多新增 {max_new_characters} 個角色。
- 每個角色必須包含 character_id/name/role/personality_traits/speaking_style/background_story/motivations/relationships/persona_prompt/content_restrictions/start_in_opening。
- role 只能是 npc | companion | antagonist。

輸出格式：只輸出 JSON（不要多餘文字）
{{
  "reasoning": "一句話說明你新增這些角色的理由",
  "patch": {{
    "characters": [
      {{
        "character_id": "unique_id",
        "name": "角色名",
        "role": "npc",
        "personality_traits": ["..."],
        "speaking_style": "...",
        "background_story": "...",
        "motivations": ["..."],
        "relationships": {{"player": "..." }},
        "persona_prompt": "...",
        "content_restrictions": [],
        "start_in_opening": false
      }}
    ]
  }}
}}
"""
        payload = self._llm_json(prompt=npc_prompt, max_tokens=750, temperature=0.35)
        if payload and isinstance(payload.get("patch"), dict):
            sub_patch = payload["patch"]
            patch = self._merge_patch(patch, sub_patch)
            contributors.append(
                {
                    "agent": "npc_designer",
                    "reasoning": str(payload.get("reasoning") or ""),
                    "patch": sub_patch,
                }
            )

        # 3) player_template_designer --------------------------------------
        player_prompt = f"""你是一個子代理 player_template_designer，負責提出玩家角色風格模板（WorldPack.player_templates）。

world_id: {world_id}
worldpack 摘要：
{worldpack_summary}

指令：
{instruction}

限制：
- 最多新增 {max_new_player_templates} 個玩家模板。
- 每個模板必須包含 template_id/name/description/personality_traits/speaking_style/background_story/motivations/persona_prompt。

輸出格式：只輸出 JSON（不要多餘文字）
{{
  "reasoning": "一句話說明這些模板如何符合世界觀與玩法",
  "patch": {{
    "player_templates": [
      {{
        "template_id": "unique_id",
        "name": "模板名",
        "description": "一句話描述",
        "personality_traits": ["..."],
        "speaking_style": "...",
        "background_story": "...",
        "motivations": ["..."],
        "persona_prompt": "可直接注入 story 的 persona prompt"
      }}
    ]
  }}
}}
"""
        payload = self._llm_json(prompt=player_prompt, max_tokens=650, temperature=0.35)
        if payload and isinstance(payload.get("patch"), dict):
            sub_patch = payload["patch"]
            patch = self._merge_patch(patch, sub_patch)
            contributors.append(
                {
                    "agent": "player_template_designer",
                    "reasoning": str(payload.get("reasoning") or ""),
                    "patch": sub_patch,
                }
            )

        # 4) visual_director (optional) ------------------------------------
        if include_visual:
            visual_prompt = f"""你是一個子代理 visual_director，負責提出世界視覺語言設定（WorldPack.visual），用於場景生成風格一致性。

world_id: {world_id}
worldpack 摘要：
{worldpack_summary}

可用 LoRA 清單（id + tags）：
{lora_catalog or "(無)"}

指令：
{instruction}

限制：
- default_loras 最多 5 個
- weight 介於 0.0 ~ 2.0

輸出格式：只輸出 JSON（不要多餘文字）
{{
  "reasoning": "一句話說明你選的視覺方向",
  "patch": {{
    "visual": {{
      "prompt_prefix": "...",
      "negative_prompt": "...",
      "base_model": null,
      "default_loras": [{{"lora_id": "xxx", "weight": 0.8}}]
    }}
  }}
}}
"""
            payload = self._llm_json(prompt=visual_prompt, max_tokens=550, temperature=0.3)
            if payload and isinstance(payload.get("patch"), dict):
                sub_patch = payload["patch"]
                patch = self._merge_patch(patch, sub_patch)
                contributors.append(
                    {
                        "agent": "visual_director",
                        "reasoning": str(payload.get("reasoning") or ""),
                        "patch": sub_patch,
                    }
                )

        # 5) Heuristic lora_recommender (no LLM, ensures something usable) --
        if include_visual and isinstance(patch.get("visual"), dict):
            visual = patch.get("visual") or {}
            if isinstance(visual, dict) and not visual.get("default_loras") and available_loras:
                first = available_loras[0]
                lora_id = first.get("lora_id") or first.get("id") or first.get("name")
                if lora_id:
                    visual["default_loras"] = [{"lora_id": str(lora_id), "weight": 0.8}]
                    patch["visual"] = visual
                    contributors.append(
                        {
                            "agent": "lora_recommender",
                            "reasoning": "未提供 default_loras 時，以 heuristic 選用第一個可用 LoRA 作為示範",
                            "patch": {"visual": {"default_loras": visual["default_loras"]}},
                        }
                    )

        # Fallback if nothing produced -------------------------------------
        if not patch:
            fallback = self._fallback_patch(
                world_id=world_id,
                instruction=instruction,
                include_visual=include_visual,
                available_loras=available_loras,
                max_new_characters=max_new_characters,
                max_new_player_templates=max_new_player_templates,
            )
            patch = fallback
            contributors.append(
                {
                    "agent": "fallback_generator",
                    "reasoning": "LLM 無可用輸出或解析失敗，因此提供可套用的最小可用 patch",
                    "patch": fallback,
                }
            )
            errors.append("LLM output not available or invalid JSON; fallback patch used")

        # Cap list sizes
        if isinstance(patch.get("characters"), list):
            patch["characters"] = patch["characters"][: max_new_characters]
        if isinstance(patch.get("player_templates"), list):
            patch["player_templates"] = patch["player_templates"][: max_new_player_templates]
        if include_visual and isinstance(patch.get("visual"), dict):
            v = patch["visual"]
            if isinstance(v.get("default_loras"), list):
                v["default_loras"] = v["default_loras"][:5]
                patch["visual"] = v

        # Final orchestrator entry
        contributors.append(
            {
                "agent": "orchestrator",
                "reasoning": f"完成 patch 聚合（llm_calls={self._llm_calls}/{self.max_llm_calls}）",
                "patch": patch,
            }
        )

        return {"patch": patch, "contributors": contributors, "errors": errors}

