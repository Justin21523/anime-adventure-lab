from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class StoryTurnContext:
    session_id: str
    player_name: str
    world: dict[str, Any]
    state: dict[str, Any]
    player_input: str
    choice_id: str | None = None
    citations: list[dict[str, Any]] | None = None


class StoryTurnProcessor(Protocol):
    def process(self, context: StoryTurnContext) -> dict[str, Any]: ...


class DeterministicStoryTurnProcessor:
    """Stable processor for tests and the isolated public demo profile."""

    def process(self, context: StoryTurnContext) -> dict[str, Any]:
        world_name = str(
            context.world.get("name") or context.world.get("world_id") or "世界"
        )
        narrative = (
            f"{context.player_name} 在{world_name}採取了行動：{context.player_input}。"
            "遠方的線索逐漸清晰，下一個選擇將改變故事走向。"
        )
        citations = list(context.citations or [])
        if citations:
            narrative += (
                f" 根據《{citations[0]['filename']}》，{citations[0]['excerpt'][:90]}"
            )
        return {
            "narrative": narrative,
            "choices": [
                {"id": "investigate", "text": "追查線索"},
                {"id": "observe", "text": "先觀察四周"},
                {"id": "retreat", "text": "暫時撤退"},
            ],
            "citations": citations,
            "state_delta": {"last_action": context.player_input},
            "trace": {"processor": "deterministic", "mock": True},
            "world_patch": {
                "discoveries": {
                    "latest_story_discovery": f"{context.player_name}發現：{context.player_input}"
                }
            },
            "proposal_reasoning": "將本回合確認的發現交由創作者審核後寫入世界設定。",
        }


class LLMStoryTurnProcessor:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def process(self, context: StoryTurnContext) -> dict[str, Any]:
        prompt = {
            "task": "Continue an interactive anime adventure story",
            "response_format": {
                "narrative": "string",
                "choices": [{"id": "string", "text": "string"}],
                "state_delta": {},
            },
            "world": context.world,
            "player_name": context.player_name,
            "state": context.state,
            "player_input": context.player_input,
            "choice_id": context.choice_id,
            "retrieved_lore": list(context.citations or []),
        }
        response = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "Return only valid JSON. Keep the narrative concise and provide 2-4 choices.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            max_length=900,
            temperature=0.7,
        )
        raw = str(getattr(response, "content", response) or "").strip()
        parsed = self._parse_json(raw)
        narrative = str(
            parsed.get("narrative") or parsed.get("narration") or raw
        ).strip()
        choices = (
            parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
        )
        return {
            "narrative": narrative,
            "choices": choices,
            "citations": list(context.citations or []),
            "state_delta": (
                parsed.get("state_delta")
                if isinstance(parsed.get("state_delta"), dict)
                else {}
            ),
            "trace": {
                "processor": "llm",
                "model": str(getattr(response, "model_name", "unknown")),
                "usage": getattr(response, "usage", {}),
            },
            "world_patch": (
                parsed.get("world_patch")
                if isinstance(parsed.get("world_patch"), dict)
                else {}
            ),
            "proposal_reasoning": str(parsed.get("proposal_reasoning") or "")[:4000],
        }

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0]
        try:
            value = json.loads(cleaned)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
