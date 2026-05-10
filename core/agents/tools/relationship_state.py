"""
Relationship State Modification Tool for Story Agent

在 enhanced mode 下，關係分數存放於 StoryContextMemory.player_relationships（範圍 -10..10）。
此工具用於讓多代理（CharacterDirector 等）能安全地調整玩家與 NPC/角色的關係。
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _clamp_relationship(value: int) -> int:
    return max(-10, min(10, int(value)))


async def update_relationship_state(session_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update relationship scores in context memory (enhanced mode).

    Args:
        session_id: Story session ID
        params: Dictionary with:
            - relationships: Dict[str, int] - mapping character_id -> delta or absolute score
            - relative: bool (optional) - True to add deltas; False to set absolute values
            - reason: str (optional) - Reason for modification

    Notes:
        - Special key "any" will apply the same change to all non-player characters
          present in the current scene (best-effort).

    Returns:
        Dictionary with modified relationship scores
    """
    try:
        from core.story.engine import get_story_engine

        engine = get_story_engine()
        session = engine.get_session(session_id)

        relationships = params.get("relationships", {}) or {}
        relative = bool(params.get("relative", True))
        reason = params.get("reason", "Agent relationship update")

        if not isinstance(relationships, dict) or not relationships:
            return {"success": False, "error": "No relationships provided"}

        context_memory = None
        if getattr(engine, "enhanced_mode", False) and hasattr(engine, "context_memories"):
            context_memory = engine.context_memories.get(session_id)  # type: ignore[attr-defined]

        if context_memory is None:
            return {"success": False, "error": "Enhanced context memory not available"}

        modified: Dict[str, Any] = {}

        def _apply_change(character_id: str, change: int) -> None:
            old = int(context_memory.player_relationships.get(character_id, 0) or 0)
            new = old + change if relative else change
            new = _clamp_relationship(new)
            context_memory.player_relationships[character_id] = new
            modified[character_id] = {"old": old, "new": new, "change": new - old}

        for raw_id, raw_value in relationships.items():
            char_id = str(raw_id or "").strip()
            try:
                change_int = int(raw_value)
            except Exception:
                continue

            if not char_id:
                continue

            if char_id == "any":
                try:
                    present = context_memory.get_characters_in_scene() if context_memory else []
                    for c in present:
                        cid = getattr(c, "character_id", None)
                        role = getattr(getattr(c, "role", None), "value", None) or getattr(c, "role", None)
                        if not cid or cid in {"player", "narrator"}:
                            continue
                        if str(role) in {"npc", "companion", "antagonist"}:
                            _apply_change(str(cid), change_int)
                except Exception:  # noqa: BLE001
                    pass
                continue

            _apply_change(char_id, change_int)

        if not modified:
            return {"success": False, "error": "No valid relationship changes applied"}

        # Save session (engine will persist context memory in enhanced mode)
        if hasattr(engine, "save_session"):
            engine.save_session(session)
        else:
            engine._save_session(session)  # type: ignore[attr-defined]

        logger.info(
            "Modified %d relationship scores for session %s | Reason: %s",
            len(modified),
            session_id,
            reason,
        )

        return {"success": True, "modified_relationships": modified, "reason": reason}

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to update relationship state: %s", exc)
        return {"success": False, "error": str(exc)}


TOOL_METADATA = {
    "name": "update_relationship_state",
    "description": "Update relationship scores in enhanced StoryContextMemory (player_relationships)",
    "parameters": {
        "relationships": {
            "type": "dict",
            "description": "Mapping character_id -> delta/score (use relative=true for delta)",
            "required": True,
        },
        "relative": {
            "type": "boolean",
            "description": "If True, treat values as deltas; if False, set absolute score",
            "required": False,
        },
        "reason": {
            "type": "string",
            "description": "Reason for relationship change",
            "required": False,
        },
    },
    "examples": [
        {"relationships": {"merchant": 1}, "relative": True, "reason": "友好互動"},
        {"relationships": {"antagonist": -2}, "relative": True, "reason": "衝突/威脅"},
        {"relationships": {"any": 1}, "relative": True, "reason": "與在場 NPC 建立好感"},
    ],
}

