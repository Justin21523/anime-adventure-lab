"""
World State Modification Tool for Story Agent

Allows Agent to modify world flags (quests, NPCs, events).
All modifications go through safety wrapper.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

_TRUE_STRINGS = {"true", "1", "yes", "y", "是", "對", "開", "啟用", "啟動"}
_FALSE_STRINGS = {"false", "0", "no", "n", "否", "不", "關", "停用", "關閉"}


def _coerce_flag_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    return bool(value)


async def modify_world_state(session_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modify world state flags

    Args:
        session_id: Story session ID
        params: Dictionary with:
            - flags: Dict[str, Any] - Flags to set/modify
            - reason: str (optional) - Reason for modification

    Returns:
        Dictionary with modified flags and their values

    Example:
        params = {
            "flags": {
                "quest_forest_started": True,
                "npc_met_elder": True,
                "location_discovered_cave": True
            },
            "reason": "Player started forest quest and met elder"
        }
    """
    try:
        from core.story.engine import get_story_engine

        engine = get_story_engine()
        session = engine.get_session(session_id)

        flags = params.get("flags", {})
        reason = params.get("reason", "Agent modification")

        if not flags:
            return {
                "success": False,
                "error": "No flags provided"
            }

        context_memory = None
        try:
            if getattr(engine, "enhanced_mode", False) and hasattr(engine, "context_memories"):
                context_memory = engine.context_memories.get(session_id)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            context_memory = None

        # Apply flag changes
        modified = {}
        for flag_name, flag_value in flags.items():
            coerced_value = _coerce_flag_value(flag_value)
            old_value = session.current_state.flags.get(flag_name)
            session.current_state.flags[flag_name] = coerced_value
            old_context_value = None
            if context_memory is not None:
                try:
                    old_context_value = context_memory.world_flags.get(flag_name)
                    context_memory.world_flags[flag_name] = coerced_value
                except Exception:  # noqa: BLE001
                    old_context_value = None
            modified[flag_name] = {
                "old": old_value,
                "new": coerced_value,
                **({"old_context": old_context_value} if context_memory is not None else {}),
            }

        # Save session
        if hasattr(engine, "save_session"):
            engine.save_session(session)
        else:
            engine._save_session(session)  # type: ignore[attr-defined]

        logger.info(
            f"Modified {len(modified)} flags for session {session_id}: "
            f"{list(modified.keys())} | Reason: {reason}"
        )

        return {
            "success": True,
            "modified_flags": modified,
            "reason": reason
        }

    except Exception as e:
        logger.error(f"Failed to modify world state: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Tool metadata for registration
TOOL_METADATA = {
    "name": "modify_world_state",
    "description": "Modify world state flags (quests, NPCs, events)",
    "parameters": {
        "flags": {
            "type": "dict",
            "description": "Flags to set/modify",
            "required": True
        },
        "reason": {
            "type": "string",
            "description": "Reason for modification",
            "required": False
        }
    },
    "examples": [
        {
            "flags": {"quest_dragon_started": True},
            "reason": "Player accepted dragon quest"
        },
        {
            "flags": {"npc_met_merchant": True, "location_discovered_market": True},
            "reason": "Player discovered market and met merchant"
        }
    ]
}
