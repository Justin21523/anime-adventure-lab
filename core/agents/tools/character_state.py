"""
Character State Modification Tool for Story Agent

Allows Agent to modify character stats (HP, MP, level, etc.).
All modifications go through safety wrapper with bounds checking.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


_STAT_ALIASES: Dict[str, str] = {
    "hp": "health",
    "mp": "energy",
    "exp": "experience",
}


def _normalize_stat_name(stat_name: str) -> str:
    key = str(stat_name or "").strip()
    return _STAT_ALIASES.get(key, key)


async def update_character_state(session_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update character stats

    Args:
        session_id: Story session ID
        params: Dictionary with:
            - stats: Dict[str, int/float] - Stats to modify
            - reason: str (optional) - Reason for modification
            - relative: bool (optional) - If True, add to current value; if False, set absolute

    Returns:
        Dictionary with modified stats and their values

    Example:
        # Damage player
        params = {
            "stats": {"hp": -20},
            "reason": "Player took damage from enemy",
            "relative": True
        }

        # Set level
        params = {
            "stats": {"level": 5, "max_hp": 150},
            "reason": "Player leveled up",
            "relative": False
        }
    """
    try:
        from core.story.engine import get_story_engine

        engine = get_story_engine()
        session = engine.get_session(session_id)

        stats = params.get("stats", {})
        reason = params.get("reason", "Agent modification")
        relative = params.get("relative", False)

        if not stats:
            return {
                "success": False,
                "error": "No stats provided"
            }

        # Apply stat changes
        modified = {}
        unknown_stats = []
        for stat_name, value in stats.items():
            canonical = _normalize_stat_name(stat_name)
            # Get current value
            old_value = getattr(session.stats, canonical, None)
            if old_value is None:
                unknown_stats.append(str(stat_name))
                continue

            # Coerce numeric value
            try:
                numeric_value: float = float(value)
            except Exception:
                unknown_stats.append(str(stat_name))
                continue

            # Calculate new value
            if relative:
                new_value = old_value + numeric_value
            else:
                new_value = numeric_value

            # Basic common-sense bounds (more strict bounds handled by safety wrapper)
            if canonical in {"health", "energy"}:
                new_value = max(0, new_value)
            if canonical == "level":
                new_value = max(1, int(new_value))
            if canonical in {"intelligence", "charisma", "luck"}:
                new_value = int(new_value)
            if canonical == "experience":
                new_value = max(0, int(new_value))

            # Set value
            setattr(session.stats, canonical, new_value)
            modified[canonical] = {
                "old": old_value,
                "new": new_value,
                "change": new_value - old_value
            }

        if not modified:
            return {
                "success": False,
                "error": f"No valid stats applied. Unknown stats: {unknown_stats}",
                "unknown_stats": unknown_stats,
            }

        # Save session
        if hasattr(engine, "save_session"):
            engine.save_session(session)
        else:
            engine._save_session(session)  # type: ignore[attr-defined]

        logger.info(
            f"Modified {len(modified)} stats for session {session_id}: "
            f"{list(modified.keys())} | Reason: {reason}"
        )

        return {
            "success": True,
            "modified_stats": modified,
            "reason": reason
        }

    except Exception as e:
        logger.error(f"Failed to update character state: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def add_inventory_item(session_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add item to player inventory

    Args:
        session_id: Story session ID
        params: Dictionary with:
            - item: str - Item name/ID
            - quantity: int (optional) - Quantity to add (default 1)
            - reason: str (optional) - Reason for addition

    Returns:
        Dictionary with success status

    Example:
        params = {
            "item": "health_potion",
            "quantity": 3,
            "reason": "Reward from quest"
        }
    """
    try:
        from core.story.engine import get_story_engine

        engine = get_story_engine()
        session = engine.get_session(session_id)

        item = params.get("item")
        quantity = params.get("quantity", 1)
        reason = params.get("reason", "Agent addition")

        if not item:
            return {"success": False, "error": "No item specified"}

        try:
            quantity_int = int(quantity)
        except Exception:
            quantity_int = 1
        if quantity_int <= 0:
            return {"success": False, "error": "quantity must be >= 1"}

        # Inventory is List[str] in this project; represent quantity by repeating entries.
        for _ in range(quantity_int):
            session.inventory.append(str(item))

        # Save session
        if hasattr(engine, "save_session"):
            engine.save_session(session)
        else:
            engine._save_session(session)  # type: ignore[attr-defined]

        logger.info(
            f"Added {quantity}x {item} to session {session_id} | Reason: {reason}"
        )

        return {
            "success": True,
            "item": item,
            "quantity": quantity_int,
            "reason": reason
        }

    except Exception as e:
        logger.error(f"Failed to add inventory item: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Tool metadata for registration
UPDATE_CHARACTER_STATE_METADATA = {
    "name": "update_character_state",
    "description": "Update character stats (HP, MP, level, etc.)",
    "parameters": {
        "stats": {
            "type": "dict",
            "description": "Stats to modify",
            "required": True
        },
        "reason": {
            "type": "string",
            "description": "Reason for modification",
            "required": False
        },
        "relative": {
            "type": "boolean",
            "description": "If True, add to current value; if False, set absolute",
            "required": False,
            "default": False
        }
    },
    "examples": [
        {
            "stats": {"hp": -30},
            "reason": "Player took damage",
            "relative": True
        },
        {
            "stats": {"level": 2, "max_hp": 120},
            "reason": "Player leveled up",
            "relative": False
        }
    ]
}

ADD_INVENTORY_ITEM_METADATA = {
    "name": "add_inventory_item",
    "description": "Add item to player inventory",
    "parameters": {
        "item": {
            "type": "string",
            "description": "Item name/ID",
            "required": True
        },
        "quantity": {
            "type": "integer",
            "description": "Quantity to add",
            "required": False,
            "default": 1
        },
        "reason": {
            "type": "string",
            "description": "Reason for addition",
            "required": False
        }
    },
    "examples": [
        {
            "item": "health_potion",
            "quantity": 2,
            "reason": "Found in chest"
        }
    ]
}
