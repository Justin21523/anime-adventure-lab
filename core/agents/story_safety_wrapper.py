"""
Story Agent Safety Wrapper

CRITICAL SECURITY COMPONENT - Wraps all Agent tool calls with:
1. Whitelist validation (only allowed flags/operations)
2. Blacklist filtering (forbidden patterns)
3. Audit logging (all actions logged)
4. Automatic rollback (restore on failure)
5. Parameter validation

This ensures Agent autonomy doesn't break the game.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import copy

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Result from tool execution"""
    success: bool
    tool_name: str
    session_id: str
    params: Dict[str, Any]
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0
    rollback_performed: bool = False


class ToolValidationError(Exception):
    """Raised when tool parameters fail validation"""
    pass


class StorySafetyWrapper:
    """
    Security wrapper for Story Agent tools

    Prevents Agent from:
    - Modifying admin/system flags
    - Setting invalid stat values
    - Breaking game invariants
    - Performing forbidden operations

    Ensures:
    - All actions are audited
    - Failed actions are rolled back
    - Only whitelisted operations succeed
    """

    # Flag pattern whitelists (regex)
    ALLOWED_FLAG_PATTERNS = [
        r"^quest_.*",           # Quest flags: quest_forest_started, quest_dragon_defeated
        r"^npc_met_.*",         # NPC tracking: npc_met_elder, npc_met_merchant
        r"^location_discovered_.*",  # Location discovery: location_discovered_cave
        r"^item_acquired_.*",   # Item tracking: item_acquired_sword
        r"^event_.*",           # Story events: event_battle_won, event_cutscene_1
        r"^achievement_.*",     # Achievements: achievement_first_kill
    ]

    # Flag pattern blacklists (forbidden)
    FORBIDDEN_FLAG_PATTERNS = [
        r"^admin_.*",           # Admin flags
        r"^system_.*",          # System flags
        r"^debug_.*",           # Debug flags
        r"^test_.*",            # Test flags
        r"^_.*",                # Internal flags (start with underscore)
    ]

    # Stat constraints (align with core/story/game_state.py PlayerStats)
    STAT_CONSTRAINTS = {
        "health": {"min": 0, "max": 9999},
        "energy": {"min": 0, "max": 9999},
        "intelligence": {"min": 0, "max": 999},
        "charisma": {"min": 0, "max": 999},
        "luck": {"min": 0, "max": 999},
        "experience": {"min": 0, "max": 999999},
        "level": {"min": 1, "max": 100},
        # Legacy aliases (accepted, normalized internally)
        "hp": {"min": 0, "max": 9999},
        "mp": {"min": 0, "max": 9999},
        "exp": {"min": 0, "max": 999999},
    }

    _STAT_ALIASES = {
        "hp": "health",
        "mp": "energy",
        "exp": "experience",
    }

    def __init__(self, tool_registry=None, audit_logger=None):
        """
        Initialize safety wrapper

        Args:
            tool_registry: Tool registry for looking up tools
            audit_logger: Audit logger for recording actions
        """
        self.tool_registry = tool_registry
        self.audit_logger = audit_logger

        # State snapshots for rollback
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    def validate_flag_name(self, flag_name: str) -> bool:
        """
        Validate flag name against whitelist/blacklist

        Args:
            flag_name: Flag name to validate

        Returns:
            True if flag is allowed

        Raises:
            ToolValidationError if flag is forbidden
        """
        # Check blacklist first (forbidden patterns)
        for pattern in self.FORBIDDEN_FLAG_PATTERNS:
            if re.match(pattern, flag_name):
                raise ToolValidationError(
                    f"Flag '{flag_name}' matches forbidden pattern '{pattern}'"
                )

        # Check whitelist (allowed patterns)
        for pattern in self.ALLOWED_FLAG_PATTERNS:
            if re.match(pattern, flag_name):
                return True

        # Not in whitelist
        raise ToolValidationError(
            f"Flag '{flag_name}' not in whitelist. Allowed patterns: {self.ALLOWED_FLAG_PATTERNS}"
        )

    def validate_stat_change(self, stat_name: str, new_value: Any) -> bool:
        """
        Validate stat change is within allowed bounds

        Args:
            stat_name: Stat name (hp, mp, level, etc.)
            new_value: New value to set

        Returns:
            True if valid

        Raises:
            ToolValidationError if invalid
        """
        key = str(stat_name or "").strip()
        canonical = self._STAT_ALIASES.get(key, key)
        if canonical not in self.STAT_CONSTRAINTS:
            raise ToolValidationError(
                f"Unknown stat '{stat_name}'. Allowed stats: {list(self.STAT_CONSTRAINTS.keys())}"
            )

        # Convert to number
        try:
            numeric_value = float(new_value)
        except (ValueError, TypeError):
            raise ToolValidationError(
                f"Stat '{stat_name}' must be numeric, got {type(new_value)}"
            )

        # Check bounds
        constraints = self.STAT_CONSTRAINTS[canonical]
        if numeric_value < constraints["min"] or numeric_value > constraints["max"]:
            raise ToolValidationError(
                f"Stat '{stat_name}' value {numeric_value} outside allowed range "
                f"[{constraints['min']}, {constraints['max']}]"
            )

        return True

    def validate_tool_params(self, tool_name: str, session_id: str, params: Dict[str, Any]) -> bool:
        """
        Validate tool parameters based on tool type

        Args:
            tool_name: Name of tool being called
            session_id: Story session ID (for validating relative changes)
            params: Parameters passed to tool

        Returns:
            True if params are valid

        Raises:
            ToolValidationError if params are invalid
        """
        # Tool-specific validation
        if tool_name == "modify_world_state":
            # Validate all flag names
            flags = params.get("flags", {})
            for flag_name in flags.keys():
                self.validate_flag_name(flag_name)

        elif tool_name == "update_character_state":
            # Validate stat changes (support relative deltas)
            from core.story.engine import get_story_engine

            engine = get_story_engine()
            session = engine.get_session(session_id)

            stats = params.get("stats", {}) or {}
            relative = bool(params.get("relative", False))

            for stat_name, value in stats.items():
                key = str(stat_name or "").strip()
                canonical = self._STAT_ALIASES.get(key, key)
                if not hasattr(session.stats, canonical):
                    raise ToolValidationError(f"Unknown stat '{stat_name}'")

                try:
                    numeric_value = float(value)
                except (ValueError, TypeError) as exc:
                    raise ToolValidationError(
                        f"Stat '{stat_name}' must be numeric, got {type(value)}"
                    ) from exc

                current_value = getattr(session.stats, canonical)
                proposed = current_value + numeric_value if relative else numeric_value

                # Mirror tool-side normalization/clamping.
                if canonical in {"health", "energy"}:
                    proposed = max(0, proposed)
                if canonical == "level":
                    proposed = max(1, int(proposed))
                if canonical in {"intelligence", "charisma", "luck"}:
                    proposed = int(proposed)
                if canonical == "experience":
                    proposed = max(0, int(proposed))

                self.validate_stat_change(canonical, proposed)

        elif tool_name == "add_inventory_item":
            item = params.get("item")
            if not item or not str(item).strip():
                raise ToolValidationError("add_inventory_item requires non-empty 'item'")
            qty = params.get("quantity", 1)
            try:
                qty_int = int(qty)
            except Exception as exc:  # noqa: BLE001
                raise ToolValidationError(f"Invalid quantity: {qty}") from exc
            if qty_int < 1 or qty_int > 99:
                raise ToolValidationError("quantity must be between 1 and 99")

        elif tool_name == "update_relationship_state":
            from core.story.engine import get_story_engine

            engine = get_story_engine()
            if not getattr(engine, "enhanced_mode", False) or session_id not in getattr(engine, "context_memories", {}):
                raise ToolValidationError("update_relationship_state requires enhanced mode context memory")

            context_memory = engine.context_memories[session_id]  # type: ignore[attr-defined]

            rels = params.get("relationships", {}) or {}
            if not isinstance(rels, dict) or not rels:
                raise ToolValidationError("relationships must be a non-empty dict")

            relative = bool(params.get("relative", True))
            for raw_id, raw_value in rels.items():
                char_id = str(raw_id or "").strip()
                if not char_id:
                    raise ToolValidationError("character_id cannot be empty")
                if len(char_id) > 64 or "\n" in char_id or "\r" in char_id:
                    raise ToolValidationError("character_id is invalid")

                try:
                    change_int = int(raw_value)
                except Exception as exc:  # noqa: BLE001
                    raise ToolValidationError(f"Invalid relationship value: {raw_value}") from exc

                if change_int < -10 or change_int > 10:
                    raise ToolValidationError("relationship change must be between -10 and 10")

                if char_id != "any":
                    current = int(context_memory.player_relationships.get(char_id, 0) or 0)
                    proposed = current + change_int if relative else change_int
                    proposed = max(-10, min(10, proposed))
                    if proposed < -10 or proposed > 10:
                        raise ToolValidationError("relationship score out of range")

        elif tool_name == "generate_scene_image":
            # Validate scene context has required fields
            scene_context = params.get("scene_context", {})
            if not scene_context.get("location"):
                raise ToolValidationError(
                    "generate_scene_image requires 'location' in scene_context"
                )

        return True

    async def _snapshot_state(self, session_id: str) -> Dict[str, Any]:
        """
        Create snapshot of current session state for rollback

        Args:
            session_id: Session ID

        Returns:
            State snapshot dictionary
        """
        try:
            from core.story.engine import get_story_engine

            engine = get_story_engine()
            session = engine.get_session(session_id)

            # Deep copy important state
            snapshot = {
                "flags": copy.deepcopy(session.current_state.flags),
                "stats": copy.deepcopy(session.stats.to_dict()),
                "inventory": copy.deepcopy(session.inventory),
                "turn_count": session.turn_count,
                "timestamp": datetime.now().isoformat()
            }

            # Enhanced mode context snapshots (world_flags / relationships)
            try:
                if getattr(engine, "enhanced_mode", False) and session_id in getattr(engine, "context_memories", {}):
                    context_memory = engine.context_memories[session_id]  # type: ignore[attr-defined]
                    snapshot["context_world_flags"] = copy.deepcopy(
                        getattr(context_memory, "world_flags", {}) or {}
                    )
                    snapshot["context_player_relationships"] = copy.deepcopy(
                        getattr(context_memory, "player_relationships", {}) or {}
                    )
            except Exception:  # noqa: BLE001
                pass

            return snapshot

        except Exception as e:
            logger.error(f"Failed to create state snapshot: {e}")
            return {}

    async def _rollback_state(self, session_id: str, snapshot: Dict[str, Any]) -> None:
        """
        Rollback session to previous snapshot

        Args:
            session_id: Session ID
            snapshot: State snapshot to restore
        """
        try:
            from core.story.engine import get_story_engine

            engine = get_story_engine()
            session = engine.get_session(session_id)

            # Restore state
            session.current_state.flags = copy.deepcopy(snapshot.get("flags", {}))
            session.inventory = copy.deepcopy(snapshot.get("inventory", []))

            # Restore stats
            stats_data = snapshot.get("stats", {})
            for key, value in stats_data.items():
                setattr(session.stats, key, value)

            # Restore enhanced context if present
            try:
                if getattr(engine, "enhanced_mode", False) and session_id in getattr(engine, "context_memories", {}):
                    context_memory = engine.context_memories[session_id]  # type: ignore[attr-defined]
                    if "context_world_flags" in snapshot:
                        context_memory.world_flags = copy.deepcopy(snapshot.get("context_world_flags", {}))
                    if "context_player_relationships" in snapshot:
                        context_memory.player_relationships = copy.deepcopy(
                            snapshot.get("context_player_relationships", {})
                        )
            except Exception:  # noqa: BLE001
                pass

            # Persist rollback so session/context files remain consistent
            try:
                if hasattr(engine, "save_session"):
                    engine.save_session(session)
                else:
                    engine._save_session(session)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

            logger.info(f"Rolled back session {session_id} to snapshot from {snapshot.get('timestamp')}")

        except Exception as e:
            logger.error(f"Failed to rollback state: {e}")

    async def execute_tool(
        self,
        tool_name: str,
        session_id: str,
        params: Dict[str, Any]
    ) -> ToolExecutionResult:
        """
        Execute tool with safety checks

        Args:
            tool_name: Name of tool to execute
            session_id: Story session ID
            params: Tool parameters

        Returns:
            ToolExecutionResult with success/failure status
        """
        start_time = datetime.now()
        snapshot = None
        rollback_performed = False

        try:
            # 1. Validate parameters
            logger.info(f"Validating tool '{tool_name}' params for session {session_id}")
            self.validate_tool_params(tool_name, session_id, params)

            # 2. Create state snapshot
            snapshot = await self._snapshot_state(session_id)

            # 3. Execute tool
            logger.info(f"Executing tool '{tool_name}' for session {session_id}")

            if self.tool_registry:
                tool_func = None
                if hasattr(self.tool_registry, "get_function"):
                    tool_func = self.tool_registry.get_function(tool_name)  # type: ignore[attr-defined]
                elif isinstance(self.tool_registry, dict):
                    tool_func = self.tool_registry.get(tool_name)
                if not tool_func:
                    raise ValueError(f"Tool '{tool_name}' not found in registry")

                # Prefer calling signature: tool(session_id, params) for story tools.
                try:
                    result = await tool_func(session_id, params)
                except TypeError:
                    # Fallback to keyword style: tool(session_id=..., **params)
                    result = await tool_func(session_id=session_id, **(params or {}))
            else:
                # Fallback: import and call directly
                result = await self._execute_tool_direct(tool_name, session_id, params)

            # 4. Log success
            if self.audit_logger:
                await self.audit_logger.log_action(
                    session_id=session_id,
                    tool_name=tool_name,
                    params=params,
                    result=result,
                    success=True
                )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return ToolExecutionResult(
                success=True,
                tool_name=tool_name,
                session_id=session_id,
                params=params,
                result=result,
                execution_time_ms=execution_time
            )

        except ToolValidationError as e:
            # Validation failed - don't rollback (nothing changed)
            logger.warning(f"Tool validation failed: {e}")

            if self.audit_logger:
                await self.audit_logger.log_action(
                    session_id=session_id,
                    tool_name=tool_name,
                    params=params,
                    result=None,
                    success=False,
                    error=str(e)
                )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                session_id=session_id,
                params=params,
                result=None,
                error=f"Validation failed: {str(e)}",
                execution_time_ms=execution_time
            )

        except Exception as e:
            # Execution failed - rollback to snapshot
            logger.error(f"Tool execution failed: {e}", exc_info=True)

            if snapshot:
                await self._rollback_state(session_id, snapshot)
                rollback_performed = True
                logger.info(f"Rolled back session {session_id} after tool failure")

            if self.audit_logger:
                await self.audit_logger.log_action(
                    session_id=session_id,
                    tool_name=tool_name,
                    params=params,
                    result=None,
                    success=False,
                    error=str(e)
                )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                session_id=session_id,
                params=params,
                result=None,
                error=f"Execution failed: {str(e)}",
                execution_time_ms=execution_time,
                rollback_performed=rollback_performed
            )

    async def _execute_tool_direct(
        self,
        tool_name: str,
        session_id: str,
        params: Dict[str, Any]
    ) -> Any:
        """Direct tool execution fallback"""
        # Import and execute tool functions
        if tool_name == "modify_world_state":
            from core.agents.tools.world_state import modify_world_state
            return await modify_world_state(session_id, params)
        elif tool_name == "update_character_state":
            from core.agents.tools.character_state import update_character_state
            return await update_character_state(session_id, params)
        elif tool_name == "add_inventory_item":
            from core.agents.tools.character_state import add_inventory_item
            return await add_inventory_item(session_id, params)
        elif tool_name == "generate_scene_image":
            from core.story.t2i_integration import get_t2i_integration
            t2i = get_t2i_integration()
            return await t2i.generate_scene_image(
                scene_context=params.get("scene_context", {}),
                narrative_text=params.get("narrative_text", ""),
                force=params.get("force", False)
            )
        elif tool_name == "rag_search":
            from core.agents.tools.rag_search import rag_search
            return await rag_search(session_id, params)
        elif tool_name == "update_relationship_state":
            from core.agents.tools.relationship_state import update_relationship_state
            return await update_relationship_state(session_id, params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


# Singleton instance
_safety_wrapper: Optional[StorySafetyWrapper] = None


def get_safety_wrapper(tool_registry=None, audit_logger=None) -> StorySafetyWrapper:
    """Get or create singleton safety wrapper"""
    global _safety_wrapper
    if _safety_wrapper is None:
        _safety_wrapper = StorySafetyWrapper(tool_registry, audit_logger)
    return _safety_wrapper
