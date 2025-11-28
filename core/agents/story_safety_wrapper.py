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

    # Stat constraints
    STAT_CONSTRAINTS = {
        "hp": {"min": 0, "max": 9999},
        "mp": {"min": 0, "max": 9999},
        "max_hp": {"min": 1, "max": 9999},
        "max_mp": {"min": 1, "max": 9999},
        "level": {"min": 1, "max": 100},
        "exp": {"min": 0, "max": 999999},
        "gold": {"min": 0, "max": 999999},
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
        if stat_name not in self.STAT_CONSTRAINTS:
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
        constraints = self.STAT_CONSTRAINTS[stat_name]
        if numeric_value < constraints["min"] or numeric_value > constraints["max"]:
            raise ToolValidationError(
                f"Stat '{stat_name}' value {numeric_value} outside allowed range "
                f"[{constraints['min']}, {constraints['max']}]"
            )

        return True

    def validate_tool_params(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """
        Validate tool parameters based on tool type

        Args:
            tool_name: Name of tool being called
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
            # Validate stat changes
            stats = params.get("stats", {})
            for stat_name, value in stats.items():
                self.validate_stat_change(stat_name, value)

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
            self.validate_tool_params(tool_name, params)

            # 2. Create state snapshot
            snapshot = await self._snapshot_state(session_id)

            # 3. Execute tool
            logger.info(f"Executing tool '{tool_name}' for session {session_id}")

            if self.tool_registry:
                tool_func = self.tool_registry.get(tool_name)
                if not tool_func:
                    raise ValueError(f"Tool '{tool_name}' not found in registry")
                result = await tool_func(session_id, params)
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
