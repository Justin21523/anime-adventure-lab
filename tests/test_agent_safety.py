"""
Tests for Agent Safety Wrapper

CRITICAL: These tests ensure Agent autonomy doesn't break the game.
Tests validation, whitelisting, blacklisting, rollback, and audit logging.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import copy

from core.agents.story_safety_wrapper import (
    StorySafetyWrapper,
    ToolValidationError,
    ToolExecutionResult,
    get_safety_wrapper
)
from core.monitoring.agent_audit_logger import AgentAuditLogger


# Mock Fixtures ---------------------------------------------------------------

@pytest.fixture
def mock_audit_logger():
    """Mock audit logger"""
    mock = AsyncMock(spec=AgentAuditLogger)
    mock.log_action = AsyncMock()
    return mock


@pytest.fixture
def safety_wrapper(mock_audit_logger):
    """Safety wrapper with mocked dependencies"""
    wrapper = StorySafetyWrapper(audit_logger=mock_audit_logger)
    return wrapper


@pytest.fixture
def mock_session():
    """Mock game session"""
    session = MagicMock()
    session.session_id = "test-session"
    session.turn_count = 5
    session.current_state.flags = {
        "quest_tutorial_complete": True,
        "npc_met_guard": False
    }
    session.stats = MagicMock()
    session.stats.hp = 100
    session.stats.max_hp = 100
    session.stats.mp = 50
    session.stats.max_mp = 50
    session.stats.level = 1
    session.stats.to_dict = lambda: {
        "hp": 100,
        "max_hp": 100,
        "mp": 50,
        "max_mp": 50,
        "level": 1
    }
    session.inventory = []
    return session


# Flag Validation Tests -------------------------------------------------------

class TestFlagValidation:
    """Test flag name validation (whitelist/blacklist)"""

    def test_allowed_quest_flag(self, safety_wrapper):
        """Quest flags should be allowed"""
        assert safety_wrapper.validate_flag_name("quest_dragon_started") is True
        assert safety_wrapper.validate_flag_name("quest_forest_complete") is True

    def test_allowed_npc_flag(self, safety_wrapper):
        """NPC flags should be allowed"""
        assert safety_wrapper.validate_flag_name("npc_met_elder") is True
        assert safety_wrapper.validate_flag_name("npc_met_merchant") is True

    def test_allowed_location_flag(self, safety_wrapper):
        """Location flags should be allowed"""
        assert safety_wrapper.validate_flag_name("location_discovered_cave") is True
        assert safety_wrapper.validate_flag_name("location_discovered_castle") is True

    def test_allowed_event_flag(self, safety_wrapper):
        """Event flags should be allowed"""
        assert safety_wrapper.validate_flag_name("event_battle_won") is True
        assert safety_wrapper.validate_flag_name("event_cutscene_1") is True

    def test_forbidden_admin_flag(self, safety_wrapper):
        """Admin flags should be forbidden"""
        with pytest.raises(ToolValidationError, match="forbidden pattern"):
            safety_wrapper.validate_flag_name("admin_god_mode")

    def test_forbidden_system_flag(self, safety_wrapper):
        """System flags should be forbidden"""
        with pytest.raises(ToolValidationError, match="forbidden pattern"):
            safety_wrapper.validate_flag_name("system_debug")

    def test_forbidden_underscore_flag(self, safety_wrapper):
        """Internal flags (starting with _) should be forbidden"""
        with pytest.raises(ToolValidationError, match="forbidden pattern"):
            safety_wrapper.validate_flag_name("_internal_state")

    def test_unwhitelisted_flag(self, safety_wrapper):
        """Flags not in whitelist should be rejected"""
        with pytest.raises(ToolValidationError, match="not in whitelist"):
            safety_wrapper.validate_flag_name("random_flag_name")


# Stat Validation Tests -------------------------------------------------------

class TestStatValidation:
    """Test stat value validation (bounds checking)"""

    def test_valid_hp(self, safety_wrapper):
        """Valid HP values should pass"""
        assert safety_wrapper.validate_stat_change("hp", 50) is True
        assert safety_wrapper.validate_stat_change("hp", 0) is True
        assert safety_wrapper.validate_stat_change("hp", 9999) is True

    def test_hp_below_min(self, safety_wrapper):
        """HP below 0 should fail"""
        with pytest.raises(ToolValidationError, match="outside allowed range"):
            safety_wrapper.validate_stat_change("hp", -1)

    def test_hp_above_max(self, safety_wrapper):
        """HP above max should fail"""
        with pytest.raises(ToolValidationError, match="outside allowed range"):
            safety_wrapper.validate_stat_change("hp", 10000)

    def test_valid_level(self, safety_wrapper):
        """Valid level values should pass"""
        assert safety_wrapper.validate_stat_change("level", 1) is True
        assert safety_wrapper.validate_stat_change("level", 50) is True
        assert safety_wrapper.validate_stat_change("level", 100) is True

    def test_level_below_min(self, safety_wrapper):
        """Level below 1 should fail"""
        with pytest.raises(ToolValidationError, match="outside allowed range"):
            safety_wrapper.validate_stat_change("level", 0)

    def test_level_above_max(self, safety_wrapper):
        """Level above 100 should fail"""
        with pytest.raises(ToolValidationError, match="outside allowed range"):
            safety_wrapper.validate_stat_change("level", 101)

    def test_unknown_stat(self, safety_wrapper):
        """Unknown stat names should fail"""
        with pytest.raises(ToolValidationError, match="Unknown stat"):
            safety_wrapper.validate_stat_change("unknown_stat", 100)

    def test_non_numeric_stat(self, safety_wrapper):
        """Non-numeric stat values should fail"""
        with pytest.raises(ToolValidationError, match="must be numeric"):
            safety_wrapper.validate_stat_change("hp", "invalid")


# Tool Parameter Validation Tests ---------------------------------------------

class TestToolParamValidation:
    """Test tool-specific parameter validation"""

    def test_modify_world_state_valid(self, safety_wrapper):
        """Valid world state modification should pass"""
        params = {
            "flags": {
                "quest_started": True,
                "npc_met_guard": True
            }
        }
        assert safety_wrapper.validate_tool_params("modify_world_state", params) is True

    def test_modify_world_state_invalid_flag(self, safety_wrapper):
        """Invalid flag in world state should fail"""
        params = {
            "flags": {
                "admin_cheat": True  # Forbidden flag
            }
        }
        with pytest.raises(ToolValidationError, match="forbidden pattern"):
            safety_wrapper.validate_tool_params("modify_world_state", params)

    def test_update_character_state_valid(self, safety_wrapper):
        """Valid character state update should pass"""
        params = {
            "stats": {
                "hp": 80,
                "mp": 40
            }
        }
        assert safety_wrapper.validate_tool_params("update_character_state", params) is True

    def test_update_character_state_invalid_value(self, safety_wrapper):
        """Invalid stat value should fail"""
        params = {
            "stats": {
                "hp": 10000  # Above max
            }
        }
        with pytest.raises(ToolValidationError, match="outside allowed range"):
            safety_wrapper.validate_tool_params("update_character_state", params)

    def test_generate_scene_image_valid(self, safety_wrapper):
        """Valid scene image generation should pass"""
        params = {
            "scene_context": {
                "location": "forest",
                "time": "night"
            }
        }
        assert safety_wrapper.validate_tool_params("generate_scene_image", params) is True

    def test_generate_scene_image_missing_location(self, safety_wrapper):
        """Scene image without location should fail"""
        params = {
            "scene_context": {
                "time": "night"
            }
        }
        with pytest.raises(ToolValidationError, match="requires 'location'"):
            safety_wrapper.validate_tool_params("generate_scene_image", params)


# Snapshot and Rollback Tests -------------------------------------------------

class TestSnapshotRollback:
    """Test state snapshot and rollback functionality"""

    @pytest.mark.asyncio
    async def test_snapshot_creation(self, safety_wrapper, mock_session):
        """Test creating state snapshot"""
        with patch('core.agents.story_safety_wrapper.get_story_engine') as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.get_session.return_value = mock_session
            mock_get_engine.return_value = mock_engine

            snapshot = await safety_wrapper._snapshot_state("test-session")

            assert "flags" in snapshot
            assert "stats" in snapshot
            assert "inventory" in snapshot
            assert "turn_count" in snapshot
            assert snapshot["turn_count"] == 5

    @pytest.mark.asyncio
    async def test_rollback_state(self, safety_wrapper, mock_session):
        """Test rolling back to snapshot"""
        snapshot = {
            "flags": {"quest_tutorial_complete": True},
            "stats": {"hp": 100, "mp": 50, "level": 1},
            "inventory": [],
            "turn_count": 5
        }

        # Modify session state
        mock_session.current_state.flags["npc_met_guard"] = True
        mock_session.stats.hp = 50

        with patch('core.agents.story_safety_wrapper.get_story_engine') as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.get_session.return_value = mock_session
            mock_get_engine.return_value = mock_engine

            await safety_wrapper._rollback_state("test-session", snapshot)

            # Should restore to snapshot state
            assert "npc_met_guard" not in mock_session.current_state.flags or \
                   not mock_session.current_state.flags["npc_met_guard"]


# Tool Execution Tests --------------------------------------------------------

class TestToolExecution:
    """Test complete tool execution flow"""

    @pytest.mark.asyncio
    async def test_successful_execution(self, safety_wrapper, mock_audit_logger):
        """Test successful tool execution"""
        async def mock_tool(session_id, params):
            return {"success": True, "modified": params["flags"]}

        safety_wrapper.tool_registry = {"modify_world_state": mock_tool}

        params = {
            "flags": {"quest_started": True}
        }

        result = await safety_wrapper.execute_tool(
            "modify_world_state",
            "test-session",
            params
        )

        assert result.success is True
        assert result.tool_name == "modify_world_state"
        assert result.error is None

        # Audit log should be called
        mock_audit_logger.log_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_validation_failure(self, safety_wrapper, mock_audit_logger):
        """Test tool execution with validation failure"""
        params = {
            "flags": {"admin_cheat": True}  # Forbidden flag
        }

        result = await safety_wrapper.execute_tool(
            "modify_world_state",
            "test-session",
            params
        )

        assert result.success is False
        assert "Validation failed" in result.error
        assert result.rollback_performed is False

        # Audit log should record failure
        mock_audit_logger.log_action.assert_called_once()
        call_args = mock_audit_logger.log_action.call_args[1]
        assert call_args["success"] is False

    @pytest.mark.asyncio
    async def test_execution_failure_with_rollback(self, safety_wrapper, mock_audit_logger, mock_session):
        """Test tool execution failure triggers rollback"""
        async def failing_tool(session_id, params):
            raise Exception("Tool execution failed")

        safety_wrapper.tool_registry = {"modify_world_state": failing_tool}

        params = {
            "flags": {"quest_started": True}
        }

        with patch.object(safety_wrapper, '_snapshot_state', return_value={"flags": {}}):
            with patch.object(safety_wrapper, '_rollback_state') as mock_rollback:
                result = await safety_wrapper.execute_tool(
                    "modify_world_state",
                    "test-session",
                    params
                )

                assert result.success is False
                assert "Execution failed" in result.error
                assert result.rollback_performed is True

                # Rollback should be called
                mock_rollback.assert_called_once()


# Integration Tests -----------------------------------------------------------

class TestAgentSafetyIntegration:
    """Integration tests for full safety pipeline"""

    @pytest.mark.asyncio
    async def test_safe_flag_modification(self, safety_wrapper, mock_audit_logger):
        """Test complete safe flag modification"""
        # This would normally call real tool, but we mock it
        async def mock_modify(session_id, params):
            return {"success": True, "modified_flags": params["flags"]}

        safety_wrapper.tool_registry = {"modify_world_state": mock_modify}

        params = {
            "flags": {
                "quest_dragon_started": True,
                "npc_met_elder": True,
                "location_discovered_cave": True
            },
            "reason": "Player progression"
        }

        result = await safety_wrapper.execute_tool(
            "modify_world_state",
            "test-session",
            params
        )

        assert result.success is True
        assert result.result["success"] is True

    @pytest.mark.asyncio
    async def test_blocked_dangerous_modification(self, safety_wrapper, mock_audit_logger):
        """Test that dangerous modifications are blocked"""
        params = {
            "flags": {
                "admin_god_mode": True,  # Should be blocked
                "system_unlock_all": True  # Should be blocked
            }
        }

        result = await safety_wrapper.execute_tool(
            "modify_world_state",
            "test-session",
            params
        )

        assert result.success is False
        assert "forbidden pattern" in result.error.lower()

    @pytest.mark.asyncio
    async def test_stat_bounds_enforcement(self, safety_wrapper):
        """Test that stat bounds are enforced"""
        params = {
            "stats": {
                "hp": 10000,  # Above max
                "level": 150  # Above max
            }
        }

        result = await safety_wrapper.execute_tool(
            "update_character_state",
            "test-session",
            params
        )

        assert result.success is False
        assert "outside allowed range" in result.error.lower()


# Singleton Tests -------------------------------------------------------------

def test_safety_wrapper_singleton():
    """Test safety wrapper singleton pattern"""
    wrapper1 = get_safety_wrapper()
    wrapper2 = get_safety_wrapper()

    # Should be same instance
    assert wrapper1 is wrapper2
