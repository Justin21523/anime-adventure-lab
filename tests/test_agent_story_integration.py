"""
Tests for Agent Story Integration

Tests the complete integration of Agent decision layer into story turn processing.
All tests use mocks to avoid GPU usage.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from core.agents.story_agent_layer import (
    StoryAgentLayer,
    AgentDecision,
    get_agent_layer,
    reset_agent_layer
)


# Mock Fixtures ---------------------------------------------------------------

@pytest.fixture
def mock_safety_wrapper():
    """Mock safety wrapper"""
    mock = AsyncMock()
    mock.execute_tool = AsyncMock(return_value=MagicMock(
        success=True,
        tool_name="modify_world_state",
        session_id="test-session",
        params={},
        result={"success": True, "modified_flags": {"quest_dragon_started": {"old": None, "new": True}}},
        error=None,
        execution_time_ms=10.0,
        rollback_performed=False
    ))
    return mock


@pytest.fixture
def mock_llm_adapter():
    """Mock LLM adapter"""
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value="Agent analysis result")
    return mock


@pytest.fixture
def agent_layer(mock_safety_wrapper, mock_llm_adapter):
    """Agent layer with mocked dependencies"""
    reset_agent_layer()
    layer = StoryAgentLayer(
        safety_wrapper=mock_safety_wrapper,
        llm_adapter=mock_llm_adapter
    )
    return layer


@pytest.fixture
def mock_context_memory():
    """Mock story context memory"""
    mock = MagicMock()

    # Mock current scene
    mock_scene = MagicMock()
    mock_scene.scene_id = "forest_entrance"
    mock_scene.title = "森林入口"
    mock_scene.location = "dark_forest"
    mock_scene.scene_objectives = ["探索森林", "尋找出路"]
    mock_scene.primary_npc = "forest_guardian"

    mock.get_current_scene = MagicMock(return_value=mock_scene)
    mock.get_characters_in_scene = MagicMock(return_value=[])

    return mock


# Intervention Detection Tests ------------------------------------------------

class TestAgentIntervention:
    """Test Agent intervention detection logic"""

    @pytest.mark.asyncio
    async def test_should_intervene_with_quest_keywords(self, agent_layer, mock_context_memory):
        """Agent should intervene when quest keywords are detected"""
        player_input = "我想接受這個任務"
        narrative = "你成功完成了龍之試煉任務"

        should_intervene, reason = await agent_layer.should_agent_intervene(
            session_id="test-session",
            player_input=player_input,
            narrative_text=narrative,
            context_memory=mock_context_memory
        )

        assert should_intervene is True
        assert "keyword" in reason.lower()

    @pytest.mark.asyncio
    async def test_should_intervene_with_scene_objectives(self, agent_layer, mock_context_memory):
        """Agent should intervene when scene has objectives"""
        player_input = "向前走"
        narrative = "你向前走了幾步"

        should_intervene, reason = await agent_layer.should_agent_intervene(
            session_id="test-session",
            player_input=player_input,
            narrative_text=narrative,
            context_memory=mock_context_memory
        )

        assert should_intervene is True
        assert "objective" in reason.lower()

    @pytest.mark.asyncio
    async def test_should_not_intervene_simple_narrative(self, agent_layer):
        """Agent should not intervene for simple narratives"""
        player_input = "看看周圍"
        narrative = "你看了看周圍，什麼也沒發現"

        should_intervene, reason = await agent_layer.should_agent_intervene(
            session_id="test-session",
            player_input=player_input,
            narrative_text=narrative,
            context_memory=None
        )

        assert should_intervene is False

    @pytest.mark.asyncio
    async def test_should_intervene_when_disabled(self, agent_layer, mock_context_memory):
        """Agent should not intervene when disabled"""
        agent_layer.enabled = False

        player_input = "完成任務"
        narrative = "你完成了任務，獲得經驗值"

        should_intervene, reason = await agent_layer.should_agent_intervene(
            session_id="test-session",
            player_input=player_input,
            narrative_text=narrative,
            context_memory=mock_context_memory
        )

        assert should_intervene is False
        assert "disabled" in reason.lower()


# Decision Making Tests -------------------------------------------------------

class TestAgentDecisionMaking:
    """Test Agent decision-making logic"""

    @pytest.mark.asyncio
    async def test_detect_quest_completion(self, agent_layer, mock_context_memory):
        """Agent should detect quest completion and set flag"""
        player_input = "擊敗巨龍"
        narrative = "你成功擊敗了巨龍，完成了龍之試煉"
        stats = {"hp": 100, "mp": 50, "level": 5}

        decision = await agent_layer.make_decision(
            session_id="test-session",
            player_input=player_input,
            narrative_text=narrative,
            context_memory=mock_context_memory,
            session_stats=stats
        )

        assert decision is not None
        assert decision.decision_type == "story_event_processing"
        assert len(decision.tool_calls) > 0

        # Should have quest completion tool call
        quest_calls = [tc for tc in decision.tool_calls if tc["tool"] == "modify_world_state"]
        assert len(quest_calls) > 0

    @pytest.mark.asyncio
    async def test_detect_damage(self, agent_layer, mock_context_memory):
        """Agent should detect damage and update HP"""
        player_input = "被攻擊"
        narrative = "敵人攻擊了你，你受到 30 傷害"
        stats = {"hp": 100, "mp": 50, "level": 5}

        decision = await agent_layer.make_decision(
            session_id="test-session",
            player_input=player_input,
            narrative_text=narrative,
            context_memory=mock_context_memory,
            session_stats=stats
        )

        assert decision is not None

        # Should have damage tool call
        damage_calls = [tc for tc in decision.tool_calls if tc["tool"] == "update_character_state"]
        assert len(damage_calls) > 0

        damage_call = damage_calls[0]
        assert damage_call["params"]["stats"]["hp"] == -30
        assert damage_call["params"]["relative"] is True

    @pytest.mark.asyncio
    async def test_detect_item_acquisition(self, agent_layer, mock_context_memory):
        """Agent should detect item acquisition"""
        player_input = "打開寶箱"
        narrative = "你打開了寶箱，獲得了一把劍"
        stats = {"hp": 100, "mp": 50, "level": 5}

        decision = await agent_layer.make_decision(
            session_id="test-session",
            player_input=player_input,
            narrative_text=narrative,
            context_memory=mock_context_memory,
            session_stats=stats
        )

        assert decision is not None

        # Should have item acquisition tool call
        item_calls = [tc for tc in decision.tool_calls if tc["tool"] == "add_inventory_item"]
        assert len(item_calls) > 0

    @pytest.mark.asyncio
    async def test_detect_npc_encounter(self, agent_layer, mock_context_memory):
        """Agent should detect NPC encounters"""
        player_input = "前進"
        narrative = "你進入了森林，遇到了守護者"
        stats = {"hp": 100, "mp": 50, "level": 5}

        decision = await agent_layer.make_decision(
            session_id="test-session",
            player_input=player_input,
            narrative_text=narrative,
            context_memory=mock_context_memory,
            session_stats=stats
        )

        assert decision is not None

        # Should have NPC flag tool call
        npc_calls = [
            tc for tc in decision.tool_calls
            if tc["tool"] == "modify_world_state" and "npc_met_" in str(tc["params"])
        ]
        assert len(npc_calls) > 0

    @pytest.mark.asyncio
    async def test_detect_location_discovery(self, agent_layer, mock_context_memory):
        """Agent should detect location discovery"""
        player_input = "探索"
        narrative = "你進入了黑暗森林，發現了一個神秘的地點"
        stats = {"hp": 100, "mp": 50, "level": 5}

        decision = await agent_layer.make_decision(
            session_id="test-session",
            player_input=player_input,
            narrative_text=narrative,
            context_memory=mock_context_memory,
            session_stats=stats
        )

        assert decision is not None

        # Should have location discovery tool call
        location_calls = [
            tc for tc in decision.tool_calls
            if tc["tool"] == "modify_world_state" and "location_discovered_" in str(tc["params"])
        ]
        assert len(location_calls) > 0

    @pytest.mark.asyncio
    async def test_no_decision_for_simple_narrative(self, agent_layer, mock_context_memory):
        """Agent should not make decisions for simple narratives"""
        player_input = "看看周圍"
        narrative = "你看了看周圍"
        stats = {"hp": 100, "mp": 50, "level": 5}

        decision = await agent_layer.make_decision(
            session_id="test-session",
            player_input=player_input,
            narrative_text=narrative,
            context_memory=mock_context_memory,
            session_stats=stats
        )

        assert decision is None


# Decision Execution Tests ----------------------------------------------------

class TestAgentExecution:
    """Test Agent decision execution"""

    @pytest.mark.asyncio
    async def test_execute_single_tool(self, agent_layer, mock_safety_wrapper):
        """Test executing a single tool call"""
        decision = AgentDecision(
            decision_type="story_event_processing",
            tool_calls=[
                {
                    "tool": "modify_world_state",
                    "params": {
                        "flags": {"quest_started": True},
                        "reason": "Quest started"
                    }
                }
            ],
            reasoning="Player started quest",
            confidence=0.9
        )

        result = await agent_layer.execute_decision("test-session", decision)

        assert result["overall_success"] is True
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["success"] is True

        # Safety wrapper should be called
        mock_safety_wrapper.execute_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_multiple_tools(self, agent_layer, mock_safety_wrapper):
        """Test executing multiple tool calls"""
        decision = AgentDecision(
            decision_type="story_event_processing",
            tool_calls=[
                {
                    "tool": "modify_world_state",
                    "params": {"flags": {"quest_complete": True}, "reason": "Quest complete"}
                },
                {
                    "tool": "update_character_state",
                    "params": {"stats": {"exp": 100}, "reason": "Reward", "relative": True}
                }
            ],
            reasoning="Quest completion rewards",
            confidence=0.95
        )

        result = await agent_layer.execute_decision("test-session", decision)

        assert result["overall_success"] is True
        assert len(result["tool_results"]) == 2
        assert mock_safety_wrapper.execute_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_failure(self, agent_layer):
        """Test execution with tool failure"""
        # Create a failing safety wrapper
        failing_wrapper = AsyncMock()
        failing_wrapper.execute_tool = AsyncMock(return_value=MagicMock(
            success=False,
            error="Validation failed: forbidden flag",
            rollback_performed=False
        ))

        agent_layer.safety_wrapper = failing_wrapper

        decision = AgentDecision(
            decision_type="story_event_processing",
            tool_calls=[
                {
                    "tool": "modify_world_state",
                    "params": {"flags": {"admin_cheat": True}}  # Forbidden
                }
            ],
            reasoning="Invalid operation",
            confidence=0.5
        )

        result = await agent_layer.execute_decision("test-session", decision)

        assert result["overall_success"] is False
        assert len(result["errors"]) > 0


# Integration Tests -----------------------------------------------------------

class TestFullAgentIntegration:
    """Integration tests for complete Agent flow"""

    @pytest.mark.asyncio
    async def test_full_quest_completion_flow(self, agent_layer, mock_context_memory, mock_safety_wrapper):
        """Test complete flow: intervention → decision → execution"""
        player_input = "擊敗巨龍"
        narrative = "你成功擊敗了巨龍，完成了任務，獲得了寶劍"
        stats = {"hp": 80, "mp": 30, "level": 5}

        # Step 1: Check intervention
        should_intervene, reason = await agent_layer.should_agent_intervene(
            "test-session",
            player_input,
            narrative,
            mock_context_memory
        )

        assert should_intervene is True

        # Step 2: Make decision
        decision = await agent_layer.make_decision(
            "test-session",
            player_input,
            narrative,
            mock_context_memory,
            stats
        )

        assert decision is not None
        assert len(decision.tool_calls) > 0

        # Step 3: Execute decision
        result = await agent_layer.execute_decision("test-session", decision)

        assert result["overall_success"] is True
        assert len(result["tool_results"]) > 0

    @pytest.mark.asyncio
    async def test_damage_and_healing_flow(self, agent_layer, mock_context_memory, mock_safety_wrapper):
        """Test damage detection and stat updates"""
        player_input = "戰鬥"
        narrative = "你與敵人戰鬥，受到 50 傷害"
        stats = {"hp": 100, "mp": 50, "level": 5}

        should_intervene, _ = await agent_layer.should_agent_intervene(
            "test-session", player_input, narrative, mock_context_memory
        )

        assert should_intervene is True

        decision = await agent_layer.make_decision(
            "test-session", player_input, narrative, mock_context_memory, stats
        )

        assert decision is not None

        # Should have damage tool call
        damage_calls = [tc for tc in decision.tool_calls if tc["tool"] == "update_character_state"]
        assert len(damage_calls) > 0


# Singleton Tests -------------------------------------------------------------

def test_agent_layer_singleton():
    """Test Agent layer singleton pattern"""
    reset_agent_layer()

    layer1 = get_agent_layer()
    layer2 = get_agent_layer()

    assert layer1 is layer2


def test_agent_decision_to_dict():
    """Test AgentDecision serialization"""
    decision = AgentDecision(
        decision_type="test",
        tool_calls=[{"tool": "test_tool", "params": {}}],
        reasoning="Test reasoning",
        confidence=0.8
    )

    result = decision.to_dict()

    assert result["decision_type"] == "test"
    assert len(result["tool_calls"]) == 1
    assert result["reasoning"] == "Test reasoning"
    assert result["confidence"] == 0.8
    assert "timestamp" in result
