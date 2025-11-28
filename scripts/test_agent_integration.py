#!/usr/bin/env python3
"""
Standalone test script for Agent Story Integration
Tests core functionality without pytest dependencies
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_agent_layer_creation():
    """Test creating Agent layer"""
    print("Test 1: Agent layer creation...")
    try:
        from core.agents.story_agent_layer import StoryAgentLayer, AgentDecision

        # Create mock dependencies
        from unittest.mock import AsyncMock, MagicMock

        mock_wrapper = AsyncMock()
        mock_llm = AsyncMock()

        agent_layer = StoryAgentLayer(
            safety_wrapper=mock_wrapper,
            llm_adapter=mock_llm
        )

        assert agent_layer is not None
        assert agent_layer.enabled is True
        print("✓ Agent layer created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


async def test_intervention_detection():
    """Test Agent intervention detection"""
    print("\nTest 2: Agent intervention detection...")
    try:
        from core.agents.story_agent_layer import StoryAgentLayer
        from unittest.mock import AsyncMock, MagicMock

        mock_wrapper = AsyncMock()
        mock_llm = AsyncMock()

        agent_layer = StoryAgentLayer(mock_wrapper, mock_llm)

        # Mock context with objectives
        mock_context = MagicMock()
        mock_scene = MagicMock()
        mock_scene.scene_objectives = ["探索", "尋找"]
        mock_context.get_current_scene = MagicMock(return_value=mock_scene)

        should_intervene, reason = await agent_layer.should_agent_intervene(
            "test-session",
            "向前走",
            "你向前走了幾步",
            mock_context
        )

        assert should_intervene is True
        print(f"✓ Intervention detected: {reason}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_decision_making():
    """Test Agent decision making"""
    print("\nTest 3: Agent decision making...")
    try:
        from core.agents.story_agent_layer import StoryAgentLayer
        from unittest.mock import AsyncMock, MagicMock

        mock_wrapper = AsyncMock()
        mock_llm = AsyncMock()

        agent_layer = StoryAgentLayer(mock_wrapper, mock_llm)

        # Mock context
        mock_context = MagicMock()
        mock_scene = MagicMock()
        mock_scene.location = "dark_forest"
        mock_scene.primary_npc = "guardian"
        mock_context.get_current_scene = MagicMock(return_value=mock_scene)

        decision = await agent_layer.make_decision(
            "test-session",
            "完成任務",
            "你成功完成了任務，獲得了經驗值",
            mock_context,
            {"hp": 100, "mp": 50, "level": 5}
        )

        assert decision is not None
        assert len(decision.tool_calls) > 0
        print(f"✓ Decision made: {len(decision.tool_calls)} tool calls")
        print(f"  Reasoning: {decision.reasoning}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_decision_execution():
    """Test Agent decision execution"""
    print("\nTest 4: Agent decision execution...")
    try:
        from core.agents.story_agent_layer import StoryAgentLayer, AgentDecision
        from unittest.mock import AsyncMock, MagicMock

        # Mock safety wrapper with success response
        mock_wrapper = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = {"success": True, "modified_flags": {}}
        mock_result.error = None
        mock_result.rollback_performed = False
        mock_wrapper.execute_tool = AsyncMock(return_value=mock_result)

        mock_llm = AsyncMock()

        agent_layer = StoryAgentLayer(mock_wrapper, mock_llm)

        decision = AgentDecision(
            decision_type="test",
            tool_calls=[
                {
                    "tool": "modify_world_state",
                    "params": {"flags": {"quest_started": True}}
                }
            ],
            reasoning="Test decision",
            confidence=0.9
        )

        result = await agent_layer.execute_decision("test-session", decision)

        assert result["overall_success"] is True
        assert len(result["tool_results"]) == 1
        print(f"✓ Decision executed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_damage_detection():
    """Test damage detection in narrative"""
    print("\nTest 5: Damage detection...")
    try:
        from core.agents.story_agent_layer import StoryAgentLayer
        from unittest.mock import AsyncMock, MagicMock

        mock_wrapper = AsyncMock()
        mock_llm = AsyncMock()

        agent_layer = StoryAgentLayer(mock_wrapper, mock_llm)

        damage = agent_layer._extract_damage_from_narrative("你受到 50 傷害")
        assert damage == 50

        damage_en = agent_layer._extract_damage_from_narrative("You took 30 damage")
        assert damage_en == 30

        print(f"✓ Damage detection works (50傷害, 30 damage)")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_story_engine_integration():
    """Test Agent integration in story engine"""
    print("\nTest 6: Story engine integration...")
    try:
        # This test just checks that the imports work
        from core.story.engine import StoryEngine, AGENT_AVAILABLE

        print(f"✓ Agent available in story engine: {AGENT_AVAILABLE}")

        if AGENT_AVAILABLE:
            # Create story engine with agent enabled
            # We won't actually run it, just check initialization
            print("  Agent layer can be integrated into story engine")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Agent Story Integration Test Suite")
    print("=" * 60)

    tests = [
        test_agent_layer_creation,
        test_intervention_detection,
        test_decision_making,
        test_decision_execution,
        test_damage_detection,
        test_story_engine_integration,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
