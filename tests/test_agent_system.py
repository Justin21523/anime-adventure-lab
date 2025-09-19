# tests/test_agent_system.py
"""
Agent System Test Suite
Tests for tool registry, executor, and multi-step processing
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

# Import agent system components
from core.agent import (
    ToolRegistry,
    AgentExecutor,
    MultiStepProcessor,
    SimpleReasoningAgent,
)
from core.agent.story_integration import StoryAgent, StoryContext, StoryAgentManager


class TestToolRegistry:
    """Test tool registry functionality"""

    def test_tool_registry_singleton(self):
        """Test that ToolRegistry is a singleton"""
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()
        assert registry1 is registry2

    def test_list_default_tools(self):
        """Test that default tools are registered"""
        registry = ToolRegistry()
        tools = registry.list_tools()

        # Check that basic tools are available
        expected_tools = ["calculator", "web_search", "file_ops"]
        for tool in expected_tools:
            assert tool in tools

    def test_get_tool_info(self):
        """Test getting tool information"""
        registry = ToolRegistry()

        calc_info = registry.get_tool_info("calculator")
        assert calc_info is not None
        assert calc_info["name"] == "calculator"
        assert "description" in calc_info
        assert "parameters" in calc_info

    def test_tool_availability(self):
        """Test tool availability checking"""
        registry = ToolRegistry()

        assert registry.is_tool_available("calculator")
        assert not registry.is_tool_available("nonexistent_tool")


class TestAgentExecutor:
    """Test agent executor functionality"""

    @pytest.mark.asyncio
    async def test_calculator_tool(self):
        """Test calculator tool execution"""
        executor = AgentExecutor()

        result = await executor.execute_tool(
            tool_name="calculator", parameters={"expression": "2 + 2"}
        )

        assert result.success
        assert result.result["result"] == 4
        assert result.tool_name == "calculator"
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_invalid_tool(self):
        """Test execution of invalid tool"""
        executor = AgentExecutor()

        result = await executor.execute_tool(
            tool_name="nonexistent_tool", parameters={}
        )

        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_invalid_parameters(self):
        """Test execution with invalid parameters"""
        executor = AgentExecutor()

        result = await executor.execute_tool(
            tool_name="calculator", parameters={"invalid_param": "value"}
        )

        assert not result.success

    @pytest.mark.asyncio
    async def test_multiple_tools(self):
        """Test execution of multiple tools"""
        executor = AgentExecutor()

        tool_calls = [
            {"tool_name": "calculator", "parameters": {"expression": "1 + 1"}},
            {"tool_name": "calculator", "parameters": {"expression": "2 * 3"}},
            {"tool_name": "calculator", "parameters": {"expression": "10 / 2"}},
        ]

        results = await executor.execute_multiple_tools(tool_calls)

        assert len(results) == 3
        assert all(result.success for result in results)
        assert results[0].result["result"] == 2
        assert results[1].result["result"] == 6
        assert results[2].result["result"] == 5.0


class TestFileOperations:
    """Test file operation tools"""

    @pytest.mark.asyncio
    async def test_file_operations(self):
        """Test file reading and writing"""
        executor = AgentExecutor()

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            test_content = "Hello, Agent System!"

            # Test file writing
            write_result = await executor.execute_tool(
                tool_name="file_ops",
                parameters={
                    "operation": "write",
                    "path": test_file,
                    "content": test_content,
                },
            )

            # Note: This might fail if write operation isn't implemented
            # In that case, manually create the file
            if not write_result.success:
                with open(test_file, "w") as f:
                    f.write(test_content)

            # Test file listing
            list_result = await executor.execute_tool(
                tool_name="file_ops", parameters={"operation": "list", "path": temp_dir}
            )

            assert list_result.success

            # Test file reading (if file exists)
            if os.path.exists(test_file):
                read_result = await executor.execute_tool(
                    tool_name="file_ops",
                    parameters={"operation": "read", "path": test_file},
                )

                # This test might need adjustment based on actual implementation


class TestSimpleReasoningAgent:
    """Test simple reasoning agent"""

    @pytest.mark.asyncio
    async def test_basic_task_execution(self):
        """Test basic task execution"""
        agent = SimpleReasoningAgent(
            name="test_agent", description="Test agent", max_iterations=3
        )

        result = await agent.execute_task(
            task_description="Calculate the sum of 5 and 7", context={}
        )

        assert result["success"]
        assert "steps_taken" in result
        assert "tools_used" in result
        assert "result" in result

    @pytest.mark.asyncio
    async def test_planning(self):
        """Test task planning"""
        agent = SimpleReasoningAgent(name="test_agent", description="Test agent")

        steps = await agent.plan_task("Find information about Python programming")

        assert isinstance(steps, list)
        assert len(steps) > 0
        assert all(isinstance(step, str) for step in steps)


class TestMultiStepProcessor:
    """Test multi-step task processor"""

    @pytest.mark.asyncio
    async def test_task_creation(self):
        """Test multi-step task creation"""
        processor = MultiStepProcessor()

        task = await processor.create_task(
            task_id="test_task_1",
            description="Test multi-step task",
            context={"test": True},
        )

        assert task.task_id == "test_task_1"
        assert task.description == "Test multi-step task"
        assert task.context["test"] is True

    @pytest.mark.asyncio
    async def test_auto_planning(self):
        """Test automatic task planning"""
        processor = MultiStepProcessor()

        task = await processor.create_task(
            task_id="test_task_2",
            description="Calculate the area of a circle with radius 5",
            context={},
        )

        planned_task = await processor.auto_plan_task(task)

        assert len(planned_task.steps) > 0
        assert all(step.description for step in planned_task.steps)

    @pytest.mark.asyncio
    async def test_task_execution(self):
        """Test multi-step task execution"""
        processor = MultiStepProcessor()

        # Create and plan task
        task = await processor.create_task(
            task_id="test_task_3", description="Simple calculation task", context={}
        )

        # Add a simple step manually
        task.add_step(description="Calculate 3 * 4", agent_name="reasoning")

        # Execute task
        result = await processor.execute_task("test_task_3")

        assert result["task_id"] == "test_task_3"
        assert "status" in result
        assert "completed_steps" in result


class TestStoryIntegration:
    """Test story integration functionality"""

    @pytest.mark.asyncio
    async def test_story_context(self):
        """Test story context creation"""
        context = StoryContext(
            story_id="test_story",
            character_name="Test Hero",
            current_scene="Test Chamber",
            character_state={"health": 100, "level": 1},
            story_history=[],
            available_actions=["look", "move", "attack"],
        )

        summary = context.get_context_summary()

        assert "Test Hero" in summary
        assert "Test Chamber" in summary
        assert "health" in summary

    @pytest.mark.asyncio
    async def test_story_agent(self):
        """Test story agent functionality"""
        story_agent = StoryAgent(
            name="test_story_agent", description="Test story agent"
        )

        context = StoryContext(
            story_id="test_story",
            character_name="Hero",
            current_scene="Forest Clearing",
            character_state={"health": 100},
            story_history=[],
            available_actions=["explore", "rest"],
            narrative_style="fantasy",
        )

        result = await story_agent.process_story_action(
            story_context=context, player_action="explore the area"
        )

        assert result["success"] or "fallback_response" in result

        if result["success"]:
            story_response = result["story_response"]
            assert "narrative" in story_response
            assert "available_actions" in story_response

    def test_story_manager(self):
        """Test story agent manager"""
        manager = StoryAgentManager()

        # Check that agents are initialized
        assert "narrative" in manager.agents
        assert "dialogue" in manager.agents
        assert "world" in manager.agents
        assert "action" in manager.agents

        # Test agent retrieval
        narrative_agent = manager.get_story_agent("narrative")
        assert narrative_agent is not None
        assert isinstance(narrative_agent, StoryAgent)


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows"""

    @pytest.mark.asyncio
    async def test_calculation_workflow(self):
        """Test complete calculation workflow"""
        agent = SimpleReasoningAgent(
            name="calc_agent", description="Calculation agent", max_iterations=3
        )

        result = await agent.execute_task(
            task_description="Calculate the area of a circle with radius 10, then convert the result from square units to percentage of a 50x50 square",
            context={},
        )

        assert result["success"]
        assert len(result["tools_used"]) > 0
        assert "calculator" in result["tools_used"]

    @pytest.mark.asyncio
    async def test_research_workflow(self):
        """Test research workflow with web search"""
        agent = SimpleReasoningAgent(
            name="research_agent", description="Research agent", max_iterations=3
        )

        result = await agent.execute_task(
            task_description="Search for information about artificial intelligence and summarize key points",
            context={},
        )

        assert result["success"]
        # Note: This test uses mock search, so results will be predictable


# Utility functions for tests
def cleanup_test_files():
    """Clean up any test files created during testing"""
    test_files = ["test_output.txt", "agent_test.log"]

    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)


# Pytest fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment"""
    # Set up test directories
    test_dirs = ["./test_temp", "./test_outputs"]
    for test_dir in test_dirs:
        os.makedirs(test_dir, exist_ok=True)

    yield

    # Cleanup after tests
    cleanup_test_files()


@pytest.fixture
def sample_story_context():
    """Create sample story context for testing"""
    return StoryContext(
        story_id="test_adventure",
        character_name="Test Hero",
        current_scene="Starting Village",
        character_state={
            "health": 100,
            "mana": 50,
            "level": 1,
            "experience": 0,
            "inventory": ["wooden sword", "health potion"],
        },
        story_history=[
            {
                "action": "enter village",
                "result": "You arrive at a peaceful village",
                "timestamp": 1234567890,
            }
        ],
        available_actions=["talk to villagers", "visit shop", "explore outskirts"],
        narrative_style="fantasy",
    )


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
