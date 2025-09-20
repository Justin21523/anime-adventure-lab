# tests/test_agent.py
"""
Tests for Agent system
"""

import pytest
import asyncio
import json
from pathlib import Path
from core.agent import BaseAgent, ToolRegistry, SafeExecutor
from core.agent.tools import calculator, web_search, file_ops


class TestCalculatorTool:
    """Test calculator tool"""

    def test_basic_arithmetic(self):
        result = calculator.calculate("2 + 3")
        assert result["result"] == 5
        assert result["type"] == "int"

    def test_floating_point(self):
        result = calculator.calculate("3.14 * 2")
        assert abs(result["result"] - 6.28) < 0.01
        assert result["type"] == "float"

    def test_mathematical_functions(self):
        result = calculator.calculate("sqrt(16)")
        assert result["result"] == 4

    def test_complex_expression(self):
        result = calculator.calculate("(2 + 3) * 4 - 1")
        assert result["result"] == 19

    def test_error_handling(self):
        result = calculator.calculate("1 / 0")
        assert "error" in result
        assert "Division by zero" in result["error"]

    def test_invalid_expression(self):
        result = calculator.calculate("invalid expression")
        assert "error" in result


@pytest.mark.asyncio
class TestWebSearchTool:
    """Test web search tool"""

    async def test_basic_search(self):
        result = await web_search.search("Python programming")
        assert "query" in result
        assert "results" in result
        assert result["query"] == "Python programming"

    async def test_max_results_limit(self):
        result = await web_search.search("AI technology", max_results=2)
        assert len(result["results"]) <= 2

    async def test_search_with_special_characters(self):
        result = await web_search.search("C++ & Java")
        assert result["query"] == "C++ & Java"


class TestFileOpsTool:
    """Test file operations tool"""

    def test_list_current_directory(self):
        result = file_ops.execute("list", ".")
        assert result["operation"] == "list"
        assert "items" in result
        assert isinstance(result["items"], list)

    def test_file_info(self):
        # Create a test file
        test_file = Path("test_file.txt")
        test_file.write_text("test content")

        try:
            result = file_ops.execute("info", str(test_file))
            assert result["operation"] == "info"
            assert result["type"] == "file"
            assert result["name"] == "test_file.txt"
        finally:
            test_file.unlink(missing_ok=True)

    def test_read_file(self):
        # Create a test file
        test_file = Path("test_read.txt")
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        try:
            result = file_ops.execute("read", str(test_file))
            assert result["operation"] == "read"
            assert result["content"] == test_content
        finally:
            test_file.unlink(missing_ok=True)

    def test_invalid_operation(self):
        result = file_ops.execute("invalid", ".")
        assert "error" in result
        assert "Unsupported operation" in result["error"]

    def test_security_path_validation(self):
        # Try to access parent directory
        result = file_ops.execute("list", "../")
        assert "error" in result
        assert "Access denied" in result["error"]


class TestToolRegistry:
    """Test tool registry functionality"""

    def test_load_default_tools(self):
        registry = ToolRegistry()
        tools = registry.list_tools()
        assert "calculator" in tools
        assert "web_search" in tools
        assert "file_ops" in tools

    def test_get_tool(self):
        registry = ToolRegistry()
        calc_tool = registry.get_tool("calculator")
        assert calc_tool is not None
        assert calc_tool.name == "calculator"
        assert "expression" in calc_tool.parameters

    def test_validate_parameters(self):
        registry = ToolRegistry()
        calc_tool = registry.get_tool("calculator")

        # Valid parameters
        params = registry.validate_parameters(calc_tool, {"expression": "2+2"})
        assert params["expression"] == "2+2"

        # Missing required parameter
        with pytest.raises(ValueError):
            registry.validate_parameters(calc_tool, {})


@pytest.mark.asyncio
class TestSafeExecutor:
    """Test safe executor functionality"""

    async def test_execute_calculator_tool(self):
        executor = SafeExecutor()
        registry = ToolRegistry()
        tool = registry.get_tool("calculator")

        result = await executor.execute_tool(tool, {"expression": "5 * 6"})
        assert result["success"] is True
        assert result["result"]["result"] == 30

    async def test_execute_with_timeout(self):
        executor = SafeExecutor()
        registry = ToolRegistry()
        tool = registry.get_tool("calculator")
        tool.timeout_seconds = 0.1  # Very short timeout

        # This should still work for simple calculations
        result = await executor.execute_tool(tool, {"expression": "1+1"})
        assert result["success"] is True

    async def test_parameter_validation(self):
        executor = SafeExecutor()
        registry = ToolRegistry()
        tool = registry.get_tool("calculator")

        # Missing required parameter
        result = await executor.execute_tool(tool, {})
        assert result["success"] is False
        assert "error" in result


@pytest.mark.asyncio
class TestBaseAgent:
    """Test base agent functionality"""

    async def test_agent_initialization(self):
        agent = BaseAgent()
        assert agent.tool_registry is not None
        assert agent.executor is not None
        assert agent.max_iterations == 5

    async def test_call_calculator_tool(self):
        agent = BaseAgent()
        result = await agent.call_tool("calculator", {"expression": "10 + 5"})
        assert result["success"] is True
        assert result["result"]["result"] == 15

    async def test_execute_math_task(self):
        agent = BaseAgent()
        response = await agent.execute_task("Calculate 7 * 8")
        assert response.success is True
        assert "calculator" in response.tools_used

    async def test_execute_search_task(self):
        agent = BaseAgent()
        response = await agent.execute_task("Search for Python tutorials")
        assert response.success is True
        assert "web_search" in response.tools_used

    async def test_execute_file_task(self):
        agent = BaseAgent()
        response = await agent.execute_task("List files in current directory")
        assert response.success is True
        assert "file_ops" in response.tools_used

    async def test_invalid_tool(self):
        agent = BaseAgent()
        with pytest.raises(ValueError):
            await agent.call_tool("nonexistent_tool", {})


# Integration tests
@pytest.mark.asyncio
class TestAgentIntegration:
    """Integration tests for agent system"""

    async def test_complete_workflow(self):
        """Test complete agent workflow"""
        agent = BaseAgent()

        # Test multiple tasks
        tasks = [
            "Calculate the square of 12",
            "Search for machine learning",
            "List current directory contents",
        ]

        for task in tasks:
            response = await agent.execute_task(task)
            assert response.success is True
            assert len(response.tools_used) > 0
            assert response.execution_time_ms > 0

    async def test_reasoning_chain(self):
        """Test reasoning chain functionality"""
        agent = BaseAgent()

        response = await agent.execute_task(
            "Calculate 15 * 20 + 100", enable_chain_of_thought=True
        )

        assert response.success is True
        assert response.reasoning_chain is not None
        assert len(response.reasoning_chain) > 0


# Smoke tests for API endpoints
def test_api_endpoints_smoke():
    """Smoke tests for API endpoints (requires running server)"""
    import requests

    base_url = "http://localhost:8000/api/v1"

    try:
        # Test health endpoint first
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip("API server not running")

        # Test agent endpoints
        endpoints = [
            ("GET", "/agent/tools"),
            ("GET", "/agent/status"),
            (
                "POST",
                "/agent/call",
                {"tool_name": "calculator", "parameters": {"expression": "2+2"}},
            ),
            (
                "POST",
                "/agent/task",
                {
                    "task_description": "Calculate 5*6",
                    "parameters": {"max_iterations": 3},
                },
            ),
        ]

        for method, endpoint, *data in endpoints:
            url = f"{base_url}{endpoint}"
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=data[0] if data else {}, timeout=10)

            assert response.status_code in [200, 201], f"Failed: {method} {endpoint}"

    except requests.exceptions.RequestException:
        pytest.skip("API server not accessible")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
