# core/agent/base_agent.py
"""
Base Agent Class
Provides foundation for intelligent task execution with tool usage
"""

import logging
import time
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
from .tool_registry import ToolRegistry
from .executor import SafeExecutor
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Agent response structure"""

    success: bool
    result: Any
    tools_used: List[str]
    execution_time_ms: float
    reasoning_chain: Optional[List[str]] = None
    error_message: Optional[str] = None


class AgentState(Enum):
    """Agent execution states"""

    IDLE = "idle"
    THINKING = "thinking"
    USING_TOOL = "using_tool"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentMemory:
    """Agent working memory"""

    task_description: str
    current_step: int = 0
    max_steps: int = 10
    tools_used: List[str] = field(default_factory=list)
    step_history: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step_type: str, content: str, result: Any = None):
        """Add step to history"""
        self.step_history.append(
            {
                "step": self.current_step,
                "type": step_type,
                "content": content,
                "result": result,
                "timestamp": time.time(),
            }
        )
        self.current_step += 1

    def get_context_summary(self) -> str:
        """Get condensed context for LLM"""
        if not self.step_history:
            return f"Task: {self.task_description}"

        recent_steps = self.step_history[-3:]  # Last 3 steps
        summary = f"Task: {self.task_description}\n"
        summary += f"Progress: {self.current_step}/{self.max_steps} steps\n"
        summary += "Recent actions:\n"

        for step in recent_steps:
            summary += f"- Step {step['step']}: {step['type']} - {step['content']}\n"

        return summary


class BaseAgent(ABC):
    """
    Base Agent for intelligent task execution
    Handles reasoning, tool selection, and multi-step planning
    """

    def __init__(
        self,
        name: Optional[str],
        description: Optional[str],
        max_iterations: int = 5,
        max_tools_per_iteration: int = 2,
        enable_reasoning: bool = True,
    ):
        self.name = name
        self.description = description
        self.tool_registry = ToolRegistry("configs/agent.yaml")
        self.executor = SafeExecutor()
        self.max_iterations = max_iterations
        self.max_tools_per_iteration = max_tools_per_iteration
        self.enable_reasoning = enable_reasoning
        self.state = AgentState.IDLE
        self.memory: Optional[AgentMemory] = None

    @abstractmethod
    async def plan_task(self, task_description: str) -> List[str]:
        """
        Break down task into actionable steps
        Returns list of planned actions
        """
        pass

    @abstractmethod
    async def select_tool(self, current_context: str) -> Optional[Dict[str, Any]]:
        """
        Select appropriate tool for current context
        Returns tool name and parameters, or None if no tool needed
        """
        pass

    @abstractmethod
    async def synthesize_result(self, memory: AgentMemory) -> str:
        """
        Synthesize final result from execution history
        """
        pass

    async def execute_task(
        self,
        task_description: str,
        parameters: Optional[Dict[str, Any]] = None,
        enable_chain_of_thought: bool = True,
    ) -> AgentResponse:
        """Execute a task using available tools"""

        import time

        start_time = time.time()
        tools_used = []
        reasoning_chain = [] if enable_chain_of_thought else None

        try:
            # Parse task and determine required tools
            if reasoning_chain is not None:
                reasoning_chain.append(f"Analyzing task: {task_description}")

            # Simple tool selection logic (can be enhanced with LLM)
            if (
                "calculate" in task_description.lower()
                or "math" in task_description.lower()
            ):
                tool_name = "calculator"
                tool_params = self._extract_calculation_params(task_description)
            elif (
                "search" in task_description.lower()
                or "find" in task_description.lower()
            ):
                tool_name = "web_search"
                tool_params = self._extract_search_params(task_description)
            elif (
                "file" in task_description.lower() or "read" in task_description.lower()
            ):
                tool_name = "file_ops"
                tool_params = self._extract_file_params(task_description)
            else:
                # Default to calculator for demo
                tool_name = "calculator"
                tool_params = {"expression": "2+2"}

            if reasoning_chain is not None:
                reasoning_chain.append(f"Selected tool: {tool_name}")

            # Execute tool
            result = await self.call_tool(tool_name, tool_params)
            tools_used.append(tool_name)

            execution_time = (time.time() - start_time) * 1000

            return AgentResponse(
                success=True,
                result=result,
                tools_used=tools_used,
                execution_time_ms=execution_time,
                reasoning_chain=reasoning_chain,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return AgentResponse(
                success=False,
                result=None,
                tools_used=tools_used,
                execution_time_ms=execution_time,
                reasoning_chain=reasoning_chain,
                error_message=str(e),
            )

    async def call_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a specific tool with parameters"""

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Execute tool in safe environment
        result = await self.executor.execute_tool(tool, parameters)
        return result

    def _extract_calculation_params(self, task: str) -> Dict[str, Any]:
        """Extract calculation parameters from task description"""
        # Simple extraction - can be enhanced with NLP
        import re

        # Look for mathematical expressions
        math_pattern = r"(\d+[\+\-\*/]\d+|\d+\.\d+[\+\-\*/]\d+\.\d+)"
        match = re.search(math_pattern, task)

        if match:
            return {"expression": match.group(1)}
        else:
            return {"expression": "2+2"}  # Default

    def _extract_search_params(self, task: str) -> Dict[str, Any]:
        """Extract search parameters from task description"""
        # Extract search query after "search for" or "find"
        import re

        patterns = [r"search for (.+)", r"find (.+)", r"look up (.+)"]

        for pattern in patterns:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                return {"query": match.group(1).strip()}

        return {"query": "AI technology"}  # Default

    def _extract_file_params(self, task: str) -> Dict[str, Any]:
        """Extract file operation parameters from task description"""
        # Simple file operation extraction
        import re

        # Look for file paths
        file_pattern = r"([^\s]+\.\w+)"
        match = re.search(file_pattern, task)

        if match:
            return {"operation": "read", "path": match.group(1)}
        else:
            return {"operation": "list", "path": "."}  # Default

    async def _execute_single_step(self, planned_action: str):
        """Execute a single planned step"""
        try:
            context = self.memory.get_context_summary()

            # Decide if tool usage is needed
            tool_selection = await self.select_tool(
                f"{context}\nCurrent action: {planned_action}"
            )

            if tool_selection:
                self.state = AgentState.USING_TOOL
                tool_name = tool_selection["tool_name"]
                tool_params = tool_selection["parameters"]

                # Import here to avoid circular imports
                from .executor import AgentExecutor

                executor = AgentExecutor()

                result = await executor.execute_tool(tool_name, tool_params)

                self.memory.tools_used.append(tool_name)
                self.memory.add_step(
                    "tool_usage",
                    f"Used {tool_name} with params: {tool_params}",
                    result.result if result.success else result.error,
                )

                if not result.success:
                    logger.warning(f"Tool {tool_name} failed: {result.error}")
            else:
                # Pure reasoning step
                self.memory.add_step("reasoning", planned_action)

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            self.state = AgentState.ERROR
            self.memory.add_step("error", f"Step failed: {str(e)}")


class SimpleReasoningAgent(BaseAgent):
    """
    Simple agent implementation with basic reasoning capabilities
    Uses LLM for planning and tool selection
    """

    def __init__(self, llm_adapter=None, **kwargs):
        super().__init__(**kwargs)
        self.llm_adapter = llm_adapter

    async def plan_task(self, task_description: str) -> List[str]:
        """Plan task using LLM reasoning"""
        if not self.llm_adapter:
            # Fallback to simple heuristic planning
            return [
                f"Analyze task: {task_description}",
                "Identify required tools and information",
                "Execute necessary actions",
                "Synthesize results",
            ]

        planning_prompt = f"""
        Task: {task_description}

        Break this task into 2-4 specific, actionable steps.
        Each step should be clear and focused.

        Steps:
        """

        try:
            response = await self.llm_adapter.generate_text(
                planning_prompt, max_tokens=200, temperature=0.3
            )

            # Parse response into steps
            lines = response.strip().split("\n")
            steps = []
            for line in lines:
                line = line.strip()
                if (
                    line
                    and not line.startswith("Task:")
                    and not line.startswith("Steps:")
                ):
                    # Remove numbering if present
                    if line.startswith(("1.", "2.", "3.", "4.", "-", "*")):
                        line = line[2:].strip()
                    if line:
                        steps.append(line)

            return steps[:4]  # Limit to 4 steps

        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return await self.plan_task(task_description)  # Fallback

    async def select_tool(self, current_context: str) -> Optional[Dict[str, Any]]:
        """Select tool based on context analysis"""
        # Import here to avoid circular imports
        from .tool_registry import ToolRegistry

        registry = ToolRegistry()
        available_tools = registry.list_tools()

        if not available_tools:
            return None

        # Simple keyword-based tool selection
        context_lower = current_context.lower()

        # Tool selection heuristics
        if any(
            word in context_lower for word in ["calculate", "math", "compute", "number"]
        ):
            if "calculator" in available_tools:
                # Extract expression from context
                # This is a simple heuristic - in practice, use LLM for better extraction
                import re

                math_pattern = r"[\d+\-*/\(\)\.\s]+"
                matches = re.findall(math_pattern, current_context)
                if matches:
                    expression = max(matches, key=len).strip()
                    return {
                        "tool_name": "calculator",
                        "parameters": {"expression": expression},
                    }

        if any(
            word in context_lower
            for word in ["search", "find", "lookup", "information"]
        ):
            if "web_search" in available_tools:
                # Extract search query
                words = current_context.split()
                # Simple heuristic to extract query
                query_words = [w for w in words if w.isalnum() and len(w) > 2]
                if query_words:
                    query = " ".join(query_words[-3:])  # Last 3 meaningful words
                    return {
                        "tool_name": "web_search",
                        "parameters": {"query": query, "max_results": 3},
                    }

        if any(
            word in context_lower
            for word in ["file", "save", "read", "write", "document"]
        ):
            if "file_ops" in available_tools:
                return {
                    "tool_name": "file_ops",
                    "parameters": {"operation": "list", "path": "."},
                }

        return None

    async def synthesize_result(self, memory: AgentMemory) -> str:
        """Synthesize final result from execution history"""
        if not memory.step_history:
            return f"Task '{memory.task_description}' completed with no specific actions taken."

        # Collect successful tool results
        tool_results = []
        reasoning_steps = []

        for step in memory.step_history:
            if step["type"] == "tool_usage" and step.get("result"):
                tool_results.append(f"- {step['content']}: {step['result']}")
            elif step["type"] == "reasoning":
                reasoning_steps.append(f"- {step['content']}")

        result_summary = f"Task: {memory.task_description}\n"
        result_summary += f"Completed in {memory.current_step} steps.\n"

        if reasoning_steps:
            result_summary += "\nReasoning:\n" + "\n".join(reasoning_steps)

        if tool_results:
            result_summary += "\nTool Results:\n" + "\n".join(tool_results)

        return result_summary
