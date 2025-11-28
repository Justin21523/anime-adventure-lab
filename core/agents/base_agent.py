# core/agent/base_agent.py
"""
Base Agent Class
Provides foundation for intelligent task execution with tool usage
"""

import json
import logging
import time
import inspect
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
from .tool_registry import ToolRegistry
from .executor import SafeExecutor, AgentExecutor
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from . import prompts

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Agent response structure"""

    success: bool
    result: Any
    tools_used: List[str]
    execution_time_ms: float
    steps_taken: int = 0
    reasoning_chain: Optional[List[str]] = None
    error_message: Optional[str] = None

    def __getitem__(self, item: str):
        """Allow dict-style access for compatibility with older tests."""
        return getattr(self, item)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)


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
        name: Optional[str] = "base_agent",
        description: Optional[str] = "General purpose agent",
        max_iterations: int = 5,
        max_tools_per_iteration: int = 2,
        enable_reasoning: bool = True,
    ):
        self.name = name
        self.description = description
        self.tool_registry = ToolRegistry()
        self.executor = SafeExecutor()
        self.agent_executor = AgentExecutor()
        self.max_iterations = max_iterations
        self.max_tools_per_iteration = max_tools_per_iteration
        self.enable_reasoning = enable_reasoning
        self.state = AgentState.IDLE
        self.memory: Optional[AgentMemory] = None

    def _extract_expression_from_text(self, text: str) -> Optional[str]:
        """Extract a simple math expression from text."""
        import re

        math_pattern = r"[\d+\-*/\(\)\.\s]+"
        matches = re.findall(math_pattern, text)
        matches = [m.strip() for m in matches if m.strip()]
        if not matches:
            return None
        return max(matches, key=len)

    async def plan_task(
        self, task_description: str, **kwargs: Any
    ) -> List[str]:
        """
        Break down task into actionable steps
        Returns list of planned actions
        """
        # Simple default: single-step plan using the provided description
        return [task_description]

    async def select_tool(
        self,
        current_action: str,
        context_text: Optional[str] = None,
        reasoning_mode: str = "cot",
        history: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Select appropriate tool for current context
        Returns tool name and parameters, or None if no tool needed
        """
        text = current_action.lower()

        if any(keyword in text for keyword in ["calculate", "sum", "add", "multiply"]):
            expression = self._extract_expression_from_text(current_action)
            return {
                "tool_name": "calculator",
                "parameters": {"expression": expression or current_action},
            }

        if any(keyword in text for keyword in ["search", "find", "look up"]):
            return {"tool_name": "web_search", "parameters": {"query": current_action}}

        if any(keyword in text for keyword in ["list files", "directory", "ls"]):
            return {"tool_name": "file_ops", "parameters": {"operation": "list", "path": "."}}

        return None

    async def synthesize_result(self, memory: AgentMemory) -> str:
        """
        Synthesize final result from execution history
        """
        if not memory.step_history:
            return "No actions executed."

        last_result = memory.step_history[-1].get("result")
        if last_result is not None:
            return str(last_result)

        # Fallback summary
        summary_lines = [f"Tools used: {', '.join(memory.tools_used)}"]
        for step in memory.step_history[-3:]:
            summary_lines.append(f"Step {step['step']}: {step['type']} - {step['content']}")
        return "\n".join(summary_lines)

    async def execute_task(
        self,
        task_description: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        enable_chain_of_thought: bool = True,
        reasoning_mode: str = "cot",
    ) -> AgentResponse:
        """Execute a task using available tools"""
        start_time = time.time()
        tools_used: List[str] = []
        reasoning_chain: Optional[List[str]] = [] if enable_chain_of_thought else None
        task_expression_hint = self._extract_expression_from_text(task_description)

        try:
            self.state = AgentState.THINKING
            self.memory = AgentMemory(task_description=task_description, max_steps=self.max_iterations)
            self.memory.context.update(context or {})

            # Plan
            plan_steps = await self.plan_task(task_description)
            if not plan_steps:
                plan_steps = [task_description]
            if reasoning_chain is not None:
                reasoning_chain.append(f"Planned steps: {plan_steps}")

            # Execute planned steps
            for step in plan_steps[: self.max_iterations]:
                step_context = self.memory.get_context_summary()
                if context:
                    step_context += f"\nExtra context: {context}"

                history_text = (
                    "\n".join(
                        [f"{h['type']}: {h['content']}" for h in self.memory.step_history]
                    )
                    if self.memory.step_history
                    else ""
                )
                selection = await self.select_tool(
                    step,
                    context_text=step_context,
                    reasoning_mode=reasoning_mode,
                    history=history_text,
                )

                if selection:
                    tool_name = selection["tool_name"]
                    tool_params = selection.get("parameters", {}) or {}
                    if tool_name in {"web_search", "web_search_summary"}:
                        tool_params["query"] = step
                    if (
                        tool_name in {"calculator", "basic_math"}
                        and not tool_params.get("expression")
                        and task_expression_hint
                    ):
                        tool_params["expression"] = task_expression_hint
                    self.state = AgentState.USING_TOOL

                    exec_result = await self.agent_executor.execute_tool(
                        tool_name, tool_params
                    )

                    tools_used.append(tool_name)
                    self.memory.tools_used.append(tool_name)
                    self.memory.add_step(
                        "tool_usage",
                        f"{tool_name}({tool_params})",
                        exec_result.result if exec_result.success else exec_result.error,
                    )

                    if reasoning_chain is not None:
                        reasoning_chain.append(
                            f"Step '{step}' used {tool_name}: {exec_result.result}"
                        )

                    if not exec_result.success:
                        self.state = AgentState.ERROR
                        break
                else:
                    self.state = AgentState.THINKING
                    self.memory.add_step("reasoning", step)
                    if reasoning_chain is not None:
                        reasoning_chain.append(f"Reasoned: {step}")

            # Ensure at least one tool execution for calculable tasks
            if not tools_used and task_expression_hint:
                exec_result = await self.agent_executor.execute_tool(
                    "calculator", {"expression": task_expression_hint}
                )
                tools_used.append("calculator")
                self.memory.tools_used.append("calculator")
                self.memory.add_step(
                    "tool_usage",
                    f"calculator({task_expression_hint})",
                    exec_result.result if exec_result.success else exec_result.error,
                )

            # Synthesize final result
            final_result = await self.synthesize_result(self.memory)
            self.state = AgentState.COMPLETED
            execution_time = (time.time() - start_time) * 1000

            return AgentResponse(
                success=True,
                result=final_result,
                tools_used=tools_used,
                execution_time_ms=execution_time,
                steps_taken=self.memory.current_step if self.memory else 0,
                reasoning_chain=reasoning_chain,
            )

        except Exception as e:
            self.state = AgentState.ERROR
            execution_time = (time.time() - start_time) * 1000
            return AgentResponse(
                success=False,
                result=None,
                tools_used=tools_used,
                execution_time_ms=execution_time,
                steps_taken=self.memory.current_step if self.memory else 0,
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
        # Try lazy load if not provided
        if llm_adapter is None:
            try:
                from core.llm.adapter import get_llm_adapter

                llm_adapter = get_llm_adapter()
            except Exception as e:
                logger.warning(f"LLM adapter unavailable, using heuristics only: {e}")
                llm_adapter = None

        super().__init__(**kwargs)
        self.llm_adapter = llm_adapter

    async def plan_task(self, task_description: str) -> List[str]:
        """Plan task using LLM reasoning"""
        if not self.llm_adapter:
            return self._default_plan(task_description)

        try:
            plan_prompt = prompts.PLAN_TEMPLATE.format(
                task=task_description, context=self.memory.context if self.memory else {}
            )
            response = await self._llm_text(plan_prompt, max_tokens=200, temperature=0.3)
            if not response:
                return self._default_plan(task_description)

            lines = str(response).strip().split("\n")
            steps: List[str] = []
            for line in lines:
                line = line.strip()
                if not line or line.lower().startswith("steps"):
                    continue
                if line[0:2] in {"1.", "2.", "3.", "4."}:
                    line = line[2:].strip()
                if line.startswith("-"):
                    line = line[1:].strip()
                if line:
                    steps.append(line)

            return steps[:5] if steps else self._default_plan(task_description)

        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return self._default_plan(task_description)

    async def select_tool(
        self,
        current_action: str,
        context_text: Optional[str] = None,
        reasoning_mode: str = "cot",
        history: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select tool based on context analysis"""
        # Import here to avoid circular imports
        from .tool_registry import ToolRegistry

        registry = ToolRegistry()
        available_tools = registry.list_tools()
        if not available_tools:
            return None

        # Prefer LLM-based selection when available
        if self.llm_adapter:
            try:
                tools_info = "\n".join(
                    [
                        f"- {name}: {registry.get_tool_info(name).get('description', '')}"
                        for name in available_tools
                        if registry.get_tool_info(name)
                    ]
                )

                if reasoning_mode == "react":
                    select_prompt = prompts.REACT_TEMPLATE.format(
                        task=current_action,
                        context=context_text or "",
                        history=history or "",
                    )
                else:
                    select_prompt = prompts.TOOL_SELECT_TEMPLATE.format(
                        action=current_action,
                        context=context_text or "",
                        tools=tools_info,
                    )

                raw = await self._llm_text(select_prompt, max_tokens=180, temperature=0.2)
                choice = self._parse_tool_choice(raw) if raw else None
                if choice:
                    return choice
            except Exception as e:
                logger.warning(f"LLM tool selection failed, fallback to heuristics: {e}")

        # Heuristic fallback
        context_lower = current_action.lower()

        if any(
            word in context_lower for word in ["search", "find", "lookup", "information"]
        ):
            if "web_search" in available_tools:
                return {"tool_name": "web_search", "parameters": {"query": current_action}}
            if "web_search_summary" in available_tools:
                return {"tool_name": "web_search_summary", "parameters": {"query": current_action}}

        if any(
            word in context_lower for word in ["calculate", "math", "compute", "number"]
        ):
            import re

            math_pattern = r"[\d+\-*/\(\)\.\s]+"
            matches = re.findall(math_pattern, current_action)
            expression = max(matches, key=len).strip() if matches else "2+2"
            if "calculator" in available_tools:
                return {"tool_name": "calculator", "parameters": {"expression": expression}}
            if "basic_math" in available_tools:
                return {
                    "tool_name": "basic_math",
                    "parameters": {"a": expression, "b": 0, "op": "+"},
                }

        if any(
            word in context_lower for word in ["file", "save", "read", "write", "document"]
        ):
            if "file_list" in available_tools:
                return {"tool_name": "file_list", "parameters": {"path": "."}}
            if "file_read" in available_tools:
                return {"tool_name": "file_read", "parameters": {"file_path": ".", "max_lines": 50}}

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

        if self.llm_adapter:
            try:
                summary_prompt = prompts.SYNTHESIS_TEMPLATE.format(
                    task=memory.task_description,
                    context=memory.context,
                    steps="\n".join(reasoning_steps),
                    tool_outputs="\n".join(tool_results),
                )
                llm_summary = await self._llm_text(
                    summary_prompt, max_tokens=180, temperature=0.2
                )
                if llm_summary:
                    return str(llm_summary).strip()
            except Exception as e:
                logger.warning(f"LLM synthesis failed, using heuristic summary: {e}")

        return result_summary

    def _default_plan(self, task_description: str) -> List[str]:
        """Heuristic fallback planning"""
        steps = [f"Analyze task: {task_description}"]
        if "search" in task_description.lower() or "find" in task_description.lower():
            steps.append(f"Search for information: {task_description}")
        if any(char.isdigit() for char in task_description):
            steps.append("Calculate required numeric values")
        steps.append("Execute necessary actions step-by-step")
        steps.append("Summarize findings and outputs")
        return steps

    def _parse_tool_choice(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse JSON-like tool choice from LLM output"""
        try:
            raw_stripped = raw.strip()
            start = raw_stripped.find("{")
            end = raw_stripped.rfind("}")
            if start != -1 and end != -1:
                raw_stripped = raw_stripped[start : end + 1]
            data = json.loads(raw_stripped)
            tool_name = data.get("tool_name")
            if not tool_name:
                return None
            return {"tool_name": tool_name, "parameters": data.get("parameters", {})}
        except Exception:
            return None

    def _extract_expression_from_text(self, text: str) -> Optional[str]:
        """Extract a simple math expression from text"""
        import re

        math_pattern = r"[\d+\-*/\(\)\.\s]+"
        matches = re.findall(math_pattern, text)
        matches = [m.strip() for m in matches if m.strip()]
        if not matches:
            return None
        return max(matches, key=len)

    async def _llm_text(
        self, prompt: str, max_tokens: int = 200, temperature: float = 0.2
    ) -> Optional[str]:
        """Robust text generation helper supporting multiple adapter interfaces."""
        if not self.llm_adapter:
            return None

        async def _maybe_await(result):
            if inspect.iscoroutine(result):
                return await result
            return result

        # Preferred: generate_text(prompt, max_tokens, temperature)
        if hasattr(self.llm_adapter, "generate_text"):
            try:
                res = await _maybe_await(
                    self.llm_adapter.generate_text(
                        prompt, max_tokens=max_tokens, temperature=temperature
                    )
                )
                if res is not None:
                    return getattr(res, "content", None) or str(res)
            except Exception:
                pass

        # Fallback: chat(messages=[{role, content}])
        if hasattr(self.llm_adapter, "chat"):
            try:
                res = await _maybe_await(
                    self.llm_adapter.chat(
                        messages=[{"role": "user", "content": prompt}],
                        max_length=max_tokens,
                        temperature=temperature,
                    )
                )
                if res is not None:
                    if isinstance(res, dict):
                        return res.get("content") or res.get("message") or str(res)
                    return getattr(res, "content", None) or str(res)
            except Exception:
                pass

        # Fallback: generate(prompt, ...)
        if hasattr(self.llm_adapter, "generate"):
            try:
                res = await _maybe_await(
                    self.llm_adapter.generate(
                        prompt, max_length=max_tokens, temperature=temperature
                    )
                )
                if res is not None:
                    return getattr(res, "content", None) or str(res)
            except Exception:
                pass

        return None
