# core/agent/executor.py
"""
Agent Tool Executor
Handles secure tool execution with timeout, error handling, and result formatting
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
import traceback
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import functools

from .tool_registry import ToolRegistry, ToolMetadata

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of tool execution"""

    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    tool_name: str = ""
    parameters: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "tool_name": self.tool_name,
            "parameters": self.parameters or {},
        }


class AgentExecutor:
    """
    Secure tool executor for agent tasks
    Handles timeout, sandboxing, and error recovery
    """

    def __init__(self, max_concurrent_tools: int = 3):
        self.registry = ToolRegistry()
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_tools)
        self.active_executions: Dict[str, asyncio.Task] = {}

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout_override: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute a tool with given parameters

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            timeout_override: Optional timeout override in seconds

        Returns:
            ExecutionResult with success status and result/error
        """
        start_time = time.time()

        try:
            # Validate tool exists
            if not self.registry.is_tool_available(tool_name):
                return ExecutionResult(
                    success=False,
                    error=f"Tool '{tool_name}' not found",
                    tool_name=tool_name,
                    parameters=parameters,
                )

            # Get tool metadata and function
            tool_metadata = self.registry.get_tool(tool_name)
            tool_function = self.registry.get_function(tool_name)

            if not tool_metadata or not tool_function:
                return ExecutionResult(
                    success=False,
                    error=f"Tool '{tool_name}' not properly registered",
                    tool_name=tool_name,
                    parameters=parameters,
                )

            # Validate parameters
            if not self.registry.validate_parameters(tool_name, parameters):
                return ExecutionResult(
                    success=False,
                    error=f"Invalid parameters for tool '{tool_name}'",
                    tool_name=tool_name,
                    parameters=parameters,
                )

            # Prepare execution
            timeout = timeout_override or tool_metadata.timeout_seconds
            execution_id = f"{tool_name}_{int(time.time() * 1000)}"

            logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")

            # Execute with timeout
            if tool_metadata.is_async:
                result = await asyncio.wait_for(
                    self._execute_async_tool(tool_function, parameters), timeout=timeout
                )
            else:
                result = await asyncio.wait_for(
                    self._execute_sync_tool(tool_function, parameters), timeout=timeout
                )

            execution_time = (time.time() - start_time) * 1000

            logger.info(f"Tool '{tool_name}' completed in {execution_time:.2f}ms")

            return ExecutionResult(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                tool_name=tool_name,
                parameters=parameters,
            )

        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Tool '{tool_name}' timed out after {timeout}s"
            logger.error(error_msg)

            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time_ms=execution_time,
                tool_name=tool_name,
                parameters=parameters,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time_ms=execution_time,
                tool_name=tool_name,
                parameters=parameters,
            )

    async def _execute_async_tool(self, tool_function, parameters: Dict[str, Any]):
        """Execute async tool function"""
        try:
            # Call with keyword arguments
            return await tool_function(**parameters)
        except TypeError as e:
            # Handle parameter mismatch
            if "unexpected keyword argument" in str(e):
                # Try with positional arguments
                return await tool_function(*parameters.values())
            raise

    async def _execute_sync_tool(self, tool_function, parameters: Dict[str, Any]):
        """Execute sync tool function in thread pool"""
        loop = asyncio.get_event_loop()
        try:
            # Wrap sync function for thread execution
            func = functools.partial(tool_function, **parameters)
            return await loop.run_in_executor(self.thread_pool, func)
        except TypeError as e:
            # Handle parameter mismatch
            if "unexpected keyword argument" in str(e):
                # Try with positional arguments
                func = functools.partial(tool_function, *parameters.values())
                return await loop.run_in_executor(self.thread_pool, func)
            raise

    async def execute_multiple_tools(
        self, tool_calls: List[Dict[str, Any]], max_concurrent: int = 3
    ) -> List[ExecutionResult]:
        """
        Execute multiple tools concurrently

        Args:
            tool_calls: List of {"tool_name": str, "parameters": dict}
            max_concurrent: Maximum concurrent executions

        Returns:
            List of ExecutionResults in order of input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(tool_call):
            async with semaphore:
                return await self.execute_tool(
                    tool_call["tool_name"], tool_call["parameters"]
                )

        tasks = [execute_with_semaphore(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed ExecutionResults
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ExecutionResult(
                        success=False,
                        error=f"Execution failed: {str(result)}",
                        tool_name=tool_calls[i].get("tool_name", "unknown"),
                        parameters=tool_calls[i].get("parameters", {}),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def get_active_executions(self) -> List[str]:
        """Get list of currently active execution IDs"""
        return list(self.active_executions.keys())

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution"""
        if execution_id in self.active_executions:
            task = self.active_executions[execution_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.active_executions[execution_id]
            return True
        return False

    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)


class SafeExecutor:
    """Safe execution environment for agent tools"""

    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.tool_registry = None  # Will be set when needed

    async def execute_tool(
        self, tool: ToolMetadata, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool safely with timeout and sandboxing"""

        try:
            # Load tool registry if not already loaded
            if self.tool_registry is None:
                self.tool_registry = ToolRegistry()

            # Validate parameters
            validated_params = self.tool_registry.validate_parameters(tool, parameters)

            # Get tool function
            tool_function = self.tool_registry.get_tool_function(tool)

            # Execute with timeout
            if tool.requires_sandbox:
                result = await self._execute_sandboxed(
                    tool_function, validated_params, tool.timeout_seconds
                )
            else:
                result = await self._execute_normal(
                    tool_function, validated_params, tool.timeout_seconds
                )

            return {
                "success": True,
                "result": result,
                "tool_name": tool.name,
                "parameters_used": validated_params,
            }

        except asyncio.TimeoutError:
            logger.error(f"Tool {tool.name} execution timed out")
            return {
                "success": False,
                "error": f"Tool execution timed out after {tool.timeout_seconds} seconds",
                "tool_name": tool.name,
            }

        except Exception as e:
            logger.error(f"Tool {tool.name} execution failed: {e}")
            return {"success": False, "error": str(e), "tool_name": tool.name}

    async def _execute_normal(
        self, function: Callable, parameters: Dict[str, Any], timeout: int
    ) -> Any:
        """Execute function normally with timeout"""

        loop = asyncio.get_event_loop()

        # Run in thread pool to avoid blocking
        future = loop.run_in_executor(self.executor, lambda: function(**parameters))

        # Wait with timeout
        result = await asyncio.wait_for(future, timeout=timeout)
        return result

    async def _execute_sandboxed(
        self, function: Callable, parameters: Dict[str, Any], timeout: int
    ) -> Any:
        """Execute function in sandboxed environment"""

        # For now, just execute normally but with additional safety checks
        # In production, you might want to use containers or restricted environments

        # Basic safety checks
        self._validate_sandbox_safety(parameters)

        return await self._execute_normal(function, parameters, timeout)

    def _validate_sandbox_safety(self, parameters: Dict[str, Any]):
        """Basic safety validation for sandboxed execution"""

        for key, value in parameters.items():
            if isinstance(value, str):
                # Check for potentially dangerous patterns
                dangerous_patterns = [
                    "import os",
                    "import subprocess",
                    "exec(",
                    "eval(",
                    "__import__",
                    "open(",
                    "file(",
                    "input(",
                ]

                value_lower = value.lower()
                for pattern in dangerous_patterns:
                    if pattern in value_lower:
                        raise ValueError(
                            f"Potentially unsafe pattern detected: {pattern}"
                        )

                # Check for file system access patterns
                if any(
                    x in value for x in ["../", "../", "~/", "/etc/", "/usr/", "/var/"]
                ):
                    # Allow only relative paths in current directory
                    if not value.startswith("./") and "/" in value:
                        raise ValueError(
                            "File system access outside current directory not allowed"
                        )

    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
