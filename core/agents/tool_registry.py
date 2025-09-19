# core/agent/tool_registry.py
"""
Tool Registry System
Manages registration, discovery, and metadata for agent tools
"""

import logging
import yaml
import importlib
from typing import Dict, Any, List, Optional, Callable, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from pathlib import Path
import inspect
from functools import wraps

logger = logging.getLogger(__name__)


class ToolParameter(BaseModel):
    """Tool parameter definition"""

    type: str
    description: str
    default: Optional[Any] = None
    required: bool = True


@dataclass
class ToolMetadata:
    """Tool metadata and configuration"""

    name: str
    description: str
    function_path: str
    parameters: Dict[str, ToolParameter] = field(default_factory=dict)
    category: str = "general"
    requires_auth: bool = False
    is_async: bool = False
    timeout_seconds: int = 30
    requires_sandbox: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "function_path": self.function_path,
            "parameters": self.parameters,
            "category": self.category,
            "requires_auth": self.requires_auth,
            "is_async": self.is_async,
            "timeout_seconds": self.timeout_seconds,
            "requires_sandbox": self.requires_sandbox,
        }


class ToolRegistry:
    """
    Central registry for agent tools
    Supports YAML configuration and runtime registration
    """

    _instance = None
    _tools: Dict[str, ToolMetadata] = {}
    _functions: Dict[str, Callable] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._load_from_config()

    def _load_from_config(self):
        """Load tools from YAML configuration"""
        try:
            from core.config import get_config

            config = get_config()
            agent_config = config.get_agent_config()

            if not agent_config or "tools" not in agent_config:
                logger.warning("No agent tools configuration found")
                self._register_default_tools()
                return

            for tool_config in agent_config["tools"]:
                try:
                    metadata = ToolMetadata(
                        name=tool_config["name"],
                        description=tool_config["description"],
                        function_path=tool_config["function"],
                        parameters=tool_config.get("parameters", {}),
                        category=tool_config.get("category", "general"),
                        requires_auth=tool_config.get("requires_auth", False),
                        timeout_seconds=tool_config.get("timeout_seconds", 30),
                    )

                    # Load and register function
                    function = self._load_function(metadata.function_path)
                    if function:
                        metadata.is_async = inspect.iscoroutinefunction(function)
                        self._tools[metadata.name] = metadata
                        self._functions[metadata.name] = function
                        logger.info(f"Registered tool: {metadata.name}")

                except Exception as e:
                    logger.error(
                        f"Failed to register tool {tool_config.get('name', 'unknown')}: {e}"
                    )

        except Exception as e:
            logger.error(f"Failed to load tool configuration: {e}")
            self._register_default_tools()

    def _load_function(self, function_path: str) -> Optional[Callable]:
        """Load function from module path"""
        try:
            # Split module and function name
            if "." not in function_path:
                logger.error(f"Invalid function path: {function_path}")
                return None

            module_path, function_name = function_path.rsplit(".", 1)

            # Handle relative imports for tools
            if module_path.startswith("tools."):
                module_path = f"core.agent.{module_path}"

            module = importlib.import_module(module_path)
            function = getattr(module, function_name)

            return function

        except Exception as e:
            logger.error(f"Failed to load function {function_path}: {e}")
            return None

    def _register_default_tools(self):
        """Register default tools if no configuration found"""
        logger.info("Registering default tools")

        # Register built-in tools
        from .tools.calculator import calculate
        from .tools.web_search import search
        from .tools.file_ops import list_files

        self.register_function(
            name="calculator",
            function=calculate,
            description="Perform mathematical calculations",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                }
            },
        )

        self.register_function(
            name="web_search",
            function=search,
            description="Search the web for information",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of results",
                },
            },
        )

        self.register_function(
            name="file_ops",
            function=list_files,
            description="File system operations",
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation to perform (list, read, write)",
                },
                "path": {"type": "string", "description": "File or directory path"},
            },
        )

    def register_function(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        category: str = "general",
        requires_auth: bool = False,
        timeout_seconds: int = 30,
    ):
        """Register a function as a tool"""
        metadata = ToolMetadata(
            name=name,
            description=description,
            function_path=f"{function.__module__}.{function.__name__}",
            parameters=parameters or {},
            category=category,
            requires_auth=requires_auth,
            is_async=inspect.iscoroutinefunction(function),
            timeout_seconds=timeout_seconds,
        )

        self._tools[name] = metadata
        self._functions[name] = function
        logger.info(f"Registered function as tool: {name}")

    def register_tool(self, metadata: ToolMetadata, function: Callable):
        """Register a tool with metadata"""
        self._tools[metadata.name] = metadata
        self._functions[metadata.name] = function
        logger.info(f"Registered tool: {metadata.name}")

    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name"""
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable]:
        """Get tool function by name"""
        return self._functions.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self._tools.keys())

    def list_tools_by_category(self, category: str) -> List[str]:
        """List tools in specific category"""
        return [
            name
            for name, metadata in self._tools.items()
            if metadata.category == category
        ]

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed tool information"""
        metadata = self.get_tool(name)
        if not metadata:
            return None

        return metadata.to_dict()

    def get_all_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all tools"""
        return {name: metadata.to_dict() for name, metadata in self._tools.items()}

    def is_tool_available(self, name: str) -> bool:
        """Check if tool is available"""
        return name in self._tools and name in self._functions

    def validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for a tool"""
        metadata = self.get_tool(tool_name)
        if not metadata:
            return False

        # Basic validation - check required parameters exist
        tool_params = metadata.parameters
        for param_name, param_info in tool_params.items():
            if param_info.get("required", True) and param_name not in parameters:  # type: ignore
                logger.error(
                    f"Required parameter '{param_name}' missing for tool '{tool_name}'"
                )
                return False

        return True


# Decorator for registering tools
def register_tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    category: str = "general",
    requires_auth: bool = False,
    timeout_seconds: int = 30,
):
    """
    Decorator to register a function as a tool

    Usage:
    @register_tool(
        name="my_tool",
        description="Does something useful",
        parameters={
            "param1": {"type": "string", "description": "Parameter 1"}
        }
    )
    def my_tool_function(param1: str):
        return f"Result: {param1}"
    """

    def decorator(func: Callable):
        registry = ToolRegistry()
        registry.register_function(
            name=name,
            function=func,
            description=description,
            parameters=parameters,
            category=category,
            requires_auth=requires_auth,
            timeout_seconds=timeout_seconds,
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Utility functions
def get_registry() -> ToolRegistry:
    """Get the global tool registry instance"""
    return ToolRegistry()


def list_available_tools() -> List[str]:
    """List all available tools"""
    return ToolRegistry().list_tools()


def get_tool_info(name: str) -> Optional[Dict[str, Any]]:
    """Get tool information"""
    return ToolRegistry().get_tool_info(name)


if __name__ == "__main__":
    # Test the registry
    registry = ToolRegistry()
    print("Available tools:", registry.list_tools())
    for tool_name in registry.list_tools():
        print(f"\n{tool_name}:", registry.get_tool_info(tool_name))
