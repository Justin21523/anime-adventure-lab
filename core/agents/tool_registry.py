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
import os
from functools import wraps

logger = logging.getLogger(__name__)


def _file_tools_enabled() -> bool:
    return os.getenv("AGENT_ENABLE_FILE_TOOLS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
            self.config_path = str(Path("configs") / "agent.yaml")
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
                    if (
                        tool_config.get("category") == "file_system"
                        and not _file_tools_enabled()
                    ):
                        continue
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
        finally:
            # Always ensure core defaults are present; add missing ones if config existed
            self._register_default_tools(add_missing=True)

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
                module_path = f"core.agents.{module_path}"

            module = importlib.import_module(module_path)
            function = getattr(module, function_name)

            return function

        except Exception as e:
            logger.error(f"Failed to load function {function_path}: {e}")
            return None

    def _register_default_tools(self, add_missing: bool = False):
        """Register default tools if no configuration found"""
        logger.info("Registering default tools")

        # Register built-in tools
        from .tools.calculator import calculate, basic_math, percentage, unit_convert
        from .tools.web_search import brave_search, brave_search_summary
        from .tools.file_ops import (
            list_files,
            read_file,
            write_file,
            file_exists,
            create_directory,
            execute as file_ops_execute,
        )
        from .tools.rag_search import rag_search

        def _register_if_needed(name, func, desc, params):
            if add_missing and name in self._tools:
                return
            self.register_function(
                name=name,
                function=func,
                description=desc,
                parameters=params,
            )

        for name, func, desc, params in [
            (
                "calculator",
                calculate,
                "Evaluate a mathematical expression safely",
                {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate",
                        "required": True,
                        "default": "2+2",
                    }
                },
            ),
            (
                "basic_math",
                basic_math,
                "Basic arithmetic with two numbers",
                {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                    "op": {
                        "type": "string",
                        "description": "Operation (+, -, *, /)",
                        "default": "+",
                    },
                },
            ),
            (
                "percentage",
                percentage,
                "Compute percentage of a value",
                {
                    "value": {"type": "number", "description": "Base value"},
                    "percent": {"type": "number", "description": "Percent (0-100)"},
                },
            ),
            (
                "unit_convert",
                unit_convert,
                "Convert units (supports length/weight/time presets)",
                {
                    "value": {"type": "number", "description": "Value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"},
                },
            ),
            (
                "web_search",
                brave_search,
                "Real web search via Brave API",
                {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of results",
                        "required": False,
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Brave API key (fallback to BRAVE_API_KEY)",
                        "required": False,
                    },
                },
            ),
            (
                "web_search_summary",
                brave_search_summary,
                "Web search with short summary (requires API key)",
                {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of results",
                        "required": False,
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Brave API key (fallback to BRAVE_API_KEY)",
                        "required": False,
                    },
                },
            ),
            (
                "rag_search",
                rag_search,
                "Search existing RAG index for contextual snippets",
                {
                    "query": {"type": "string", "description": "Query text"},
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Top K results",
                        "required": False,
                    },
                },
            ),
            (
                "file_list",
                list_files,
                "List files in a directory",
                {
                    "path": {"type": "string", "description": "Directory path"},
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., *.txt)",
                        "default": "*",
                    },
                },
            ),
            (
                "file_read",
                read_file,
                "Read a text file safely",
                {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "encoding": {
                        "type": "string",
                        "description": "Text encoding",
                        "default": "utf-8",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Max lines to read",
                        "default": 1000,
                    },
                },
            ),
            (
                "file_write",
                write_file,
                "Write content to a file safely",
                {
                    "file_path": {"type": "string", "description": "Target file"},
                    "content": {"type": "string", "description": "Content to write"},
                    "encoding": {
                        "type": "string",
                        "description": "Text encoding",
                        "default": "utf-8",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Allow overwriting existing files",
                        "default": False,
                        "required": False,
                    },
                },
            ),
            (
                "file_exists",
                file_exists,
                "Check if file exists",
                {"file_path": {"type": "string", "description": "Path to check"}},
            ),
            (
                "create_directory",
                create_directory,
                "Create a directory safely",
                {"dir_path": {"type": "string", "description": "Directory path"}},
            ),
            (
                "file_ops",
                file_ops_execute,
                "Unified file operations (list/read/write/info)",
                {
                    "operation": {
                        "type": "string",
                        "description": "Operation name (list/read/write/info/exists/delete/create_dir)",
                        "required": True,
                    },
                    "path": {
                        "type": "string",
                        "description": "Target path for the operation",
                        "required": True,
                    },
                    "content": {
                        "type": "string",
                        "description": "Content for write operation",
                        "required": False,
                        "default": "",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern for list",
                        "required": False,
                        "default": "*",
                    },
                },
            ),
        ]:
            try:
                if name in {
                    "file_list",
                    "file_read",
                    "file_write",
                    "file_exists",
                    "create_directory",
                    "file_ops",
                } and not _file_tools_enabled():
                    continue
                _register_if_needed(name, func, desc, params)
            except Exception as e:
                logger.error(f"Failed to register tool {name}: {e}")

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

    # 向後兼容：舊版介面名稱
    def get_tool_function(self, name: Union[str, ToolMetadata]) -> Optional[Callable]:
        tool_name = name.name if isinstance(name, ToolMetadata) else name
        return self.get_function(tool_name)

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

    def validate_parameters(
        self, tool: Union[str, ToolMetadata], parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Validate parameters for a tool and apply defaults"""
        metadata = tool if isinstance(tool, ToolMetadata) else self.get_tool(tool)
        if not metadata:
            raise ValueError(f"Tool '{tool}' not found")

        tool_params = metadata.parameters or {}
        validated = dict(parameters)

        for param_name, param_info in tool_params.items():
            required = param_info.get("required", False)  # type: ignore
            if param_name not in validated:
                if "default" in param_info:
                    validated[param_name] = param_info.get("default")
                elif required:
                    msg = f"Required parameter '{param_name}' missing for tool '{metadata.name}'"
                    logger.error(msg)
                    raise ValueError(msg)

        return validated


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
