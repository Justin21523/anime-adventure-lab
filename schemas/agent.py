# schemas/agent.py
"""
Agent API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from .schemas_base import BaseRequest, BaseResponse, BaseParameters


class AgentParameters(BaseParameters):
    """Agent-specific parameters"""

    max_iterations: int = Field(5, ge=1, le=10, description="Maximum iterations")
    max_tools_per_iteration: int = Field(
        2, ge=1, le=5, description="Max tools per iteration"
    )
    enable_chain_of_thought: bool = Field(True, description="Enable reasoning chain")


class AgentToolCallRequest(BaseRequest):
    """Agent tool call request"""

    tool_name: str = Field(..., description="Tool name to call")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")

    @field_validator("tool_name")
    def validate_tool_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip().lower()


class AgentToolCallResponse(BaseResponse):
    """Agent tool call response"""

    tool_name: str = Field(..., description="Tool that was called")
    result: Dict[str, Any] = Field(..., description="Tool execution result")
    execution_time_ms: Optional[float] = Field(None, description="Execution time")
    parameters: Dict[str, Any] = Field(..., description="Parameters used")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class AgentTaskRequest(BaseRequest):
    """Agent task execution request"""

    task_description: str = Field(
        ..., min_length=5, max_length=1000, description="Task description"
    )
    parameters: Optional[AgentParameters] = Field(
        default_factory=AgentParameters, description="Agent execution parameters"
    )

    @field_validator("task_description")
    def validate_task_description(cls, v):
        if not v or not v.strip():
            raise ValueError("Task description cannot be empty")
        return v.strip()


class AgentTaskResponse(BaseResponse):
    """Agent task execution response"""

    task_description: str = Field(..., description="Original task")
    result: str = Field(..., description="Task execution result")
    success: bool = Field(..., description="Whether task succeeded")
    tools_used: List[str] = Field(..., description="Tools that were used")
    steps_taken: int = Field(..., description="Number of steps taken")
    execution_time_ms: float = Field(..., description="Total execution time")
    reasoning_chain: Optional[List[str]] = Field(None, description="Chain of reasoning")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    parameters: Dict[str, Any] = Field(..., description="Parameters used")


class ToolInfo(BaseModel):
    """Tool information structure"""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Dict[str, Any]] = Field(
        ..., description="Tool parameters schema"
    )
    timeout_seconds: int = Field(..., description="Tool timeout")


class AgentToolListResponse(BaseResponse):
    """Agent tool list response"""

    tools: List[ToolInfo] = Field(..., description="Available tools")
    total_count: int = Field(..., description="Total number of tools")


class AgentStatusResponse(BaseResponse):
    """Agent status response"""

    status: str = Field(..., description="Agent status")
    tools_loaded: int = Field(..., description="Number of tools loaded")
    available_tools: List[str] = Field(..., description="List of available tool names")
    max_iterations: int = Field(..., description="Max iterations setting")
    max_tools_per_iteration: int = Field(
        ..., description="Max tools per iteration setting"
    )
    config_file: str = Field(..., description="Configuration file path")
