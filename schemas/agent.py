# schemas/agent.py
"""
Agent API Schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from .base import BaseRequest, BaseResponse, BaseParameters


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


class AgentToolCallResponse(BaseResponse):
    """Agent tool call response"""

    tool_name: str = Field(..., description="Tool that was called")
    result: Dict[str, Any] = Field(..., description="Tool execution result")
    execution_time_ms: Optional[float] = Field(None, description="Execution time")
    parameters: Dict[str, Any] = Field(..., description="Parameters used")


class AgentTaskRequest(BaseRequest):
    """Agent task execution request"""

    task_description: str = Field(..., min_length=5, description="Task description")
    parameters: Optional[AgentParameters] = Field(default_factory=AgentParameters)  # type: ignore


class AgentTaskResponse(BaseResponse):
    """Agent task execution response"""

    task_description: str = Field(..., description="Original task")
    result: str = Field(..., description="Task execution result")
    tools_used: List[str] = Field(..., description="Tools that were used")
    steps_taken: int = Field(..., description="Number of steps taken")
    parameters: AgentParameters = Field(..., description="Parameters used")
