# schemas/chat.py
"""
Text Chat API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from .base import BaseRequest, BaseResponse, UsageInfo


class ChatMessage(BaseModel):
    """Single chat message"""

    role: str = Field(..., description="Message role (system/user/assistant)")
    content: str = Field(..., min_length=1, description="Message content")

    @field_validator("role", mode="after")
    def validate_role(cls, v):
        if v not in ["system", "user", "assistant"]:
            raise ValueError("Role must be 'system', 'user', or 'assistant'")
        return v

    @field_validator("content", mode="after")
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()


class ChatRequest(BaseRequest):
    """Text chat completion request"""

    messages: List[ChatMessage] = Field(
        ..., min_items=1, description="Conversation messages"  # type: ignore
    )
    max_length: int = Field(512, ge=50, le=1000, description="Maximum response length")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    model: Optional[str] = Field(None, description="Optional model override")

    # Advanced parameters
    top_p: Optional[float] = Field(
        0.9, ge=0.1, le=1.0, description="Nucleus sampling parameter"
    )
    repetition_penalty: Optional[float] = Field(
        1.0, ge=0.5, le=2.0, description="Repetition penalty"
    )

    @field_validator("messages", mode="after")
    def validate_messages(cls, v):
        if not v:
            raise ValueError("At least one message required")

        # Check for valid role sequence
        roles = [msg.role for msg in v]
        if roles[-1] != "user":
            raise ValueError("Last message must be from user")

        return v


class ChatResponse(BaseResponse):
    """Text chat completion response"""

    message: str = Field(..., description="Generated response message")
    model_used: str = Field(..., description="Model used for generation")
    usage: UsageInfo = Field(..., description="Token usage information")

    # Analysis metadata
    response_quality: Optional[Dict[str, Any]] = Field(
        None, description="Response quality metrics"
    )
    safety_check: Optional[Dict[str, bool]] = Field(
        None, description="Safety filter results"
    )


class ChatStreamChunk(BaseModel):
    """Single chunk in streaming chat response"""

    content: str = Field(..., description="Incremental content")
    done: bool = Field(False, description="Whether generation is complete")
    model: Optional[str] = Field(None, description="Model name")
    usage: Optional[UsageInfo] = Field(
        None, description="Final usage info (only when done=True)"
    )
    error: Optional[str] = Field(None, description="Error message if failed")


class SystemPromptTestRequest(BaseModel):
    """System prompt testing request"""

    system_prompt: str = Field(..., min_length=10, description="System prompt to test")
    user_message: str = Field(..., min_length=1, description="Test user message")
    model: Optional[str] = Field(None, description="Model to use for testing")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")

    @field_validator("system_prompt", "user_message", mode="after")
    def validate_non_empty(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class SystemPromptTestResponse(BaseResponse):
    """System prompt testing response"""

    system_prompt: str = Field(..., description="Tested system prompt")
    user_message: str = Field(..., description="Test user message")
    assistant_response: str = Field(..., description="Generated assistant response")
    model_used: str = Field(..., description="Model used")
    usage: UsageInfo = Field(..., description="Resource usage")

    # Evaluation metrics
    response_length: int = Field(..., description="Response length in characters")
    follows_system_prompt: Optional[bool] = Field(
        None, description="Whether response follows system prompt"
    )
    prompt_adherence_score: Optional[float] = Field(
        None, description="Prompt adherence score (0-1)"
    )


class ChatModelInfo(BaseModel):
    """Chat model information"""

    name: str = Field(..., description="Model identifier")
    description: str = Field(..., description="Model description")
    languages: List[str] = Field(..., description="Supported languages")
    parameters: str = Field(..., description="Parameter count")
    context_length: int = Field(..., description="Maximum context length")
    recommended: bool = Field(False, description="Whether this model is recommended")
    loaded: bool = Field(False, description="Whether currently loaded in memory")


class ChatModelsResponse(BaseResponse):
    """List of available chat models"""

    available_models: List[ChatModelInfo] = Field(..., description="Available models")
    loaded_models: List[str] = Field(..., description="Currently loaded model names")
    default_model: str = Field(..., description="Default model name")
