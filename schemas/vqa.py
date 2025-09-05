# schemas/vqa.py
"""
Visual Question Answering API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from .base import BaseRequest, BaseResponse, UsageInfo, BaseParameters


class VQAParameters(BaseParameters):
    """VQA-specific parameters"""

    max_length: int = Field(100, ge=20, le=300, description="Maximum answer length")
    language: str = Field("auto", description="Answer language (auto/en/zh)")

    @field_validator("language", mode="after")
    def validate_language(cls, v):
        if v not in ["auto", "en", "zh"]:
            raise ValueError("Language must be 'auto', 'en', or 'zh'")
        return v


class VQARequest(BaseRequest):
    """Visual Question Answering request"""

    question: str = Field(
        ..., min_length=3, max_length=500, description="Question about the image"
    )
    parameters: Optional[VQAParameters] = Field(default_factory=VQAParameters)  # type: ignore
    context: Optional[str] = Field(
        None, description="Additional context for the question"
    )

    @field_validator("question", mode="after")
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        return v.strip()


class VQAResponse(BaseResponse):
    """Visual Question Answering response"""

    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Answer confidence score"
    )
    model_used: str = Field(..., description="Model used for VQA")
    language: str = Field(..., description="Detected answer language")

    # Unified parameters field
    parameters: VQAParameters = Field(..., description="Parameters used for generation")

    # Optional analysis
    usage: Optional[UsageInfo] = Field(None, description="Resource usage")
    question_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Question complexity analysis"
    )


class ConversationMessage(BaseModel):
    """Single message in VQA conversation"""

    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")

    @field_validator("role", mode="after")
    def validate_role(cls, v):
        if v not in ["user", "assistant"]:
            raise ValueError("Role must be 'user' or 'assistant'")
        return v


class VQAConversationRequest(BaseRequest):
    """Multi-turn VQA conversation request"""

    messages: List[ConversationMessage] = Field(
        ..., min_items=1, description="Conversation history"  # type: ignore
    )
    max_length: int = Field(150, ge=20, le=300, description="Maximum response length")

    @field_validator("messages", mode="after")
    def validate_messages(cls, v):
        if not v:
            raise ValueError("At least one message required")

        # Check that last message is from user
        if v[-1].role != "user":
            raise ValueError("Last message must be from user")

        return v


class VQAConversationResponse(BaseResponse):
    """Multi-turn VQA conversation response"""

    question: str = Field(..., description="Latest user question")
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str = Field(..., description="VQA model used")
    conversation_turns: int = Field(..., description="Number of conversation turns")

    # Context analysis
    has_context: bool = Field(..., description="Whether previous context was used")
    context_relevance: Optional[float] = Field(
        None, description="Context relevance score"
    )
