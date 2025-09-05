# schemas/rag.py
"""
RAG API Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .base import BaseRequest, BaseResponse, BaseParameters, UsageInfo


class RAGParameters(BaseParameters):
    """RAG-specific parameters"""

    top_k: int = Field(5, ge=1, le=20, description="Number of top results to retrieve")
    min_score: float = Field(0.3, ge=0.0, le=1.0, description="Minimum relevance score")
    max_context_length: int = Field(
        1000, ge=100, le=3000, description="Maximum context length"
    )


class RAGAddDocumentRequest(BaseRequest):
    """Add document to RAG index request"""

    doc_id: str = Field(..., description="Document identifier")
    content: str = Field(..., min_length=10, description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    parameters: Optional[RAGParameters] = Field(default_factory=RAGParameters)  # type: ignore


class RAGAddDocumentResponse(BaseResponse):
    """Add document response"""

    doc_id: str = Field(..., description="Document identifier")
    added: bool = Field(..., description="Whether document was added successfully")
    parameters: RAGParameters = Field(..., description="Parameters used")


class RAGSearchRequest(BaseRequest):
    """Search documents request"""

    query: str = Field(..., min_length=3, description="Search query")
    parameters: Optional[RAGParameters] = Field(default_factory=RAGParameters)  # type: ignore


class RAGSearchResponse(BaseResponse):
    """Search documents response"""

    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total results found")
    parameters: RAGParameters = Field(..., description="Parameters used")


class RAGQueryRequest(BaseRequest):
    """RAG-enhanced query request"""

    query: str = Field(..., min_length=3, description="Question to answer")
    parameters: Optional[RAGParameters] = Field(default_factory=RAGParameters)  # type: ignore


class RAGQueryResponse(BaseResponse):
    """RAG query response"""

    query: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    context: str = Field(..., description="Retrieved context")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    parameters: RAGParameters = Field(..., description="Parameters used")
    usage: Optional[UsageInfo] = Field(None, description="Resource usage")
