# schemas/rag.py
"""
RAG API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional
from .schemas_base import BaseRequest, BaseResponse, BaseParameters


class RAGParameters(BaseParameters):
    """RAG operation parameters"""

    top_k: int = Field(5, ge=1, le=50, description="Number of top results to return")
    min_score: float = Field(
        0.1, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    max_context_length: int = Field(
        2000, ge=100, le=8000, description="Maximum context length"
    )
    chunk_size: int = Field(512, ge=100, le=2048, description="Text chunk size")
    chunk_overlap: int = Field(50, ge=0, le=200, description="Chunk overlap size")
    hybrid_weight: float = Field(
        0.7, ge=0.0, le=1.0, description="Semantic vs BM25 weight"
    )


class RAGAddDocumentRequest(BaseRequest):
    """Add document to RAG index request"""

    doc_id: str = Field(..., min_length=1, description="Unique document identifier")
    content: str = Field(..., min_length=1, description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    parameters: Optional[RAGParameters] = Field(default_factory=RAGParameters)

    @field_validator("content")
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v


class RAGAddDocumentResponse(BaseResponse):
    """Add document response"""

    doc_id: str = Field(..., description="Document identifier")
    added: bool = Field(..., description="Whether document was successfully added")
    chunks_created: int = Field(0, description="Number of chunks created")
    parameters: Optional[RAGParameters] = Field(None, description="Parameters used")


class RAGSearchRequest(BaseRequest):
    """Search documents request"""

    query: str = Field(..., min_length=1, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    parameters: RAGParameters = Field(default_factory=RAGParameters)

    @field_validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v


class RAGSearchResult(BaseModel):
    """Single search result"""

    doc_id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Relevance score")
    rank: int = Field(..., description="Result rank")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")


class RAGSearchResponse(BaseResponse):
    """Search documents response"""

    query: str = Field(..., description="Original query")
    results: List[RAGSearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results found")
    parameters: RAGParameters = Field(..., description="Search parameters used")


class RAGQueryRequest(BaseRequest):
    """RAG-enhanced query request"""

    query: str = Field(..., min_length=1, description="Question or query")
    context_filters: Optional[Dict[str, Any]] = Field(
        None, description="Context filtering"
    )
    parameters: RAGParameters = Field(default_factory=RAGParameters)

    @field_validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v


class RAGSource(BaseModel):
    """RAG source information"""

    doc_id: str = Field(..., description="Source document ID")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(..., description="Source metadata")


class RAGQueryResponse(BaseResponse):
    """RAG query response"""

    query: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    context: str = Field(..., description="Retrieved context")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    parameters: RAGParameters = Field(..., description="Parameters used")


class RAGStatsResponse(BaseResponse):
    """RAG statistics response"""

    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    index_size: int = Field(..., description="Vector index size")
    embedding_model: str = Field(..., description="Embedding model used")
    embedding_dim: int = Field(..., description="Embedding dimension")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class RAGRebuildRequest(BaseRequest):
    """Rebuild index request"""

    force: bool = Field(False, description="Force rebuild even if not needed")
    optimize: bool = Field(True, description="Optimize index after rebuild")


class RAGRebuildResponse(BaseResponse):
    """Rebuild index response"""

    success: bool = Field(..., description="Whether rebuild was successful")
    documents_processed: int = Field(..., description="Number of documents processed")
    time_taken_seconds: float = Field(..., description="Time taken for rebuild")
    message: str = Field(..., description="Status message")


class RAGUploadRequest(BaseModel):
    """File upload request"""

    world_id: str = Field("default", description="World/namespace identifier")
    tags: Optional[str] = Field(None, description="Comma-separated tags")
    chunk_size: int = Field(512, ge=100, le=2048, description="Text chunk size")
    chunk_overlap: int = Field(50, ge=0, le=200, description="Chunk overlap")


class RAGDocumentInfo(BaseModel):
    """Document information"""

    doc_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    chunks: int = Field(..., description="Number of chunks")
    total_chars: int = Field(..., description="Total character count")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")


class RAGListDocumentsResponse(BaseResponse):
    """List documents response"""

    documents: List[RAGDocumentInfo] = Field(..., description="Document list")
    total: int = Field(..., description="Total number of documents")
    limit: int = Field(..., description="Results limit")
    offset: int = Field(..., description="Results offset")


class RAGDeleteDocumentResponse(BaseResponse):
    """Delete document response"""

    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Status message")
    chunks_removed: int = Field(..., description="Number of chunks removed")


# Batch operations
class RAGBatchAddRequest(BaseRequest):
    """Batch add documents request"""

    documents: List[Dict[str, Any]] = Field(
        ..., min_items=1, description="Documents to add"  # type: ignore
    )
    parameters: RAGParameters = Field(default_factory=RAGParameters)  # type: ignore

    @field_validator("documents")
    def validate_documents(cls, v):
        for doc in v:
            if "doc_id" not in doc or "content" not in doc:
                raise ValueError("Each document must have doc_id and content")
            if not doc["content"].strip():
                raise ValueError("Document content cannot be empty")
        return v


class RAGBatchAddResponse(BaseResponse):
    """Batch add documents response"""

    total_requested: int = Field(..., description="Total documents requested")
    successful: int = Field(..., description="Successfully added documents")
    failed: int = Field(..., description="Failed documents")
    failed_docs: List[str] = Field(..., description="IDs of failed documents")
    parameters: RAGParameters = Field(..., description="Parameters used")
