# core/rag/__init__.py
"""
RAG (Retrieval Augmented Generation) Module

提供中英文混合內容的檢索增強生成功能
"""

from .engine import (
    ChineseRAGEngine,
    get_rag_engine,
    DocumentMemory,
    Document,
    DocumentMetadata,
    SearchResult,
    ChunkResult,
)

from .document_processor import DocumentProcessor, ProcessedDocument

from .vector_store import VectorStore, VectorMetadata, HybridVectorStore

from .retriever import (
    BaseRetriever,
    SemanticRetriever,
    BM25Retriever,
    HybridRetriever,
    AdvancedRetriever,
    RetrievalResult,
    RetrievalQuery,
)

from .embeddings import (
    BaseEmbeddingModel,
    SentenceTransformerEmbedding,
    TransformerEmbedding,
    EmbeddingModelFactory,
    EmbeddingManager,
    get_embedding_manager,
    encode_text,
    get_embedding_dimension,
)

__all__ = [
    # Engine components
    "ChineseRAGEngine",
    "get_rag_engine",
    "DocumentMemory",
    "Document",
    "DocumentMetadata",
    "SearchResult",
    "ChunkResult",
    # Document processing
    "DocumentProcessor",
    "ProcessedDocument",
    # Vector storage
    "VectorStore",
    "VectorMetadata",
    "HybridVectorStore",
    # Retrieval
    "BaseRetriever",
    "SemanticRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "AdvancedRetriever",
    "RetrievalResult",
    "RetrievalQuery",
    # Embeddings
    "BaseEmbeddingModel",
    "SentenceTransformerEmbedding",
    "TransformerEmbedding",
    "EmbeddingModelFactory",
    "EmbeddingManager",
    "get_embedding_manager",
    "encode_text",
    "get_embedding_dimension",
]

# Version info
__version__ = "1.0.0"
__author__ = "Multi-Modal Lab"
