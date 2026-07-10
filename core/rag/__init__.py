"""RAG package with dependency-safe lazy exports.

Importing a lightweight helper such as ``core.rag.context_retrieval`` must not
eagerly import sentence-transformers, FAISS, or model runtimes.  Runtime-heavy
objects are resolved only when their public export is actually requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    # Engine
    "ChineseRAGEngine": (".engine", "ChineseRAGEngine"),
    "get_rag_engine": (".engine", "get_rag_engine"),
    "DocumentMemory": (".engine", "DocumentMemory"),
    "Document": (".engine", "Document"),
    "DocumentMetadata": (".engine", "DocumentMetadata"),
    "SearchResult": (".engine", "SearchResult"),
    "ChunkResult": (".engine", "ChunkResult"),
    # Document processing
    "DocumentProcessor": (".document_processor", "DocumentProcessor"),
    "ProcessedDocument": (".document_processor", "ProcessedDocument"),
    # Vector storage
    "VectorStore": (".vector_store", "VectorStore"),
    "VectorMetadata": (".vector_store", "VectorMetadata"),
    "HybridVectorStore": (".vector_store", "HybridVectorStore"),
    # Retrieval
    "BaseRetriever": (".retriever", "BaseRetriever"),
    "SemanticRetriever": (".retriever", "SemanticRetriever"),
    "BM25Retriever": (".retriever", "BM25Retriever"),
    "HybridRetriever": (".retriever", "HybridRetriever"),
    "AdvancedRetriever": (".retriever", "AdvancedRetriever"),
    "RetrievalResult": (".retriever", "RetrievalResult"),
    "RetrievalQuery": (".retriever", "RetrievalQuery"),
    # Embeddings
    "BaseEmbeddingModel": (".embeddings", "BaseEmbeddingModel"),
    "SentenceTransformerEmbedding": (".embeddings", "SentenceTransformerEmbedding"),
    "TransformerEmbedding": (".embeddings", "TransformerEmbedding"),
    "EmbeddingModelFactory": (".embeddings", "EmbeddingModelFactory"),
    "EmbeddingManager": (".embeddings", "EmbeddingManager"),
    "get_embedding_manager": (".embeddings", "get_embedding_manager"),
    "encode_text": (".embeddings", "encode_text"),
    "get_embedding_dimension": (".embeddings", "get_embedding_dimension"),
}

__all__ = sorted(_EXPORTS)
__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
