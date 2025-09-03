# core/rag/engine.py

import os, pathlib, torch, json, hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import opencc
import re
from pathlib import Path
import pickle
from urllib.parse import quote
import uuid

import sys
from pathlib import Path

from ..exceptions import (
    RAGError,
    EmbeddingError,
    DocumentIndexError,
    handle_model_error,
)
from ..config import get_config
from ..shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    doc_id: str
    title: str
    source: str
    world_id: str
    upload_time: str
    language: str = "zh-TW"
    license: str = ""
    tags: List[str] = None  # type: ignore


@dataclass
class Document:
    """Document structure for RAG"""

    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: Optional[datetime] = None


@dataclass
class ChunkResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    section_title: str = ""
    metadata: Dict[str, Any] = None  # type: ignore


@dataclass
class SearchResult:
    """Search result with relevance score"""

    document: Document
    score: float
    rank: int


class ChineseRAGEngine:
    """RAG engine optimized for Chinese/English mixed content"""

    def __init__(self, embedding_model: Optional[str] = None):
        self.config = get_config()
        self.cache = get_shared_cache()

        # Model configuration
        self.embedding_model_name = embedding_model or self.config.model.embedding_model  # type: ignore
        self._embedding_model = None
        self._tokenizer = None
        self._loaded = False

        # Document storage
        self.documents: Dict[str, Document] = {}
        self.index: Optional[faiss.Index] = None
        self.doc_id_map: List[str] = []  # Maps FAISS index to doc_id

        # Index parameters
        self.embedding_dim = 768  # Default for BGE models
        self.max_chunk_size = 500  # Characters per chunk

    @handle_model_error
    def load_embedding_model(self) -> None:
        """Load embedding model for document encoding"""
        if self._loaded:
            return

        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")

            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name, cache_dir=self.cache.cache_root / "hf"  # type: ignore
            )

            self._embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name,
                device_map="auto",
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                cache_dir=self.cache.cache_root / "hf",  # type: ignore
                trust_remote_code=True,
            )

            # Get actual embedding dimension
            sample_input = self._tokenizer(
                "test", return_tensors="pt", max_length=512, truncation=True
            )
            with torch.no_grad():
                sample_output = self._embedding_model(**sample_input)
                self.embedding_dim = sample_output.last_hidden_state.mean(dim=1).shape[
                    -1
                ]

            self._loaded = True
            logger.info(f"Embedding model loaded, dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError("", f"Model loading failed: {str(e)}")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self._loaded:
            self.load_embedding_model()

        try:
            # Tokenize input
            inputs = self._tokenizer(
                text, return_tensors="pt", max_length=512, truncation=True, padding=True  # type: ignore
            )

            # Move to model device
            device = next(self._embedding_model.parameters()).device  # type: ignore
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = self._embedding_model(**inputs)  # type: ignore
                # Use mean pooling for sentence embedding
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            return embedding[0]  # Remove batch dimension

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(text[:50], str(e))

    def _chunk_document(self, content: str, doc_id: str) -> List[Document]:
        """Split document into chunks for indexing"""
        chunks = []

        # Simple chunking by character count with overlap
        overlap = 50
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + self.max_chunk_size
            chunk_text = content[start:end]

            # Try to break at sentence boundary
            if end < len(content):
                last_period = chunk_text.rfind("。")
                last_question = chunk_text.rfind("？")
                last_exclamation = chunk_text.rfind("！")
                last_newline = chunk_text.rfind("\n")

                break_point = max(
                    last_period, last_question, last_exclamation, last_newline
                )
                if break_point > start + 100:  # Minimum chunk size
                    chunk_text = chunk_text[: break_point + 1]
                    end = start + len(chunk_text)

            chunk_id = f"{doc_id}_chunk_{chunk_idx}"
            chunks.append(
                Document(
                    doc_id=chunk_id,
                    content=chunk_text.strip(),
                    metadata={
                        "parent_doc_id": doc_id,
                        "chunk_index": chunk_idx,
                        "start_char": start,
                        "end_char": end,
                    },
                    created_at=datetime.now(),
                )
            )

            start = end - overlap
            chunk_idx += 1

        return chunks

    def add_document(
        self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add document to RAG index"""
        try:
            # Validate input
            if not content.strip():
                raise DocumentIndexError(doc_id, "Empty content")

            if doc_id in self.documents:
                logger.warning(f"Document {doc_id} already exists, updating...")

            # Split into chunks
            chunks = self._chunk_document(content, doc_id)

            # Generate embeddings for chunks
            chunk_embeddings = []
            for chunk in chunks:
                embedding = self._generate_embedding(chunk.content)
                chunk.embedding = embedding
                chunk_embeddings.append(embedding)

                # Store chunk
                self.documents[chunk.doc_id] = chunk

            # Add to FAISS index
            if self.index is None:
                self.index = faiss.IndexFlatIP(
                    self.embedding_dim
                )  # Inner product for cosine similarity

            # Normalize embeddings for cosine similarity
            embeddings_matrix = np.array(chunk_embeddings).astype(np.float32)
            faiss.normalize_L2(embeddings_matrix)

            # Add to index
            self.index.add(embeddings_matrix)  # type: ignore

            # Update doc_id mapping
            for chunk in chunks:
                self.doc_id_map.append(chunk.doc_id)

            logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            raise DocumentIndexError(doc_id, str(e))

    def search(
        self, query: str, top_k: int = 5, min_score: float = 0.3
    ) -> List[SearchResult]:
        """Search for relevant documents"""
        if self.index is None or len(self.documents) == 0:
            logger.warning("No documents indexed for search")
            return []

        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            query_vector = np.array([query_embedding]).astype(np.float32)
            faiss.normalize_L2(query_vector)

            # Search in index
            scores, indices = self.index.search(
                query_vector, min(top_k, len(self.doc_id_map))
            )

            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if score < min_score:
                    continue

                doc_id = self.doc_id_map[idx]
                document = self.documents[doc_id]

                results.append(
                    SearchResult(document=document, score=float(score), rank=i + 1)
                )

            logger.info(f"Search for '{query[:50]}...' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed for query '{query[:50]}...': {e}")
            raise RAGError(f"Search failed: {str(e)}")

    def generate_context(
        self, query: str, max_context_length: int = 1000, top_k: int = 3
    ) -> Dict[str, Any]:
        """Generate context for RAG-enhanced generation"""
        search_results = self.search(query, top_k=top_k)

        if not search_results:
            return {
                "context": "",
                "sources": [],
                "relevance_scores": [],
                "total_documents": 0,
            }

        # Build context from top results
        context_parts = []
        sources = []
        scores = []
        total_length = 0

        for result in search_results:
            chunk_content = result.document.content

            # Check if adding this chunk exceeds limit
            if total_length + len(chunk_content) > max_context_length:
                # Truncate the chunk to fit
                remaining_space = max_context_length - total_length
                if remaining_space > 100:  # Only add if meaningful space left
                    chunk_content = chunk_content[:remaining_space] + "..."
                else:
                    break

            context_parts.append(chunk_content)
            sources.append(
                {
                    "doc_id": result.document.metadata.get(
                        "parent_doc_id", result.document.doc_id
                    ),
                    "chunk_id": result.document.doc_id,
                    "score": result.score,
                }
            )
            scores.append(result.score)
            total_length += len(chunk_content)

        context = "\n\n".join(context_parts)

        return {
            "context": context,
            "sources": sources,
            "relevance_scores": scores,
            "total_documents": len(search_results),
            "context_length": len(context),
        }

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.documents.get(doc_id)

    def remove_document(self, doc_id: str) -> bool:
        """Remove document and its chunks from index"""
        # Find all chunks for this document
        chunks_to_remove = [
            chunk_id
            for chunk_id, doc in self.documents.items()
            if doc.metadata.get("parent_doc_id") == doc_id or doc.doc_id == doc_id
        ]

        if not chunks_to_remove:
            return False

        # Remove from documents dict
        for chunk_id in chunks_to_remove:
            del self.documents[chunk_id]

        # Note: FAISS doesn't support efficient removal,
        # so we'd need to rebuild index in production
        logger.warning(
            f"Removed {len(chunks_to_remove)} chunks for doc {doc_id}. "
            "Index rebuild recommended for production."
        )

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        return {
            "total_documents": len(
                set(
                    doc.metadata.get("parent_doc_id", doc.doc_id)
                    for doc in self.documents.values()
                )
            ),
            "total_chunks": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "model_loaded": self._loaded,
        }

    def save_index(self, filepath: Optional[Path] = None) -> Path:
        """Save FAISS index and metadata to disk"""
        if filepath is None:
            filepath = self.cache.get_output_path("rag") / "index"  # type: ignore

        filepath = Path(filepath)  # type: ignore
        filepath.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(filepath / "faiss.index"))

        # Save metadata
        metadata = {
            "doc_id_map": self.doc_id_map,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "created_at": datetime.now().isoformat(),
            "total_documents": len(self.documents),
        }

        with open(filepath / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save documents
        documents_data = {}
        for doc_id, doc in self.documents.items():
            documents_data[doc_id] = {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "metadata": doc.metadata,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "embedding": (
                    doc.embedding.tolist() if doc.embedding is not None else None
                ),
            }

        with open(filepath / "documents.json", "w", encoding="utf-8") as f:
            json.dump(documents_data, f, indent=2, ensure_ascii=False)

        logger.info(f"RAG index saved to {filepath}")
        return filepath

    def load_index(self, filepath: Path) -> bool:
        """Load FAISS index and metadata from disk"""
        filepath = Path(filepath)

        try:
            # Load metadata
            with open(filepath / "metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.doc_id_map = metadata["doc_id_map"]
            self.embedding_dim = metadata["embedding_dim"]

            # Load FAISS index
            index_file = filepath / "faiss.index"
            if index_file.exists():
                self.index = faiss.read_index(str(index_file))

            # Load documents
            with open(filepath / "documents.json", "r", encoding="utf-8") as f:
                documents_data = json.load(f)

            for doc_id, doc_data in documents_data.items():
                embedding = None
                if doc_data["embedding"]:
                    embedding = np.array(doc_data["embedding"])

                created_at = None
                if doc_data["created_at"]:
                    created_at = datetime.fromisoformat(doc_data["created_at"])

                self.documents[doc_id] = Document(
                    doc_id=doc_data["doc_id"],
                    content=doc_data["content"],
                    metadata=doc_data["metadata"],
                    embedding=embedding,
                    created_at=created_at,
                )

            logger.info(
                f"RAG index loaded from {filepath}, {len(self.documents)} documents"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load RAG index from {filepath}: {e}")
            return False


# Global RAG engine instance
_rag_engine: Optional[ChineseRAGEngine] = None


def get_rag_engine(embedding_model: Optional[str] = None) -> ChineseRAGEngine:
    """Get global RAG engine instance"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = ChineseRAGEngine(embedding_model)

        # Try to load existing index
        config = get_config()
        cache = get_shared_cache()
        index_path = cache.get_output_path("rag") / "index"  # type: ignore

        if index_path.exists():
            try:
                _rag_engine.load_index(index_path)
                logger.info("Existing RAG index loaded")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")

    return _rag_engine


class DocumentMemory:
    """Simple document memory for RAG (alias for compatibility)"""

    def __init__(self):
        self.rag_engine = get_rag_engine()

    def add(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add document to memory"""
        self.rag_engine.add_document(doc_id, content, metadata or {})

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents"""
        results = self.rag_engine.search(query, top_k=top_k)
        return [
            {
                "content": result.document.content,
                "score": result.score,
                "metadata": result.document.metadata,
            }
            for result in results
        ]
