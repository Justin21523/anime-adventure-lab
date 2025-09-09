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

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class Document:
    """Document structure for RAG"""

    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ChunkResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    section_title: str = ""
    metadata: Dict[str, Any] = None  # type: ignore

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """Search result with relevance score"""

    document: Document
    score: float
    rank: int


class ChineseTextProcessor:
    """Chinese text processing utilities"""

    def __init__(self):
        try:
            self.cc = opencc.OpenCC("t2s")  # Traditional to Simplified
        except:
            logger.warning("OpenCC not available, using basic text processing")
            self.cc = None

    def normalize_text(self, text: str) -> str:
        """Normalize Chinese/English mixed text"""
        if not text:
            return ""

        # Basic cleaning
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        text = re.sub(
            r"[^\w\s\u4e00-\u9fff.,!?;:]", "", text
        )  # Keep Chinese, English, basic punctuation

        return text.strip()

    def chunk_text(
        self, text: str, chunk_size: int = 512, overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []

        text = self.normalize_text(text)

        # For Chinese text, we split by sentences and paragraphs
        sentences = re.split(r"[。！？\n]", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Add overlap between chunks
        final_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0 and overlap > 0:
                # Add overlap from previous chunk
                prev_words = chunks[i - 1].split()[-overlap:]
                overlap_text = " ".join(prev_words)
                chunk = overlap_text + " " + chunk
            final_chunks.append(chunk)

        return final_chunks


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
        self.index: Optional[faiss.IndexFlatIP] = None
        self.doc_id_map: List[str] = []  # Maps FAISS index to doc_id
        # Index parameters
        self.embedding_dim = 768  # Default for BGE models
        self.max_chunk_size = 500  # Characters per chunk

        # BM25 for hybrid search
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[str] = []

        # Text processor
        self.text_processor = ChineseTextProcessor()

    @handle_model_error
    def _load_model(self) -> bool:
        """Load embedding model"""
        if self._loaded:
            return True

        try:
            cache_dir = self.cache.get_path("MODELS_EMBEDDING")

            logger.info(f"Loading embedding model: {self.embedding_model_name}")

            # Try SentenceTransformer first (better for retrieval)
            try:
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    cache_folder=str(cache_dir),
                    device=self.config.device,
                )
                self.embedding_dim = (
                    self._embedding_model.get_sentence_embedding_dimension()
                )
                logger.info(
                    f"Loaded SentenceTransformer model, dim={self.embedding_dim}"
                )

            except Exception as e:
                logger.warning(f"SentenceTransformer failed, trying AutoModel: {e}")

                # Fallback to Transformers
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.embedding_model_name, cache_dir=str(cache_dir)
                )
                self._embedding_model = AutoModel.from_pretrained(
                    self.embedding_model_name,
                    cache_dir=str(cache_dir),
                    torch_dtype=(
                        torch.float16 if self.config.use_fp16 else torch.float32
                    ),
                    device_map="auto" if torch.cuda.is_available() else None,
                )

                # Test embedding dimension
                test_embedding = self._encode_text("test")
                self.embedding_dim = test_embedding.shape[0]
                logger.info(f"Loaded AutoModel, dim={self.embedding_dim}")

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Model loading failed: {e}")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self._loaded:
            self._load_model()

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

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        if not self._loaded:
            self._load_model()

        try:
            if isinstance(self._embedding_model, SentenceTransformer):
                # SentenceTransformer path
                embedding = self._embedding_model.encode(
                    text, convert_to_numpy=True, normalize_embeddings=True
                )
                return embedding.astype(np.float32)

            else:
                # Transformers path
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._embedding_model(**inputs)
                    # Use CLS token or mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

                    # Normalize
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

                return embedding.cpu().numpy().astype(np.float32)

        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            raise EmbeddingError(f"Encoding failed: {e}")

    def _create_index(self) -> faiss.IndexFlatIP:
        """Create new FAISS index"""
        if not self._loaded:
            self._load_model()

        # Inner Product index for cosine similarity (with normalized vectors)
        index = faiss.IndexFlatIP(self.embedding_dim)
        logger.info(f"Created FAISS index with dimension {self.embedding_dim}")
        return index

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

    def _update_bm25(self):
        """Update BM25 index with current documents"""
        try:
            self.bm25_corpus = [
                self.text_processor.normalize_text(doc.content)
                for doc in self.documents.values()
            ]

            if self.bm25_corpus:
                tokenized_corpus = [doc.split() for doc in self.bm25_corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
                logger.debug(f"Updated BM25 with {len(self.bm25_corpus)} documents")

        except Exception as e:
            logger.warning(f"BM25 update failed: {e}")
            self.bm25 = None

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

    def _semantic_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Semantic search using FAISS"""
        if not self.index or not self.documents:
            return []

        try:
            # Encode query
            query_embedding = self._encode_text(query)

            # Search FAISS index
            scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx in self.doc_id_map:
                    doc_id = self.doc_id_map[idx]
                    if doc_id in self.documents:
                        results.append(
                            SearchResult(
                                document=self.documents[doc_id],
                                score=float(score),
                                rank=0,
                            )
                        )

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        """BM25 lexical search"""
        if not self.bm25 or not self.bm25_corpus:
            return []

        try:
            query_tokens = self.text_processor.normalize_text(query).split()
            scores = self.bm25.get_scores(query_tokens)

            # Get top results
            top_indices = np.argsort(scores)[-top_k:][::-1]

            results = []
            doc_ids = list(self.documents.keys())

            for idx in top_indices:
                if idx < len(doc_ids) and scores[idx] > 0:
                    doc_id = doc_ids[idx]
                    results.append(
                        SearchResult(
                            document=self.documents[doc_id],
                            score=float(scores[idx]),
                            rank=0,
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def generate_context(
        self, query: str, max_context_length: int = 2000, top_k: int = 5
    ) -> Dict[str, Any]:
        """Generate context for RAG-enhanced generation"""
        try:
            search_results = self.search(query, top_k=top_k)

            if not search_results:
                return {"context": "", "sources": [], "total_chars": 0}

            # Build context from top results
            context_parts = []
            sources = []
            total_chars = 0

            for result in search_results:
                content = result.document.content

                # Check if adding this would exceed limit
                if total_chars + len(content) > max_context_length:
                    # Truncate to fit
                    remaining = max_context_length - total_chars
                    if remaining > 100:  # Only add if meaningful content fits
                        content = content[:remaining] + "..."
                    else:
                        break

                context_parts.append(f"[相關資料 {len(context_parts)+1}]\n{content}\n")
                sources.append(
                    {
                        "doc_id": result.document.doc_id,
                        "score": result.score,
                        "metadata": result.document.metadata,
                    }
                )

                total_chars += len(content)

            context = "\n".join(context_parts)

            return {"context": context, "sources": sources, "total_chars": total_chars}

        except Exception as e:
            logger.error(f"Context generation failed: {e}")
            return {"context": "", "sources": [], "total_chars": 0}

    def rebuild_index(self) -> bool:
        """Rebuild FAISS index from existing documents"""
        try:
            if not self.documents:
                logger.warning("No documents to rebuild index")
                return False

            # Create new index
            self.index = self._create_index()
            self.doc_id_map.clear()

            # Re-add all documents
            embeddings = []
            doc_ids = []

            for doc_id, document in self.documents.items():
                if document.embedding is not None:
                    embeddings.append(document.embedding)
                    doc_ids.append(doc_id)
                else:
                    # Re-generate embedding if missing
                    embedding = self._encode_text(document.content)
                    document.embedding = embedding
                    embeddings.append(embedding)
                    doc_ids.append(doc_id)

            if embeddings:
                # Batch add to FAISS
                embeddings_array = np.vstack(embeddings)
                self.index.add(embeddings_array)

                # Update mapping
                for i, doc_id in enumerate(doc_ids):
                    self.doc_id_map[i] = doc_id

            # Update BM25
            self._update_bm25()

            logger.info(f"Rebuilt index with {len(embeddings)} documents")
            return True

        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        unique_docs = set(
            doc.metadata.get("parent_doc_id", doc.doc_id)
            for doc in self.documents.values()
        )

        return {
            "total_documents": len(unique_docs),
            "total_chunks": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "model_loaded": self._loaded,
            "bm25_enabled": self.bm25 is not None,
        }

    def save_index(self, filepath: Optional[Path] = None) -> Path:
        """Save FAISS index and metadata to disk"""
        if filepath is None:
            filepath = self.cache.get_output_path("rag") / "index"

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        try:
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

            # Save documents (without embeddings to save space)
            documents_data = {}
            for doc_id, doc in self.documents.items():
                documents_data[doc_id] = {
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "created_at": (
                        doc.created_at.isoformat() if doc.created_at else None
                    ),
                }

            with open(filepath / "documents.json", "w", encoding="utf-8") as f:
                json.dump(documents_data, f, indent=2, ensure_ascii=False)

            logger.info(f"RAG index saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save RAG index: {e}")
            raise RAGError(f"Index save failed: {e}")

    def load_index(self, filepath: Path) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            filepath = Path(filepath)

            if not filepath.exists():
                logger.warning(f"Index path does not exist: {filepath}")
                return False

            # Load metadata
            metadata_file = filepath / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                self.doc_id_map = {int(k): v for k, v in metadata["doc_id_map"].items()}
                self.embedding_dim = metadata["embedding_dim"]

                # Verify model compatibility
                if metadata["embedding_model"] != self.embedding_model_name:
                    logger.warning(
                        f"Model mismatch: loaded={metadata['embedding_model']}, "
                        f"current={self.embedding_model_name}. Index rebuild recommended."
                    )

            # Load FAISS index
            faiss_file = filepath / "faiss.index"
            if faiss_file.exists():
                self.index = faiss.read_index(str(faiss_file))

            # Load documents
            documents_file = filepath / "documents.json"
            if documents_file.exists():
                with open(documents_file, "r", encoding="utf-8") as f:
                    documents_data = json.load(f)

                for doc_id, doc_data in documents_data.items():
                    created_at = None
                    if doc_data.get("created_at"):
                        created_at = datetime.fromisoformat(doc_data["created_at"])

                    self.documents[doc_id] = Document(
                        doc_id=doc_data["doc_id"],
                        content=doc_data["content"],
                        metadata=doc_data["metadata"],
                        created_at=created_at,
                    )

            # Update BM25
            self._update_bm25()

            logger.info(
                f"RAG index loaded from {filepath}, {len(self.documents)} documents"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load RAG index from {filepath}: {e}")
            return False

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
