# core/rag/engine.py

import os, pathlib, torch, json, hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
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

from .schemas import DocumentMetadata, RetrievalItem
from .retrievers import SimpleRetriever
from .parsers import SimpleParser
from .memory import RAGMemory
from ..shared_cache import get_shared_cache


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
class ChunkResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    section_title: str = ""
    metadata: Dict[str, Any] = None  # type: ignore


class RAGEngine:
    """Main RAG processing engine"""

    def __init__(self):
        self.cache = get_shared_cache()
        self.retriever = SimpleRetriever()
        self.parser = SimpleParser()
        self.memory = RAGMemory()

    def add_document(self, text: str, metadata: DocumentMetadata | dict) -> dict:
        """Add document to RAG index"""
        try:
            # Convert dict to DocumentMetadata if needed
            if isinstance(metadata, dict):
                metadata = DocumentMetadata(**metadata)

            # Parse document into chunks
            chunks = self.parser.parse_text(text)

            # Generate chunk IDs and prepare for indexing
            doc_id = metadata.doc_id
            chunk_ids = [f"{doc_id}_{chunk['chunk_id']}" for chunk in chunks]
            texts = [chunk["text"] for chunk in chunks]
            doc_ids = [doc_id] * len(chunks)

            # Add to retrieval index
            self.retriever.add_documents(texts, doc_ids, chunk_ids)

            return {"chunks_added": len(chunks)}

        except Exception as e:
            raise RuntimeError(f"Failed to add document: {str(e)}")

    def retrieve(
        self,
        *,
        query: str,
        world_id: Optional[str] = None,
        top_k: int = 8,
        alpha: float = 0.7,
    ) -> List[RetrievalItem]:
        """Retrieve relevant documents"""
        try:
            return self.retriever.retrieve(query, world_id, top_k)
        except Exception as e:
            raise RuntimeError(f"Retrieval failed: {str(e)}")

    def write_memory(
        self,
        *,
        world_id: str,
        scope: str,
        content: str,
        doc_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Write to memory store"""
        try:
            self.memory.write_memory(
                world_id=world_id,
                scope=scope,
                content=content,
                doc_id=doc_id,
                metadata=metadata,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to write memory: {str(e)}")


class ChineseRAGEngine:
    """Chinese-optimized RAG engine with hybrid retrieval"""

    def __init__(self, index_dir: str, embed_model=None, cc_converter=None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.embed_model = embed_model
        self.cc_converter = cc_converter
        self.chunker = ChineseHierarchicalChunker()

        # Storage
        self.documents = {}  # doc_id -> DocumentMetadata
        self.chunks = {}  # chunk_id -> chunk_data
        self.chunk_embeddings = {}  # chunk_id -> embedding

        # Index components
        self.faiss_index = None
        self.chunk_id_list = []  # Mapping FAISS index to chunk_id
        self.bm25_corpus = []
        self.bm25_index = None

        # Load existing index if available
        self._load_index()

    def add_document(self, content: str, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Add document to RAG index"""
        print(f"[rag] Adding document: {metadata.doc_id}")

        # Store document metadata
        self.documents[metadata.doc_id] = metadata

        # Chunk the document
        chunks = self.chunker.chunk_text(content, metadata.doc_id)
        print(f"[rag] Generated {len(chunks)} chunks")

        # Process each chunk
        new_embeddings = []
        new_chunk_ids = []
        new_bm25_docs = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["text"]

            # Store chunk
            self.chunks[chunk_id] = {**chunk, "metadata": asdict(metadata)}

            # Generate embedding
            try:
                embedding = self.embed_model.encode(
                    chunk_text, normalize_embeddings=True, convert_to_numpy=True
                )
                self.chunk_embeddings[chunk_id] = embedding
                new_embeddings.append(embedding)
                new_chunk_ids.append(chunk_id)

                # Prepare for BM25 (tokenize Chinese text)
                tokens = self._tokenize_chinese(chunk_text)
                new_bm25_docs.append(tokens)

            except Exception as e:
                print(f"[rag] Error processing chunk {chunk_id}: {e}")
                continue

        # Update FAISS index
        if new_embeddings:
            embeddings_array = np.array(new_embeddings).astype("float32")

            if self.faiss_index is None:
                # Initialize FAISS index
                dim = embeddings_array.shape[1]
                self.faiss_index = faiss.IndexFlatIP(
                    dim
                )  # Inner Product (cosine similarity)

            self.faiss_index.add(embeddings_array)
            self.chunk_id_list.extend(new_chunk_ids)

            # Update BM25
            self.bm25_corpus.extend(new_bm25_docs)
            if self.bm25_corpus:
                self.bm25_index = BM25Okapi(self.bm25_corpus)

        # Save index
        self._save_index()

        return {
            "doc_id": metadata.doc_id,
            "chunks_added": len(chunks),
            "total_chunks": len(self.chunks),
            "status": "success",
        }

    def retrieve(
        self, query: str, world_id: str = None, top_k: int = 8, alpha: float = 0.7  # type: ignore
    ) -> List[ChunkResult]:
        """Hybrid retrieval with semantic + BM25"""
        if not self.faiss_index or not self.bm25_index:
            return []

        print(f"[rag] Retrieving for query: {query[:50]}...")

        # Query preprocessing
        normalized_query = self._normalize_query(query)
        query_tokens = self._tokenize_chinese(normalized_query)

        # Semantic search
        query_embedding = (
            self.embed_model.encode(
                normalized_query, normalize_embeddings=True, convert_to_numpy=True
            )
            .reshape(1, -1)
            .astype("float32")
        )

        semantic_scores, semantic_indices = self.faiss_index.search(
            query_embedding, min(50, len(self.chunk_id_list))
        )

        # BM25 search
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # Hybrid scoring
        hybrid_results = {}

        # Add semantic results
        for i, (score, idx) in enumerate(zip(semantic_scores[0], semantic_indices[0])):
            if idx < len(self.chunk_id_list):
                chunk_id = self.chunk_id_list[idx]
                hybrid_results[chunk_id] = {
                    "semantic_score": float(score),
                    "bm25_score": 0.0,
                    "chunk_id": chunk_id,
                }

        # Add BM25 results
        for chunk_idx, bm25_score in enumerate(bm25_scores):
            if chunk_idx < len(self.chunk_id_list):
                chunk_id = self.chunk_id_list[chunk_idx]
                if chunk_id in hybrid_results:
                    hybrid_results[chunk_id]["bm25_score"] = float(bm25_score)
                else:
                    hybrid_results[chunk_id] = {
                        "semantic_score": 0.0,
                        "bm25_score": float(bm25_score),
                        "chunk_id": chunk_id,
                    }

        # Calculate hybrid scores
        for result in hybrid_results.values():
            # Normalize BM25 scores (0-1 range)
            max_bm25 = max([r["bm25_score"] for r in hybrid_results.values()])
            if max_bm25 > 0:
                result["bm25_score"] = result["bm25_score"] / max_bm25

            result["hybrid_score"] = (
                alpha * result["semantic_score"] + (1 - alpha) * result["bm25_score"]
            )

        # Sort by hybrid score and take top-k
        sorted_results = sorted(
            hybrid_results.values(), key=lambda x: x["hybrid_score"], reverse=True
        )[:top_k]

        # Convert to ChunkResult objects
        chunk_results = []
        for result in sorted_results:
            chunk_id = result["chunk_id"]
            if chunk_id in self.chunks:
                chunk_data = self.chunks[chunk_id]

                # Filter by world_id if specified
                if world_id and chunk_data["metadata"].get("world_id") != world_id:
                    continue

                chunk_results.append(
                    ChunkResult(
                        chunk_id=chunk_id,
                        doc_id=chunk_data["doc_id"],
                        text=chunk_data["text"],
                        score=result["hybrid_score"],
                        section_title=chunk_data.get("section_title", ""),
                        metadata=chunk_data["metadata"],
                    )
                )

        print(f"[rag] Retrieved {len(chunk_results)} results")
        return chunk_results

    def _tokenize_chinese(self, text: str) -> List[str]:
        """Simple Chinese tokenization for BM25"""
        # Remove punctuation and split by characters + spaces
        text = re.sub(r"[^\w\s]", " ", text)
        # Split by whitespace and individual characters
        tokens = []
        for word in text.split():
            if re.search(r"[\u4e00-\u9fff]", word):  # Contains Chinese
                tokens.extend(list(word))  # Character-level for Chinese
            else:
                tokens.append(word)  # Word-level for English
        return [t for t in tokens if t.strip()]

    def _normalize_query(self, query: str) -> str:
        """Normalize query with Traditional/Simplified conversion"""
        # Basic cleaning
        query = re.sub(r"\s+", " ", query).strip()
        # Convert to Traditional Chinese for consistency
        try:
            query = self.cc_converter.convert(query)
        except:
            pass
        return query

    def _save_index(self):
        """Save index to disk"""
        try:
            # Save FAISS index
            if self.faiss_index:
                faiss.write_index(self.faiss_index, str(self.index_dir / "faiss.index"))

            # Save metadata
            with open(self.index_dir / "metadata.pkl", "wb") as f:
                pickle.dump(
                    {
                        "documents": self.documents,
                        "chunks": self.chunks,
                        "chunk_embeddings": self.chunk_embeddings,
                        "chunk_id_list": self.chunk_id_list,
                        "bm25_corpus": self.bm25_corpus,
                    },
                    f,
                )

            print(f"[rag] Index saved to {self.index_dir}")
        except Exception as e:
            print(f"[rag] Error saving index: {e}")

    def _load_index(self):
        """Load existing index from disk"""
        try:
            faiss_path = self.index_dir / "faiss.index"
            metadata_path = self.index_dir / "metadata.pkl"

            if faiss_path.exists() and metadata_path.exists():
                # Load FAISS
                self.faiss_index = faiss.read_index(str(faiss_path))

                # Load metadata
                with open(metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data["documents"]
                    self.chunks = data["chunks"]
                    self.chunk_embeddings = data["chunk_embeddings"]
                    self.chunk_id_list = data["chunk_id_list"]
                    self.bm25_corpus = data["bm25_corpus"]

                # Rebuild BM25
                if self.bm25_corpus:
                    self.bm25_index = BM25Okapi(self.bm25_corpus)

                print(f"[rag] Loaded existing index: {len(self.chunks)} chunks")
        except Exception as e:
            print(f"[rag] Error loading index: {e}")


if __name__ == "__name__":
    # Initialize RAG engine
    rag_engine = ChineseRAGEngine(APP_DIRS["RAG_INDEX"])
