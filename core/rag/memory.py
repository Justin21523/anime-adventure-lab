# core/rag/memory.py
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import json
from .parsers import get_parser
from .embedders import EmbeddingModel
from .retrievers import VectorRetriever
from .rerankers import SimpleReranker
from ..shared_cache import get_shared_cache


class RAGMemory:
    """Simple memory storage for RAG contexts"""

    def __init__(self):
        self.cache = get_shared_cache()
        self.memory_dir = Path(self.cache.get_path("RAG_DOCS")) / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def write_memory(
        self,
        *,
        world_id: str,
        scope: str,
        content: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
    ):
        """Write memory entry"""
        try:
            memory_file = self.memory_dir / f"{world_id}_{scope}.json"

            # Load existing memories
            if memory_file.exists():
                with open(memory_file, "r") as f:
                    memories = json.load(f)
            else:
                memories = []

            # Add new memory
            memory_entry = {
                "doc_id": doc_id,
                "content": content,
                "metadata": metadata or {},
                "timestamp": "2024-01-01T00:00:00",  # Mock timestamp
            }
            memories.append(memory_entry)

            # Save memories
            with open(memory_file, "w") as f:
                json.dump(memories, f, indent=2)

        except Exception as e:
            raise RuntimeError(f"Failed to write memory: {str(e)}")

    def read_memories(self, world_id: str, scope: str) -> List[Dict]:
        """Read memory entries"""
        try:
            memory_file = self.memory_dir / f"{world_id}_{scope}.json"

            if memory_file.exists():
                with open(memory_file, "r") as f:
                    return json.load(f)
            return []

        except Exception as e:
            return []


class DocumentMemory:
    """Manage document storage and retrieval pipeline"""

    def __init__(self, collection_name: str = "default"):
        self.cache = get_shared_cache()
        self.collection_name = collection_name
        self.embedding_model = EmbeddingModel()
        self.retriever = VectorRetriever(self.embedding_model, collection_name)
        self.reranker = SimpleReranker()

    def add_document(self, file_path: str) -> bool:
        """Add a document to the memory"""
        try:
            parser = get_parser(file_path)
            chunks = parser.parse(file_path)

            if chunks:
                self.retriever.add_documents(chunks)
                print(f"Added {len(chunks)} chunks from {file_path}")
                return True
            else:
                print(f"No content extracted from {file_path}")
                return False

        except Exception as e:
            print(f"Failed to add document {file_path}: {e}")
            return False

    def search(
        self, query: str, top_k: int = 5, rerank: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for relevant documents"""

        # Initial retrieval
        candidates = self.retriever.search(query, top_k * 2)  # Get more candidates

        if rerank and candidates:
            # Rerank results
            candidates = self.reranker.rerank(query, candidates, top_k)

        return candidates[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "collection_name": self.collection_name,
            "total_documents": len(self.retriever.documents),
            "embedding_model": self.embedding_model.model_name,
        }
