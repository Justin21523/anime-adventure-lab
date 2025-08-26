# core/rag/memory.py
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from .parsers import get_parser
from .embedders import EmbeddingModel
from .retrievers import VectorRetriever
from .rerankers import SimpleReranker

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")


class DocumentMemory:
    """Manage document storage and retrieval pipeline"""

    def __init__(self, collection_name: str = "default"):
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
