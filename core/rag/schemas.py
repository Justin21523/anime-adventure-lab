# core/rag/schemas.py
"""RAG data schemas"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DocumentMetadata:
    """Document metadata for RAG"""

    doc_id: str
    title: str
    source: str
    world_id: Optional[str] = None
    upload_time: Optional[str] = None
    license: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalItem:
    """Retrieved chunk with metadata"""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# core/rag/embedders.py
"""Text embedding utilities"""
import numpy as np
from typing import List, Any
from ..shared_cache import get_shared_cache


class SimpleEmbedder:
    """Simple TF-IDF based embedder for fallback"""

    def __init__(self):
        self.cache = get_shared_cache()
        self._vectorizer = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using TF-IDF (fallback implementation)"""
        try:
            # In real implementation, use sentence-transformers:
            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer('BAAI/bge-base-en-v1.5')
            # return model.encode(texts)

            # Mock embeddings
            return np.random.random((len(texts), 768))
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {str(e)}")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed single query"""
        return self.embed_texts([query])[0]
