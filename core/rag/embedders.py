# core/rag/embedders.py
import os
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")


class EmbeddingModel:
    """Wrapper for sentence embedding models"""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        try:
            cache_dir = f"{AI_CACHE_ROOT}/models/embeddings"
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=cache_dir,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"Failed to load embedding model {self.model_name}: {e}")
            # Fallback to a smaller model
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=len(texts) > 10
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding"""
        return self.encode([text])[0]
