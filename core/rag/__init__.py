# core/rag/__init__.py
"""
Retrieval-Augmented Generation components
"""
from .parsers import DocumentParser
from .embedders import EmbeddingModel
from .retrievers import VectorRetriever
from .rerankers import SimpleReranker
from .memory import DocumentMemory

__all__ = [
    "DocumentParser",
    "EmbeddingModel",
    "VectorRetriever",
    "SimpleReranker",
    "DocumentMemory",
]

# Cell 2: Dependencies & Model Setup

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import torch
import opencc
import re
from pathlib import Path
import pickle
from urllib.parse import quote
import uuid


# Initialize models with low-VRAM options
def setup_rag_models():
    """Setup RAG models with VRAM optimization"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Chinese-English embedding model
    embedding_model = SentenceTransformer(
        "BAAI/bge-m3",
        device=device,
        cache_folder=f"{AI_CACHE_ROOT}/hf/sentence-transformers",
    )

    # Traditional/Simplified Chinese converter
    cc_converter = opencc.OpenCC("s2t")  # Simplified to Traditional

    print(f"[models] Embedding: bge-m3 on {device}")
    print(f"[models] OpenCC converter ready")

    return embedding_model, cc_converter


embed_model, cc_converter = setup_rag_models()
