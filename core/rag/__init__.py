# core/rag/__init__.py
# Shared Cache Bootstrap
import os, pathlib, torch, json, hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Shared cache setup
AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)

# App directories
APP_DIRS = {
    "RAG_INDEX": f"{AI_CACHE_ROOT}/rag/indexes",
    "RAG_DOCS": f"{AI_CACHE_ROOT}/rag/documents",
    "RAG_EMBEDDINGS": f"{AI_CACHE_ROOT}/rag/embeddings",
    "WORLDPACKS": f"{AI_CACHE_ROOT}/worldpacks",
}
for p in APP_DIRS.values():
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

print(f"[cache] {AI_CACHE_ROOT} | GPU: {torch.cuda.is_available()}")
print(f"[rag] Indexes: {APP_DIRS['RAG_INDEX']}")

# Cell 2: Dependencies & Model Setup

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
