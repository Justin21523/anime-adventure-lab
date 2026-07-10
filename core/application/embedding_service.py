from __future__ import annotations

import hashlib
import math
import os
import re

from core.config import get_config


EMBEDDING_DIMENSIONS = 1024


def deterministic_embedding(
    text: str, dimensions: int = EMBEDDING_DIMENSIONS
) -> list[float]:
    values = [0.0] * dimensions
    for token in re.findall(r"[\w\u3400-\u9fff]+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        values[index] += 1.0 if digest[4] % 2 else -1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def embed_texts(contents: list[str]) -> list[list[float]]:
    mode = os.getenv("RAG_RUNTIME_MODE", "model").strip().lower()
    if mode in {"mock", "deterministic"}:
        return [deterministic_embedding(content) for content in contents]

    from sentence_transformers import SentenceTransformer

    config = get_config().rag
    model = SentenceTransformer(config.embedding_model, device=config.device)
    encoded = model.encode(contents, normalize_embeddings=True)
    results = [list(map(float, row)) for row in encoded]
    if any(len(row) != EMBEDDING_DIMENSIONS for row in results):
        raise RuntimeError("EMBEDDING_DIMENSION_MISMATCH")
    return results


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
