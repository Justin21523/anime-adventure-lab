# core/rag/rerankers.py
from typing import List, Tuple, Dict, Any


class SimpleReranker:
    """Simple text-based reranking"""

    def __init__(self):
        pass

    def rerank(
        self, query: str, documents: List[Tuple[Dict[str, Any], float]], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Rerank documents based on text matching"""

        if not documents:
            return []

        query_words = set(query.lower().split())
        reranked = []

        for doc, score in documents:
            content = doc["content"].lower()
            content_words = set(content.split())

            # Simple word overlap bonus
            overlap = len(query_words.intersection(content_words))
            overlap_bonus = overlap / len(query_words) if query_words else 0

            # Length penalty (prefer shorter, more relevant chunks)
            length_penalty = min(1.0, 200 / len(content.split()))

            # Combined score
            final_score = score + 0.1 * overlap_bonus + 0.05 * length_penalty

            reranked.append((doc, final_score))

        # Sort by final score and return top-k
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
