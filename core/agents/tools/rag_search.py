"""
RAG search tool for agents.
Uses core.rag.engine to retrieve context snippets.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _get_rag_engine():
    try:
        from core.rag.engine import get_rag_engine

        return get_rag_engine()
    except Exception as e:
        logger.error(f"Failed to load RAG engine: {e}")
        return None


async def rag_search(session_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search RAG index and return snippets

    Args:
        session_id: Story session ID (reserved; may be used for audit/logging)
        params: Dictionary with:
            - query: str - Search query
            - top_k: int (optional) - Number of results (default 5)
            - min_score: float (optional) - Minimum similarity score (default 0.3)
            - world_id: str (optional) - Filter by metadata.world_id

    Returns:
        Dictionary with search results
    """
    query = params.get("query", "")
    top_k = params.get("top_k", 5)
    min_score = params.get("min_score", 0.3)
    world_id = params.get("world_id")
    enable_rerank = params.get("enable_rerank", None)
    rerank_top_k = params.get("rerank_top_k", None)

    if not query or not query.strip():
        return {"success": False, "error": "Query cannot be empty"}

    rag_engine = _get_rag_engine()
    if rag_engine is None:
        return {"success": False, "error": "RAG engine unavailable"}

    try:
        resolved_world_id: Optional[str] = None
        if world_id is not None and str(world_id).strip():
            resolved_world_id = str(world_id).strip()

        resolved_enable_rerank: Optional[bool] = None
        if enable_rerank is not None:
            resolved_enable_rerank = bool(enable_rerank)

        resolved_rerank_top_k: Optional[int] = None
        if rerank_top_k is not None:
            try:
                resolved_rerank_top_k = int(rerank_top_k)
            except Exception:
                resolved_rerank_top_k = None

        results = rag_engine.search(
            query.strip(),
            top_k=int(top_k),
            min_score=float(min_score),
            world_id=resolved_world_id,
            enable_rerank=resolved_enable_rerank,
            rerank_top_k=resolved_rerank_top_k,
        )
        formatted = []
        for r in results:
            metadata = r.document.metadata or {}
            formatted.append(
                {
                    "doc_id": r.document.doc_id,
                    "score": r.score,
                    "semantic_score": getattr(r, "semantic_score", None),
                    "bm25_score": getattr(r, "bm25_score", None),
                    "combined_score": getattr(r, "combined_score", None),
                    "rerank_score": getattr(r, "rerank_score", None),
                    "content": r.document.content,
                    "metadata": metadata,
                }
            )
        return {
            "success": True,
            "query": query,
            "results": formatted,
            "results_count": len(formatted),
        }
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return {"success": False, "error": str(e)}
