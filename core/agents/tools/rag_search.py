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


async def rag_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search RAG index and return snippets."""
    if not query or not query.strip():
        return {"success": False, "error": "Query cannot be empty"}

    rag_engine = _get_rag_engine()
    if rag_engine is None:
        return {"success": False, "error": "RAG engine unavailable"}

    try:
        results = rag_engine.search(query.strip(), top_k=top_k)
        formatted = []
        for r in results:
            formatted.append(
                {
                    "doc_id": r.document.doc_id,
                    "score": r.score,
                    "content": r.document.content,
                    "metadata": r.document.metadata,
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
