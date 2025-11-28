"""
Context-Aware Retrieval for Story Memory

Provides specialized RAG search with:
- Session-based filtering
- Recency weighting
- Relevance scoring
- Multi-query expansion
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContextRetrievalConfig:
    """Configuration for context retrieval"""
    top_k: int = 5
    recency_weight: float = 0.3  # Weight for recency in scoring
    relevance_weight: float = 0.7  # Weight for semantic relevance
    min_score_threshold: float = 0.3  # Minimum score to include result
    include_turn_memories: bool = True
    include_summaries: bool = True


class StoryContextRetriever:
    """
    Context-aware retrieval for story memories

    Enhances basic RAG search with:
    - Session filtering (only search within current story)
    - Recency bias (recent events weighted higher)
    - Type-based filtering (turn memories vs summaries)
    - Score normalization
    """

    def __init__(self, rag_engine=None):
        """
        Initialize context retriever

        Args:
            rag_engine: RAG engine instance (will be lazy-loaded if None)
        """
        self._rag_engine = rag_engine

    @property
    def rag_engine(self):
        """Lazy load RAG engine"""
        if self._rag_engine is None:
            try:
                from core.rag import get_rag_engine
                self._rag_engine = get_rag_engine()
            except Exception as e:
                logger.warning(f"Failed to load RAG engine: {e}")
                self._rag_engine = None
        return self._rag_engine

    async def search_story_context(
        self,
        session_id: str,
        query: str,
        config: Optional[ContextRetrievalConfig] = None
    ) -> List[Dict[str, Any]]:
        """
        Search story context with session filtering

        Args:
            session_id: Story session ID to filter by
            query: Search query
            config: Retrieval configuration

        Returns:
            List of relevant context results with scores
        """
        if not self.rag_engine:
            logger.warning("RAG engine not available")
            return []

        config = config or ContextRetrievalConfig()

        try:
            # Build metadata filter
            metadata_filter = {"session_id": session_id}

            # Search RAG
            raw_results = await self.rag_engine.search(
                query=query,
                top_k=config.top_k * 2,  # Get more, then filter
                filter_metadata=metadata_filter
            )

            # Process and score results
            processed_results = []
            for result in raw_results:
                # Extract metadata
                metadata = result.document.metadata
                result_type = metadata.get("type", "unknown")

                # Type filtering
                if result_type == "turn_memory" and not config.include_turn_memories:
                    continue
                if result_type == "memory_summary" and not config.include_summaries:
                    continue

                # Calculate combined score
                relevance_score = float(result.score)
                recency_score = self._calculate_recency_score(metadata)

                combined_score = (
                    relevance_score * config.relevance_weight +
                    recency_score * config.recency_weight
                )

                # Threshold filtering
                if combined_score < config.min_score_threshold:
                    continue

                processed_results.append({
                    "content": result.document.content,
                    "relevance_score": relevance_score,
                    "recency_score": recency_score,
                    "combined_score": combined_score,
                    "type": result_type,
                    "metadata": metadata
                })

            # Sort by combined score
            processed_results.sort(key=lambda x: x["combined_score"], reverse=True)

            # Return top-k
            return processed_results[:config.top_k]

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []

    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate recency score based on turn number

        More recent turns get higher scores (exponential decay)
        """
        turn_number = metadata.get("turn_number")
        if turn_number is None:
            # For summaries, use average of turn range
            turn_range = metadata.get("turn_range", "0-0")
            try:
                start, end = map(int, turn_range.split("-"))
                turn_number = (start + end) // 2
            except (ValueError, AttributeError):
                turn_number = 0

        # Exponential decay: score = e^(-0.1 * age)
        # Recent turns (low turn_number) get scores close to 1.0
        # Older turns decay exponentially
        import math
        max_turn = 100  # Assume max 100 turns for normalization
        age = max(0, max_turn - turn_number)
        recency_score = math.exp(-0.05 * age / max_turn)

        return recency_score

    async def retrieve_contextual_prompt(
        self,
        session_id: str,
        player_input: str,
        config: Optional[ContextRetrievalConfig] = None
    ) -> str:
        """
        Retrieve relevant context and format as prompt injection

        Args:
            session_id: Story session ID
            player_input: Player's current input
            config: Retrieval configuration

        Returns:
            Formatted context string for LLM prompt
        """
        results = await self.search_story_context(session_id, player_input, config)

        if not results:
            return ""

        # Format context for prompt
        context_parts = ["[相關記憶]"]

        for idx, result in enumerate(results, 1):
            result_type = result["type"]
            content = result["content"]

            if result_type == "turn_memory":
                context_parts.append(f"{idx}. 過去回合: {content}")
            elif result_type == "memory_summary":
                context_parts.append(f"{idx}. 歷史總結: {content}")
            else:
                context_parts.append(f"{idx}. {content}")

        context_parts.append("[記憶結束]\n")

        return "\n".join(context_parts)

    async def expand_query(self, original_query: str) -> List[str]:
        """
        Expand query with variations for better recall

        Args:
            original_query: Original search query

        Returns:
            List of query variations
        """
        queries = [original_query]

        # Simple expansion strategies
        # 1. Extract key actions/entities
        action_keywords = ["攻擊", "逃跑", "對話", "探索", "使用"]
        for keyword in action_keywords:
            if keyword in original_query:
                queries.append(keyword)

        # 2. Extract locations
        location_keywords = ["森林", "城堡", "地城", "村莊", "山脈"]
        for keyword in location_keywords:
            if keyword in original_query:
                queries.append(keyword)

        # Limit to 3 queries to avoid noise
        return queries[:3]

    async def search_with_expansion(
        self,
        session_id: str,
        query: str,
        config: Optional[ContextRetrievalConfig] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with query expansion for better recall

        Args:
            session_id: Story session ID
            query: Original query
            config: Retrieval configuration

        Returns:
            Combined results from expanded queries
        """
        # Expand queries
        expanded_queries = await self.expand_query(query)

        # Search with each query
        all_results = []
        seen_content = set()

        for expanded_query in expanded_queries:
            results = await self.search_story_context(session_id, expanded_query, config)

            for result in results:
                content = result["content"]
                if content not in seen_content:
                    all_results.append(result)
                    seen_content.add(content)

        # Re-sort by combined score
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)

        # Return top-k
        config = config or ContextRetrievalConfig()
        return all_results[:config.top_k]


# Singleton instance
_context_retriever: Optional[StoryContextRetriever] = None


def get_context_retriever(rag_engine=None) -> StoryContextRetriever:
    """Get or create singleton context retriever"""
    global _context_retriever
    if _context_retriever is None:
        _context_retriever = StoryContextRetriever(rag_engine)
    return _context_retriever
