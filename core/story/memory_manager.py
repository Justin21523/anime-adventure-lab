"""
Story Memory Manager

Implements three-layer memory architecture for story-driven gameplay:
1. Short-term: Recent turns (deque, last 10 turns)
2. Mid-term: Compressed summaries (key events, every 5 turns)
3. Long-term: Vector embeddings in RAG (semantic search)

This allows the AI to remember and reference past events naturally.
"""

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Any, Deque
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# TTL for retrieve_relevant_context cache (seconds)
_RETRIEVE_CACHE_TTL = 15


@dataclass
class TurnMemory:
    """Single turn memory record"""
    turn_number: int
    player_input: str
    narrative_response: str
    scene_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    choices_made: Optional[List[str]] = None
    flags_changed: Optional[Dict[str, Any]] = None
    stats_changed: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.choices_made is None:
            self.choices_made = []
        if self.flags_changed is None:
            self.flags_changed = {}
        if self.stats_changed is None:
            self.stats_changed = {}

    def to_text(self) -> str:
        """Convert turn memory to text for RAG embedding"""
        text_parts = [
            f"回合 {self.turn_number}:",
            f"玩家行動: {self.player_input}",
            f"結果: {self.narrative_response}"
        ]

        if self.scene_id:
            text_parts.append(f"場景: {self.scene_id}")

        if self.choices_made:
            text_parts.append(f"選擇: {', '.join(self.choices_made)}")

        return " | ".join(text_parts)


@dataclass
class MemorySummary:
    """Mid-term memory summary"""
    summary_id: str
    turn_range: tuple  # (start_turn, end_turn)
    summary_text: str
    key_events: List[str]
    characters_met: List[str]
    locations_visited: List[str]
    important_flags: Dict[str, Any]
    created_at: datetime

    def to_text(self) -> str:
        """Convert summary to text for RAG embedding"""
        text_parts = [
            f"回合 {self.turn_range[0]}-{self.turn_range[1]} 總結:",
            self.summary_text
        ]

        if self.key_events:
            text_parts.append(f"重要事件: {'; '.join(self.key_events)}")

        if self.characters_met:
            text_parts.append(f"遇見角色: {', '.join(self.characters_met)}")

        if self.locations_visited:
            text_parts.append(f"造訪地點: {', '.join(self.locations_visited)}")

        return " | ".join(text_parts)


class StoryMemoryManager:
    """
    Manages three-layer memory for story sessions

    Architecture:
    - Short-term: deque(maxlen=10) - recent turns, instant access
    - Mid-term: List of summaries - compressed history every 5 turns
    - Long-term: RAG vector store - semantic search across all history
    """

    def __init__(self, session_id: str, rag_engine=None, max_short_term: int = 10):
        """
        Initialize memory manager

        Args:
            session_id: Story session ID
            rag_engine: RAG engine instance (will be lazy-loaded if None)
            max_short_term: Maximum short-term memories to keep
        """
        self.session_id = session_id
        self._rag_engine = rag_engine
        self.max_short_term = max_short_term

        # Short-term memory (recent turns)
        self.short_term: Deque[TurnMemory] = deque(maxlen=max_short_term)

        # Mid-term memory (summaries)
        self.summaries: List[MemorySummary] = []

        # Compression settings
        self.compression_interval = 5  # Summarize every N turns
        self.turns_since_summary = 0

        # Retrieve cache — avoid redundant RAG searches within TTL
        self._retrieve_cache: Dict[str, Dict[str, Any]] = {}
        self._retrieve_cache_ts: float = 0.0
        self._retrieve_cache_key: str = ""

        logger.info(f"Memory manager initialized for session {session_id}")

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

    async def record_turn(
        self,
        turn_number: int,
        player_input: str,
        narrative_response: str,
        scene_id: Optional[str] = None,
        choices_made: Optional[List[str]] = None,
        flags_changed: Optional[Dict[str, Any]] = None,
        stats_changed: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a story turn into memory

        Args:
            turn_number: Current turn number
            player_input: Player's action/input
            narrative_response: AI's narrative response
            scene_id: Current scene ID
            choices_made: Choices selected this turn
            flags_changed: Flags changed this turn
            stats_changed: Stats changed this turn
        """
        # Create turn memory
        turn_memory = TurnMemory(
            turn_number=turn_number,
            player_input=player_input,
            narrative_response=narrative_response,
            scene_id=scene_id,
            choices_made=choices_made or [],
            flags_changed=flags_changed or {},
            stats_changed=stats_changed or {}
        )

        # Add to short-term memory
        self.short_term.append(turn_memory)
        logger.debug(f"Turn {turn_number} added to short-term memory")

        # Invalidate retrieve cache — fresh turn means stale context
        self._retrieve_cache_key = ""

        # Check if we need to compress to mid-term
        self.turns_since_summary += 1
        if self.turns_since_summary >= self.compression_interval:
            await self._compress_to_summary()
            self.turns_since_summary = 0

        # Add to long-term (RAG) asynchronously
        if self.rag_engine:
            try:
                await self._add_to_long_term(turn_memory)
            except Exception as e:
                logger.warning(f"Failed to add turn to RAG: {e}")

    async def _compress_to_summary(self) -> None:
        """Compress oldest short-term memories into a summary"""
        if len(self.short_term) < 3:
            # Not enough data to summarize
            return

        # Get turns to summarize (first half of short-term)
        turns_to_summarize = list(self.short_term)[:len(self.short_term) // 2]

        if not turns_to_summarize:
            return

        # Extract key information
        turn_range = (turns_to_summarize[0].turn_number, turns_to_summarize[-1].turn_number)

        # Collect key events, characters, locations
        key_events = []
        characters_met = set()
        locations_visited = set()
        important_flags = {}

        for turn in turns_to_summarize:
            # Extract from narrative (simple keyword detection)
            if any(keyword in turn.narrative_response for keyword in ["遇見", "看見", "發現", "meet", "encounter"]):
                key_events.append(f"回合{turn.turn_number}: {turn.player_input}")

            if turn.scene_id and turn.scene_id not in locations_visited:
                locations_visited.add(turn.scene_id)

            # Collect important flag changes
            for flag, value in turn.flags_changed.items():
                if flag.startswith("quest_") or flag.startswith("npc_met_"):
                    important_flags[flag] = value

        # Generate summary text
        summary_text = self._generate_summary_text(turns_to_summarize)

        # Create summary
        summary = MemorySummary(
            summary_id=f"{self.session_id}_summary_{len(self.summaries)}",
            turn_range=turn_range,
            summary_text=summary_text,
            key_events=key_events,
            characters_met=list(characters_met),
            locations_visited=list(locations_visited),
            important_flags=important_flags,
            created_at=datetime.now()
        )

        self.summaries.append(summary)
        logger.info(f"Created summary for turns {turn_range[0]}-{turn_range[1]}")

        # Add summary to RAG
        if self.rag_engine:
            try:
                await self._add_summary_to_rag(summary)
            except Exception as e:
                logger.warning(f"Failed to add summary to RAG: {e}")

    def _generate_summary_text(self, turns: List[TurnMemory]) -> str:
        """Generate summary text from multiple turns"""
        if not turns:
            return ""

        # Simple summary: concatenate key actions
        actions = [f"{turn.player_input}: {turn.narrative_response[:50]}..." for turn in turns]
        return " → ".join(actions)

    async def _add_to_long_term(self, turn_memory: TurnMemory) -> None:
        """Add turn to RAG long-term memory"""
        if not self.rag_engine:
            return

        try:
            self.rag_engine.add_document(
                doc_id=f"{self.session_id}_turn_{turn_memory.turn_number}",
                content=turn_memory.to_text(),
                metadata={
                    "type": "turn_memory",
                    "session_id": self.session_id,
                    "turn_number": turn_memory.turn_number,
                    "scene_id": turn_memory.scene_id,
                    "timestamp": turn_memory.timestamp.isoformat() if turn_memory.timestamp else None
                }
            )
            logger.debug(f"Turn {turn_memory.turn_number} added to RAG")
        except Exception as e:
            logger.error(f"Failed to add turn to RAG: {e}")
            raise

    async def _add_summary_to_rag(self, summary: MemorySummary) -> None:
        """Add summary to RAG long-term memory"""
        if not self.rag_engine:
            return

        try:
            self.rag_engine.add_document(
                doc_id=f"{self.session_id}_summary_{summary.turn_range[0]}_{summary.turn_range[1]}",
                content=summary.to_text(),
                metadata={
                    "type": "memory_summary",
                    "session_id": self.session_id,
                    "turn_range": f"{summary.turn_range[0]}-{summary.turn_range[1]}",
                    "timestamp": summary.created_at.isoformat()
                }
            )
            logger.debug(f"Summary {summary.summary_id} added to RAG")
        except Exception as e:
            logger.error(f"Failed to add summary to RAG: {e}")

    async def retrieve_relevant_context(
        self,
        query: str,
        max_results: int = 5,
        include_short_term: bool = True,
        force_rag: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query

        Args:
            query: Query text (player input or context)
            max_results: Maximum RAG results to retrieve
            include_short_term: Whether to include short-term memories
            force_rag: If True, skip cache and always run RAG search

        Returns:
            Dictionary with short_term, summaries, and rag_results
        """
        # Cache key: query hash + include_short_term flag + short_term length
        cache_key = f"{hash(query)}:{include_short_term}:{len(self.short_term)}"
        now = time.monotonic()

        # Return cached result if still fresh and not forcing RAG
        if (
            not force_rag
            and self._retrieve_cache_key == cache_key
            and (now - self._retrieve_cache_ts) < _RETRIEVE_CACHE_TTL
        ):
            return dict(self._retrieve_cache)

        context = {
            "short_term": [],
            "summaries": [],
            "rag_results": []
        }

        # Short-term memories (always recent, no search needed)
        if include_short_term:
            context["short_term"] = [
                {
                    "turn": turn.turn_number,
                    "action": turn.player_input,
                    "result": turn.narrative_response,
                    "scene": turn.scene_id
                }
                for turn in self.short_term
            ]

        # Recent summaries
        context["summaries"] = [
            {
                "turn_range": f"{s.turn_range[0]}-{s.turn_range[1]}",
                "summary": s.summary_text,
                "key_events": s.key_events
            }
            for s in self.summaries[-3:]  # Last 3 summaries
        ]

        # RAG semantic search — skip if no engine or RAG is not available
        # (avoids lazy-load overhead when RAG is not configured)
        if self.rag_engine:
            try:
                rag_results = await self.rag_engine.search(
                    query=query,
                    top_k=max_results,
                    filter_metadata={"session_id": self.session_id}
                )

                context["rag_results"] = [
                    {
                        "content": result.document.content,
                        "score": float(result.score),
                        "metadata": result.document.metadata
                    }
                    for result in rag_results
                ]
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")

        # Store in cache for subsequent calls within TTL
        self._retrieve_cache = dict(context)
        self._retrieve_cache_key = cache_key
        self._retrieve_cache_ts = now

        return context

    def get_short_term_summary(self) -> str:
        """Get text summary of short-term memories"""
        if not self.short_term:
            return "無近期記憶"

        recent_turns = list(self.short_term)[-3:]  # Last 3 turns
        summary_parts = []

        for turn in recent_turns:
            summary_parts.append(
                f"回合{turn.turn_number}: {turn.player_input} → {turn.narrative_response[:50]}..."
            )

        return "\n".join(summary_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "session_id": self.session_id,
            "short_term_count": len(self.short_term),
            "summaries_count": len(self.summaries),
            "total_turns_covered": (
                self.summaries[-1].turn_range[1] if self.summaries else 0
            ) + len(self.short_term),
            "turns_since_last_summary": self.turns_since_summary,
            "rag_available": self.rag_engine is not None
        }


# Manager registry
_memory_managers: Dict[str, StoryMemoryManager] = {}


def get_memory_manager(session_id: str, rag_engine=None) -> StoryMemoryManager:
    """Get or create memory manager for session"""
    if session_id not in _memory_managers:
        _memory_managers[session_id] = StoryMemoryManager(session_id, rag_engine)
    return _memory_managers[session_id]


def clear_memory_manager(session_id: str) -> None:
    """Clear memory manager for session"""
    if session_id in _memory_managers:
        del _memory_managers[session_id]
        logger.info(f"Cleared memory manager for session {session_id}")
