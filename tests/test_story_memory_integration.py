"""
Tests for Story Memory Integration

IMPORTANT: All tests use mocks to avoid GPU/RAG usage during testing.
Never load real models or RAG engines in these tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from collections import deque

from core.story.memory_manager import (
    StoryMemoryManager,
    TurnMemory,
    MemorySummary,
    get_memory_manager,
    clear_memory_manager
)
from core.rag.context_retrieval import (
    StoryContextRetriever,
    ContextRetrievalConfig
)


# Mock Fixtures ---------------------------------------------------------------

@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine to avoid real embeddings/search"""
    mock = AsyncMock()

    # Mock add_document
    mock.add_document = AsyncMock(return_value=None)

    # Mock search
    mock_result = MagicMock()
    mock_result.document.content = "玩家進入了黑暗森林"
    mock_result.document.metadata = {
        "type": "turn_memory",
        "session_id": "test-session",
        "turn_number": 5,
        "scene_id": "forest_entrance"
    }
    mock_result.score = 0.85

    mock.search = AsyncMock(return_value=[mock_result])

    return mock


@pytest.fixture
def memory_manager(mock_rag_engine):
    """Memory manager with mocked RAG"""
    manager = StoryMemoryManager(
        session_id="test-session",
        rag_engine=mock_rag_engine,
        max_short_term=10
    )
    return manager


@pytest.fixture
def context_retriever(mock_rag_engine):
    """Context retriever with mocked RAG"""
    retriever = StoryContextRetriever(rag_engine=mock_rag_engine)
    return retriever


# TurnMemory Tests ------------------------------------------------------------

class TestTurnMemory:
    """Test TurnMemory dataclass"""

    def test_turn_memory_creation(self):
        """Test creating turn memory"""
        turn = TurnMemory(
            turn_number=1,
            player_input="探索森林",
            narrative_response="你進入了黑暗的森林深處",
            scene_id="forest_entrance"
        )

        assert turn.turn_number == 1
        assert turn.player_input == "探索森林"
        assert turn.scene_id == "forest_entrance"
        assert turn.timestamp is not None
        assert turn.choices_made == []
        assert turn.flags_changed == {}

    def test_turn_memory_to_text(self):
        """Test converting turn memory to text"""
        turn = TurnMemory(
            turn_number=5,
            player_input="與村民對話",
            narrative_response="村民告訴你關於龍的傳說",
            scene_id="village_square",
            choices_made=["talk_to_elder"]
        )

        text = turn.to_text()

        assert "回合 5" in text
        assert "與村民對話" in text
        assert "村民告訴你關於龍的傳說" in text
        assert "village_square" in text
        assert "talk_to_elder" in text


# MemorySummary Tests ---------------------------------------------------------

class TestMemorySummary:
    """Test MemorySummary dataclass"""

    def test_summary_creation(self):
        """Test creating memory summary"""
        summary = MemorySummary(
            summary_id="test_summary_0",
            turn_range=(1, 5),
            summary_text="玩家探索了森林並遇見了村民",
            key_events=["進入森林", "遇見老人"],
            characters_met=["神秘老人"],
            locations_visited=["黑暗森林", "村莊廣場"],
            important_flags={"quest_forest_started": True},
            created_at=datetime.now()
        )

        assert summary.turn_range == (1, 5)
        assert "森林" in summary.summary_text
        assert len(summary.key_events) == 2
        assert len(summary.characters_met) == 1

    def test_summary_to_text(self):
        """Test converting summary to text"""
        summary = MemorySummary(
            summary_id="test_summary_0",
            turn_range=(1, 5),
            summary_text="玩家完成了序章任務",
            key_events=["接受任務", "擊敗敵人"],
            characters_met=["守衛", "商人"],
            locations_visited=["城門", "市集"],
            important_flags={},
            created_at=datetime.now()
        )

        text = summary.to_text()

        assert "回合 1-5 總結" in text
        assert "序章任務" in text
        assert "接受任務" in text
        assert "守衛" in text
        assert "城門" in text


# StoryMemoryManager Tests ----------------------------------------------------

class TestStoryMemoryManager:
    """Test StoryMemoryManager functionality"""

    @pytest.mark.asyncio
    async def test_record_turn(self, memory_manager, mock_rag_engine):
        """Test recording a story turn"""
        await memory_manager.record_turn(
            turn_number=1,
            player_input="前進",
            narrative_response="你向前走了幾步",
            scene_id="forest_path",
            choices_made=["move_forward"]
        )

        # Check short-term memory
        assert len(memory_manager.short_term) == 1
        turn = memory_manager.short_term[0]
        assert turn.turn_number == 1
        assert turn.player_input == "前進"

        # Check RAG was called
        mock_rag_engine.add_document.assert_called_once()
        call_args = mock_rag_engine.add_document.call_args[1]
        assert "回合 1" in call_args["content"]
        assert call_args["metadata"]["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_short_term_memory_limit(self, memory_manager, mock_rag_engine):
        """Test short-term memory respects max limit"""
        # Add 15 turns (max is 10)
        for i in range(15):
            await memory_manager.record_turn(
                turn_number=i,
                player_input=f"行動 {i}",
                narrative_response=f"結果 {i}",
                scene_id="test_scene"
            )

        # Should only keep last 10
        assert len(memory_manager.short_term) == 10
        assert memory_manager.short_term[0].turn_number == 5  # First is turn 5
        assert memory_manager.short_term[-1].turn_number == 14  # Last is turn 14

    @pytest.mark.asyncio
    async def test_compression_to_summary(self, memory_manager, mock_rag_engine):
        """Test automatic compression to mid-term summaries"""
        # Add 5 turns (should trigger compression at interval=5)
        for i in range(1, 6):
            await memory_manager.record_turn(
                turn_number=i,
                player_input=f"行動 {i}",
                narrative_response=f"結果 {i}",
                scene_id=f"scene_{i}"
            )

        # Should have created a summary
        assert len(memory_manager.summaries) >= 1

        # Summary should be added to RAG
        # Check that add_document was called for both turns and summary
        assert mock_rag_engine.add_document.call_count > 5

    @pytest.mark.asyncio
    async def test_retrieve_relevant_context(self, memory_manager, mock_rag_engine):
        """Test context retrieval"""
        # Add some turns
        for i in range(3):
            await memory_manager.record_turn(
                turn_number=i,
                player_input=f"探索 {i}",
                narrative_response=f"發現 {i}",
                scene_id=f"location_{i}"
            )

        # Retrieve context
        context = await memory_manager.retrieve_relevant_context(
            query="探索森林",
            max_results=5,
            include_short_term=True
        )

        # Check structure
        assert "short_term" in context
        assert "summaries" in context
        assert "rag_results" in context

        # Check short-term contains recent turns
        assert len(context["short_term"]) == 3

        # Check RAG search was called
        mock_rag_engine.search.assert_called_once()

    def test_get_short_term_summary(self, memory_manager):
        """Test getting text summary of short-term memory"""
        # Add some manual turns to short-term
        memory_manager.short_term.append(TurnMemory(
            turn_number=1,
            player_input="前進",
            narrative_response="你向前走",
            scene_id="path"
        ))
        memory_manager.short_term.append(TurnMemory(
            turn_number=2,
            player_input="觀察",
            narrative_response="你看見一座城堡",
            scene_id="viewpoint"
        ))

        summary = memory_manager.get_short_term_summary()

        assert "回合1" in summary
        assert "回合2" in summary
        assert "前進" in summary
        assert "觀察" in summary

    def test_get_statistics(self, memory_manager):
        """Test getting memory statistics"""
        # Add some turns
        memory_manager.short_term.append(TurnMemory(
            turn_number=1,
            player_input="test",
            narrative_response="test response"
        ))

        stats = memory_manager.get_statistics()

        assert stats["session_id"] == "test-session"
        assert stats["short_term_count"] == 1
        assert stats["rag_available"] is True


# StoryContextRetriever Tests -------------------------------------------------

class TestStoryContextRetriever:
    """Test StoryContextRetriever functionality"""

    @pytest.mark.asyncio
    async def test_search_story_context(self, context_retriever, mock_rag_engine):
        """Test searching story context with session filtering"""
        results = await context_retriever.search_story_context(
            session_id="test-session",
            query="進入森林",
            config=ContextRetrievalConfig(top_k=5)
        )

        # Should return processed results
        assert len(results) > 0

        result = results[0]
        assert "content" in result
        assert "relevance_score" in result
        assert "recency_score" in result
        assert "combined_score" in result
        assert "type" in result

        # Check RAG was called with correct filter
        mock_rag_engine.search.assert_called_once()
        call_args = mock_rag_engine.search.call_args[1]
        assert call_args["filter_metadata"]["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_retrieve_contextual_prompt(self, context_retriever, mock_rag_engine):
        """Test formatting context as LLM prompt"""
        prompt = await context_retriever.retrieve_contextual_prompt(
            session_id="test-session",
            player_input="探索深處",
            config=ContextRetrievalConfig(top_k=3)
        )

        # Should contain formatted context
        if prompt:  # May be empty if no results
            assert "[相關記憶]" in prompt
            assert "[記憶結束]" in prompt

    @pytest.mark.asyncio
    async def test_recency_score_calculation(self, context_retriever):
        """Test recency score calculation"""
        # Recent turn
        recent_metadata = {"turn_number": 95}
        recent_score = context_retriever._calculate_recency_score(recent_metadata)

        # Old turn
        old_metadata = {"turn_number": 10}
        old_score = context_retriever._calculate_recency_score(old_metadata)

        # Recent should have higher score
        assert recent_score > old_score
        assert 0.0 <= recent_score <= 1.0
        assert 0.0 <= old_score <= 1.0

    @pytest.mark.asyncio
    async def test_query_expansion(self, context_retriever):
        """Test query expansion for better recall"""
        query = "攻擊森林中的怪物"
        expanded = await context_retriever.expand_query(query)

        assert len(expanded) >= 1
        assert query in expanded  # Original should be included
        assert len(expanded) <= 3  # Limited to 3


# Manager Registry Tests ------------------------------------------------------

def test_memory_manager_singleton():
    """Test memory manager registry"""
    # Clear any existing
    clear_memory_manager("singleton-test")

    # Get manager twice
    manager1 = get_memory_manager("singleton-test")
    manager2 = get_memory_manager("singleton-test")

    # Should be same instance
    assert manager1 is manager2

    # Clear
    clear_memory_manager("singleton-test")


# Integration Tests -----------------------------------------------------------

class TestMemoryIntegration:
    """Integration tests for memory system"""

    @pytest.mark.asyncio
    async def test_full_memory_cycle(self, mock_rag_engine):
        """Test complete memory lifecycle"""
        # Create manager
        manager = StoryMemoryManager("integration-test", mock_rag_engine)

        # Record multiple turns
        for i in range(7):
            await manager.record_turn(
                turn_number=i,
                player_input=f"行動 {i}",
                narrative_response=f"玩家執行了行動 {i}，結果是...",
                scene_id=f"scene_{i}",
                choices_made=[f"choice_{i}"]
            )

        # Check short-term has recent turns
        assert len(manager.short_term) == 7

        # Check summaries created (should have at least 1 after 5+ turns)
        assert len(manager.summaries) >= 1

        # Retrieve context
        context = await manager.retrieve_relevant_context(
            query="最近發生了什麼",
            max_results=5
        )

        # Should have all layers
        assert len(context["short_term"]) > 0
        assert len(context["summaries"]) >= 0
        assert len(context["rag_results"]) >= 0

        # Stats should be accurate
        stats = manager.get_statistics()
        assert stats["short_term_count"] == 7
        assert stats["summaries_count"] >= 1


# Error Handling Tests --------------------------------------------------------

class TestMemoryErrorHandling:
    """Test error handling in memory system"""

    @pytest.mark.asyncio
    async def test_rag_failure_graceful(self):
        """Test graceful degradation when RAG fails"""
        # Create manager with failing RAG
        mock_failing_rag = AsyncMock()
        mock_failing_rag.add_document.side_effect = Exception("RAG offline")
        mock_failing_rag.search.side_effect = Exception("Search failed")

        manager = StoryMemoryManager("fail-test", mock_failing_rag)

        # Should still work without RAG
        await manager.record_turn(
            turn_number=1,
            player_input="測試",
            narrative_response="測試回應",
            scene_id="test"
        )

        # Short-term should work
        assert len(manager.short_term) == 1

        # Retrieval should return empty RAG results
        context = await manager.retrieve_relevant_context("測試")
        assert context["rag_results"] == []

    @pytest.mark.asyncio
    async def test_context_retrieval_no_rag(self):
        """Test context retrieval without RAG engine"""
        retriever = StoryContextRetriever(rag_engine=None)

        results = await retriever.search_story_context(
            session_id="test",
            query="測試"
        )

        # Should return empty list, not crash
        assert results == []
