# tests/test_story_engine.py
"""
Story Engine Unit Tests
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path

from core.story.engine import StoryEngine
from core.story.game_state import GameSession, GameState, PlayerStats
from core.story.persona import GamePersona
from core.exceptions import GameError, SessionNotFoundError, InvalidChoiceError


class TestStoryEngine:

    @pytest.fixture
    def story_engine(self, tmp_path):
        """Create story engine for testing"""
        return StoryEngine(config_dir=tmp_path)

    def test_create_session(self, story_engine):
        """Test creating new game session"""
        session = story_engine.create_session(
            player_name="測試玩家", persona_id="wise_sage", setting="fantasy"
        )

        assert session.player_name == "測試玩家"
        assert session.persona_id == "wise_sage"
        assert session.turn_count == 0
        assert session.is_active is True
        assert isinstance(session.stats, PlayerStats)

    def test_get_session(self, story_engine):
        """Test getting existing session"""
        session = story_engine.create_session("測試玩家", "wise_sage")
        retrieved = story_engine.get_session(session.session_id)

        assert retrieved.session_id == session.session_id
        assert retrieved.player_name == session.player_name

    def test_session_not_found(self, story_engine):
        """Test getting non-existent session"""
        with pytest.raises(SessionNotFoundError):
            story_engine.get_session("invalid_session_id")

    @pytest.mark.asyncio
    async def test_process_turn(self, story_engine):
        """Test processing game turn"""
        session = story_engine.create_session("測試玩家", "wise_sage")

        response = await story_engine.process_turn(
            session_id=session.session_id, player_input="我想探索森林", choice_id=None
        )

        assert "session_id" in response
        assert "narrative" in response
        assert "choices" in response
        assert "stats" in response
        assert response["turn_count"] > 0

    def test_list_sessions(self, story_engine):
        """Test listing game sessions"""
        # Create test sessions
        session1 = story_engine.create_session("玩家1", "wise_sage")
        session2 = story_engine.create_session("玩家2", "mischievous_fairy")

        sessions = story_engine.list_sessions()
        assert len(sessions) >= 2

        session_ids = [s["session_id"] for s in sessions]
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids

    def test_end_session(self, story_engine):
        """Test ending game session"""
        session = story_engine.create_session("測試玩家", "wise_sage")
        story_engine.end_session(session.session_id)

        updated_session = story_engine.get_session(session.session_id)
        assert updated_session.is_active is False

    def test_list_personas(self, story_engine):
        """Test listing available personas"""
        personas = story_engine.list_personas()
        assert len(personas) > 0

        persona_ids = [p["persona_id"] for p in personas]
        assert "wise_sage" in persona_ids


if __name__ == "__main__":
    pytest.main([__file__])
