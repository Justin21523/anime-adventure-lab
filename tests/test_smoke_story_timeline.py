from datetime import datetime

import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_story_session_turn_history_and_template(monkeypatch):
    from api.routers.story import get_story_session

    class StubStats:
        def to_dict(self):
            return {"health": 100, "energy": 100, "level": 1}

    class StubState:
        def __init__(self):
            self.scene_id = "opening"
            self.scene_description = ""
            self.available_choices = []
            self.flags = {}
            self.story_context = {
                "player_template_id": "template_1",
                "rag_auto": True,
                "rag_available": False,
            }

    class StubSession:
        def __init__(self):
            self.session_id = "smoke_session"
            self.player_name = "Alice"
            self.persona_id = "wise_sage"
            self.world_id = "default"
            self.created_at = datetime.now()
            self.updated_at = datetime.now()
            self.turn_count = 1
            self.is_active = True
            self.stats = StubStats()
            self.inventory = []
            self.current_state = StubState()
            self.history = [
                {
                    "turn": 0,
                    "timestamp": datetime.now().isoformat(),
                    "player_input": "hello",
                    "ai_response": "world responds",
                    "choice_id": None,
                    "scene_id": "opening",
                    "agent_used": True,
                    "agent_actions": {
                        "decision_type": "multi_agent_story_director",
                        "reasoning": "stub",
                        "tool_results": [],
                        "overall_success": True,
                        "errors": [],
                    },
                }
            ]

    class StubEngine:
        def get_session(self, session_id: str):
            assert session_id == "smoke_session"
            return StubSession()

    monkeypatch.setattr("api.routers.story.get_story_engine", lambda: StubEngine())
    monkeypatch.setattr("api.routers.story._world_has_rag_documents", lambda _wid: False)

    class StubMemoryManager:
        def get_statistics(self):
            return None

        async def retrieve_relevant_context(self, **_kwargs):
            return None

    monkeypatch.setattr("core.story.memory_manager.get_memory_manager", lambda _sid: StubMemoryManager())

    resp = await get_story_session("smoke_session")
    assert resp.session_id == "smoke_session"
    assert resp.player_template_id == "template_1"
    assert resp.turn_history
    assert resp.turn_history[0].player_input == "hello"

