import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_story_turn_rag_mode_overrides(monkeypatch):
    from api.routers import story as story_router
    from api.routers.story import process_story_turn
    from schemas.story import StoryTurnRequest

    calls = {"rag_search": 0}

    def _stub_world_has_rag_documents(_world_id: str) -> bool:
        return False

    def _stub_safe_rag_search(
        query: str,
        top_k: int = 3,
        world_id: str | None = None,
        *,
        enable_rerank: bool | None = None,
        rerank_top_k: int | None = None,
    ):
        _ = (query, top_k, world_id, enable_rerank, rerank_top_k)
        calls["rag_search"] += 1
        return [{"content": "stub", "score": 1.0, "metadata": {"world_id": world_id or "default"}}]

    monkeypatch.setattr(story_router, "_world_has_rag_documents", _stub_world_has_rag_documents)
    monkeypatch.setattr(story_router, "_safe_rag_search", _stub_safe_rag_search)

    class StubState:
        def __init__(self):
            self.story_context = {}
            self.scene_id = "opening"
            self.flags = {}

    class StubSession:
        def __init__(self):
            self.session_id = "smoke_session"
            self.world_id = "world_a"
            self.current_state = StubState()

    class StubEngine:
        def __init__(self):
            self._session = StubSession()

        def get_session(self, session_id: str):
            assert session_id == self._session.session_id
            return self._session

        async def process_turn(self, session_id: str, player_input: str, choice_id=None):
            _ = (choice_id,)
            return {
                "session_id": session_id,
                "turn_count": 1,
                "narrative": f"ok: {player_input}",
                "choices": [],
                "stats": {},
                "inventory": [],
                "flags": {},
            }

        def get_session_context(self, _session_id: str):
            return {}

        def save_session(self, _session):
            return None

    monkeypatch.setattr("api.routers.story.get_story_engine", lambda: StubEngine())

    # rag_mode=on should force retrieval attempt even if world_has_rag_documents=False
    resp = await process_story_turn(
        StoryTurnRequest(
            session_id="smoke_session",
            player_input="hello",
            rag_mode="on",
            include_image=False,
        )
    )
    assert resp.session_id == "smoke_session"
    assert calls["rag_search"] == 1

    # rag_mode=off should disable retrieval
    resp = await process_story_turn(
        StoryTurnRequest(
            session_id="smoke_session",
            player_input="hello2",
            rag_mode="off",
            include_image=False,
        )
    )
    assert resp.session_id == "smoke_session"
    assert calls["rag_search"] == 1


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_story_turn_rag_mode_can_restore_auto(monkeypatch):
    from api.routers import story as story_router
    from api.routers.story import process_story_turn
    from schemas.story import StoryTurnRequest

    def _stub_world_has_rag_documents(_world_id: str) -> bool:
        return True

    monkeypatch.setattr(story_router, "_world_has_rag_documents", _stub_world_has_rag_documents)
    monkeypatch.setattr(story_router, "_safe_rag_search", lambda *args, **kwargs: [])

    class StubState:
        def __init__(self):
            self.story_context = {"rag_auto": False, "enrich_with_rag": False}
            self.scene_id = "opening"
            self.flags = {}

    class StubSession:
        def __init__(self):
            self.session_id = "smoke_session"
            self.world_id = "world_a"
            self.current_state = StubState()

    class StubEngine:
        def __init__(self):
            self._session = StubSession()

        def get_session(self, session_id: str):
            assert session_id == self._session.session_id
            return self._session

        async def process_turn(self, session_id: str, player_input: str, choice_id=None):
            _ = (player_input, choice_id)
            return {
                "session_id": session_id,
                "turn_count": 1,
                "narrative": "ok",
                "choices": [],
                "stats": {},
                "inventory": [],
                "flags": {},
            }

        def get_session_context(self, _session_id: str):
            return {}

        def save_session(self, _session):
            return None

    engine = StubEngine()
    monkeypatch.setattr("api.routers.story.get_story_engine", lambda: engine)

    # rag_mode=auto should flip rag_auto back to True
    await process_story_turn(
        StoryTurnRequest(
            session_id="smoke_session",
            player_input="hello",
            rag_mode="auto",
            include_image=False,
        )
    )
    assert engine._session.current_state.story_context.get("rag_auto") is True
