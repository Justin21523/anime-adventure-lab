import sys
import types
from datetime import datetime

import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_story_turn_job_persists_pipeline_artifacts(tmp_path, monkeypatch):
    from core.train.job_manager import TrainJobManager

    import api.routers.jobs as jobs_router
    import api.routers.story as story_router

    from schemas.story import StoryTurnRequest

    isolated = TrainJobManager(cache_root=str(tmp_path))
    monkeypatch.setattr(jobs_router, "job_manager", isolated, raising=False)
    monkeypatch.setattr(story_router, "story_turn_job_manager", isolated, raising=False)

    # Force sync fallback (avoid depending on Celery/Redis availability).
    dummy = types.ModuleType("workers.tasks.story")
    monkeypatch.setitem(sys.modules, "workers.tasks.story", dummy)

    class StubStats:
        def to_dict(self):
            return {}

    class StubState:
        def __init__(self):
            self.scene_id = "opening"
            self.flags = {}
            self.available_choices = []
            self.story_context = {"rag_auto": False, "rag_available": False, "enrich_with_rag": False}

    class StubSession:
        def __init__(self, session_id: str):
            self.session_id = session_id
            self.player_name = "Alice"
            self.persona_id = "wise_sage"
            self.world_id = "default"
            self.stats = StubStats()
            self.inventory = []
            self.current_state = StubState()
            self.history = []

    session = StubSession("smoke_pipeline_session")

    class StubEngine:
        def get_session(self, session_id: str):
            assert session_id == session.session_id
            return session

        async def process_turn(self, session_id: str, player_input: str, choice_id=None):
            session.history.append(
                {
                    "turn": 0,
                    "timestamp": datetime.utcnow().isoformat(),
                    "player_input": player_input,
                    "ai_response": "ok",
                    "choice_id": choice_id,
                    "scene_id": "opening",
                }
            )
            return {
                "session_id": session_id,
                "turn_count": 1,
                "narrative": "ok",
                "choices": [],
                "stats": {},
                "inventory": [],
                "scene_id": "opening",
                "flags": {},
            }

        def get_session_context(self, _session_id: str):
            return {"present_characters": [], "current_scene": {"name": "opening"}}

        def save_session(self, _session):
            return None

    monkeypatch.setattr(story_router, "get_story_engine", lambda: StubEngine())
    monkeypatch.setattr(story_router, "_world_has_rag_documents", lambda _wid: False)
    monkeypatch.setattr(story_router, "_world_enable_rerank", lambda _wid: False)

    resp = await story_router.enqueue_story_turn_job(
        StoryTurnRequest(
            session_id=session.session_id,
            player_input="hello",
            include_image=False,
            use_agent=False,
        )
    )
    assert resp.job_id

    job = await jobs_router.get_job_status(resp.job_id)
    assert job["job_id"] == resp.job_id
    assert job["status"] == "completed"

    assert session.history and isinstance(session.history[-1], dict)
    artifacts = session.history[-1].get("artifacts")
    assert isinstance(artifacts, dict)

    job_bucket = artifacts.get("job")
    assert isinstance(job_bucket, dict)
    assert str(job_bucket.get("job_id") or "") == resp.job_id

    stage_events = job_bucket.get("stage_events")
    assert isinstance(stage_events, list)
    assert len(stage_events) >= 1
    assert any(
        isinstance(e, dict) and str(e.get("stage") or "") == "load_session"
        for e in stage_events
    )

