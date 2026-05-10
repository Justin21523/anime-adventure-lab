import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_story_turn_job_enqueue_and_sync_fallback(tmp_path, monkeypatch):
    from core.train.job_manager import TrainJobManager

    import api.routers.jobs as jobs_router
    import api.routers.story as story_router

    from schemas.story import StoryTurnRequest, StoryTurnResponse

    # Isolate job storage to a temp dir for tests
    isolated = TrainJobManager(cache_root=str(tmp_path))
    jobs_router.job_manager = isolated
    story_router.story_turn_job_manager = isolated

    class StubSession:
        def __init__(self, session_id: str):
            self.session_id = session_id

    class StubEngine:
        def get_session(self, session_id: str):
            return StubSession(session_id)

    monkeypatch.setattr(story_router, "get_story_engine", lambda: StubEngine())

    async def _stub_process_story_turn(request: StoryTurnRequest):
        return StoryTurnResponse(
            session_id=request.session_id,
            world_id="default",
            turn_count=1,
            narrative="ok",
            choices=[],
            stats={},
            inventory=[],
            flags={},
            agent_used=False,
            agent_overlay=None,
            agent_actions=None,
            knowledge_used=None,
            context=None,
            scene_image_job_id=None,
            scene_image=None,
        )

    monkeypatch.setattr(story_router, "process_story_turn", _stub_process_story_turn)

    resp = await story_router.enqueue_story_turn_job(
        StoryTurnRequest(
            session_id="smoke_session",
            player_input="hello",
            include_image=False,
            use_agent=False,
        )
    )
    assert resp.job_id

    job = await jobs_router.get_job_status(resp.job_id)
    assert job["job_id"] == resp.job_id
    assert job["status"] == "completed"
    assert isinstance(job.get("result"), dict)
    assert job["result"]["narrative"] == "ok"
