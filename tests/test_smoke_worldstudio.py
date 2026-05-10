import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_sync_worldpack_endpoint(monkeypatch):
    from api.routers.story import sync_story_worldpack
    from schemas.story import StoryWorldSyncRequest

    class StubEngine:
        def sync_worldpack_into_session(self, session_id: str, *, mode: str = "add_only"):
            return {
                "session_id": session_id,
                "world_id": "default",
                "mode": mode,
                "flags_added": ["quest_smoke_started"],
                "flags_updated": [],
                "characters_added": ["smoke_npc"],
                "characters_updated": [],
                "worldpack_updated_at": "2025-01-01T00:00:00",
            }

    monkeypatch.setattr("api.routers.story.get_story_engine", lambda: StubEngine())

    resp = await sync_story_worldpack("smoke_session", StoryWorldSyncRequest(mode="add_only"))
    assert resp.session_id == "smoke_session"
    assert resp.world_id == "default"
    assert resp.mode == "add_only"
    assert "quest_smoke_started" in resp.flags_added


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_world_agents_suggest_endpoint(monkeypatch):
    from api.routers.worlds import suggest_worldpack_update
    from core.agents import world_studio_orchestrator as wso
    from schemas.world import WorldAgentSuggestRequest

    def _stub_init(self, *args, **kwargs):
        _ = (args, kwargs)
        self.llm_adapter = None
        self.max_llm_calls = 0
        self._llm_calls = 0

    def _stub_suggest(self, **_kwargs):
        return {
            "patch": {"world_flags": {"quest_smoke_started": True}},
            "contributors": [
                {
                    "agent": "smoke_agent",
                    "reasoning": "smoke test stub",
                    "patch": {"world_flags": {"quest_smoke_started": True}},
                }
            ],
            "errors": [],
        }

    monkeypatch.setattr(wso.WorldStudioOrchestrator, "__init__", _stub_init)
    monkeypatch.setattr(wso.WorldStudioOrchestrator, "suggest_worldpack_patch", _stub_suggest)

    resp = await suggest_worldpack_update(
        "default",
        WorldAgentSuggestRequest(
            instruction="smoke test: add a flag",
            apply=False,
            rag_top_k=0,
            max_new_characters=0,
            max_new_player_templates=0,
            include_visual=False,
        ),
    )
    assert resp.success is True
    assert resp.applied is False
    assert resp.world_id == "default"
    assert resp.worldpack.world_flags.get("quest_smoke_started") is True


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_rag_world_id_filter(monkeypatch):
    from datetime import datetime

    from api.routers.rag import list_documents, search_documents
    from schemas.rag import RAGSearchRequest

    class StubDoc:
        def __init__(self, doc_id: str, content: str, metadata: dict):
            self.doc_id = doc_id
            self.content = content
            self.metadata = metadata
            self.created_at = datetime.now()

    class StubResult:
        def __init__(self, document: StubDoc, score: float):
            self.document = document
            self.score = score

    class StubRagEngine:
        def __init__(self):
            self.documents = {
                "chunk_a1": StubDoc(
                    "chunk_a1",
                    "alpha",
                    {
                        "world_id": "world_a",
                        "parent_doc_id": "doc_a",
                        "title": "Doc A",
                        "tags": ["a"],
                    },
                ),
                "chunk_a2": StubDoc(
                    "chunk_a2",
                    "alpha2",
                    {"world_id": "world_a", "parent_doc_id": "doc_a", "title": "Doc A"},
                ),
                "chunk_b1": StubDoc(
                    "chunk_b1",
                    "bravo",
                    {"world_id": "world_b", "parent_doc_id": "doc_b", "title": "Doc B"},
                ),
            }

        def search(
            self,
            query: str,
            *,
            top_k: int = 5,
            min_score: float = 0.1,
            world_id: str | None = None,
            enable_rerank: bool | None = None,
            rerank_top_k: int | None = None,
        ):
            _ = (query, min_score, enable_rerank, rerank_top_k)
            docs = list(self.documents.values())
            if world_id:
                docs = [d for d in docs if str(d.metadata.get("world_id", "default")) == str(world_id)]
            return [StubResult(d, 0.9) for d in docs[: max(0, int(top_k or 0))]]

    engine = StubRagEngine()
    monkeypatch.setattr("api.routers.rag.get_rag_engine", lambda: engine)

    # list_documents: world_id query param
    resp = await list_documents(world_id="world_a")
    assert resp["success"] is True
    assert resp["total"] == 1
    assert resp["documents"][0]["doc_id"] == "doc_a"
    assert resp["documents"][0]["metadata"]["world_id"] == "world_a"

    # search: filters.world_id
    resp = await search_documents(RAGSearchRequest(query="alpha", filters={"world_id": "world_a"}))
    assert resp.query == "alpha"
    assert all(r.metadata.get("world_id") == "world_a" for r in resp.results)
