import io
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import pytest
from starlette.datastructures import UploadFile


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_rag_upload_batch_job_zip_sync_fallback(tmp_path, monkeypatch):
    from core.rag.document_processor import ProcessedDocument
    from core.train.job_manager import TrainJobManager

    import api.routers.rag as rag_router
    import core.rag.job_runner as rag_runner

    isolated = TrainJobManager(cache_root=str(tmp_path))

    # Ensure router + worker share the same job store.
    monkeypatch.setattr(rag_router, "TrainJobManager", lambda *args, **kwargs: isolated, raising=False)

    class FakeRagEngine:
        def __init__(self):
            self.added = []

        def add_document(self, doc_id: str, content: str, metadata: dict):
            self.added.append({"doc_id": doc_id, "content": content, "metadata": metadata})
            return True

        def save_index(self):
            return None

    engine = FakeRagEngine()
    monkeypatch.setattr(rag_runner, "get_rag_engine", lambda: engine, raising=False)

    class FakeDocumentProcessor:
        def process_file(self, file_path: str, metadata: dict | None = None, doc_id=None):
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            return ProcessedDocument(
                doc_id=f"doc_{Path(file_path).stem}",
                content=text,
                metadata=metadata or {},
                file_type="text",
                file_size=len(text.encode("utf-8")),
                processed_at=datetime.utcnow(),
            )

    monkeypatch.setattr(rag_runner, "DocumentProcessor", FakeDocumentProcessor, raising=False)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.txt", "hello world")
        zf.writestr("b.md", "# Title\ncontent")
    zip_bytes = buf.getvalue()

    spooled = tempfile.SpooledTemporaryFile(max_size=1024 * 1024)
    spooled.write(zip_bytes)
    spooled.seek(0)

    upload = UploadFile(filename="docs.zip", file=spooled)
    resp = await rag_router.upload_documents_batch_job(files=[upload], world_id="w1", tags="lore,world")
    job_id = str(resp.get("job_id") or "").strip()
    assert job_id

    job = isolated.get_job(job_id, auto_progress=False) or {}
    assert str(job.get("status") or "").lower() == "completed"
    assert isinstance(job.get("result"), dict)
    assert int(job["result"].get("documents_added") or 0) == 2
    assert engine.added and len(engine.added) == 2
