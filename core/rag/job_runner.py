from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from core.rag.document_processor import DocumentProcessor
from core.rag.engine import get_rag_engine
from core.shared_cache import get_shared_cache
from core.train.job_manager import TrainJobManager


def _job_report(
    job_manager: TrainJobManager,
    job_id: str,
    *,
    stage: str,
    progress: float,
    message: str,
    meta: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update job stage/progress (best-effort). Returns False if job is cancelled/terminal."""
    try:
        existing = job_manager.get_job(job_id, auto_progress=False) or {}
        status = str(existing.get("status") or "").lower()
        if status in {"completed", "failed", "cancelled"}:
            return False
        if bool(existing.get("cancel_requested")):
            return False

        p = float(progress or 0.0)
        if not (p == p):  # NaN
            p = 0.0
        p = max(0.0, min(100.0, p))

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        events = existing.get("stage_events")
        if not isinstance(events, list):
            events = []
        event: Dict[str, Any] = {"ts": now, "stage": str(stage), "progress": round(p, 2)}
        if message:
            event["message"] = str(message)
        if meta:
            event["meta"] = meta
        events.append(event)
        if len(events) > 50:
            events = events[-50:]

        updates: Dict[str, Any] = {
            "status": "running",
            "progress": round(p, 2),
            "stage": str(stage),
            "stage_message": str(message),
            "stage_updated_at": now,
            "stage_events": events,
        }
        if not existing.get("started_at"):
            updates["started_at"] = now

        job_manager.update_job(job_id, **updates)
        return True
    except Exception:
        return True


def _safe_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _patch_story_session_rag_rebuild_job(
    *,
    session_id: str,
    turn: Optional[int],
    job_snapshot: Dict[str, Any],
) -> None:
    cache = get_shared_cache()
    base_dir = Path(cache.get_output_path("games")) / "story_sessions"
    session_file = base_dir / f"{session_id}.json"
    if not session_file.exists():
        session_file = Path("outputs") / "story_sessions" / f"{session_id}.json"
    if not session_file.exists():
        return

    for _ in range(3):
        try:
            before_mtime = session_file.stat().st_mtime
            data = json.loads(session_file.read_text(encoding="utf-8"))
            history = data.get("history")
            if isinstance(history, list) and history:
                target_entry = None
                if turn is not None:
                    for entry in reversed(history):
                        if not isinstance(entry, dict):
                            continue
                        if int(entry.get("turn", -1) or -1) == int(turn):
                            target_entry = entry
                            break
                if target_entry is None:
                    target_entry = history[-1] if isinstance(history[-1], dict) else None

                if isinstance(target_entry, dict):
                    artifacts = target_entry.get("artifacts")
                    if not isinstance(artifacts, dict):
                        artifacts = {}
                    rag_bucket = artifacts.get("rag")
                    if not isinstance(rag_bucket, dict):
                        rag_bucket = {}
                    rag_bucket["rebuild_job"] = job_snapshot
                    artifacts["rag"] = rag_bucket
                    target_entry["artifacts"] = artifacts

            if session_file.exists() and session_file.stat().st_mtime != before_mtime:
                continue
            session_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        except Exception:
            continue


def run_rag_rebuild_job(
    job_id: str,
    payload: Dict[str, Any],
    *,
    job_manager: Optional[TrainJobManager] = None,
) -> Dict[str, Any]:
    job_id = str(job_id or "").strip()
    if not job_id:
        raise ValueError("job_id is required")

    jm = job_manager or TrainJobManager()
    existing = jm.get_job(job_id, auto_progress=False) or {}
    if str(existing.get("status") or "").lower() == "cancelled":
        return existing

    session_id = str(payload.get("session_id") or "").strip() or None
    turn = payload.get("turn")
    turn_int: Optional[int]
    try:
        turn_int = int(turn) if turn is not None else None
    except Exception:
        turn_int = None

    started = time.time()
    if not _job_report(jm, job_id, stage="rebuild", progress=1.0, message="開始重建索引"):
        return jm.get_job(job_id, auto_progress=False) or {"job_id": job_id}

    rag_engine = get_rag_engine()
    try:
        _job_report(jm, job_id, stage="rebuild", progress=10.0, message="重建（FAISS/BM25）")
        success = rag_engine.rebuild_index()
        if success:
            _job_report(jm, job_id, stage="save", progress=90.0, message="保存索引")
            try:
                rag_engine.save_index()
            except Exception:
                pass

        duration = time.time() - started
        stats = rag_engine.get_stats() if success else {}

        latest = jm.get_job(job_id, auto_progress=False) or {}
        latest_status = str(latest.get("status") or "").lower()
        if latest_status == "cancelled" or bool(latest.get("cancel_requested")):
            jm.update_job(
                job_id,
                status="cancelled",
                progress=float(latest.get("progress") or 0.0),
                result={
                    "success": bool(success),
                    "time_taken_seconds": duration,
                    "documents_processed": stats.get("total_documents", 0) if success else 0,
                    "stats": stats,
                },
                duration_seconds=duration,
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        elif success:
            _job_report(jm, job_id, stage="done", progress=99.0, message="完成")
            jm.update_job(
                job_id,
                status="completed",
                progress=100.0,
                result={
                    "success": True,
                    "time_taken_seconds": duration,
                    "documents_processed": stats.get("total_documents", 0),
                    "stats": stats,
                },
                duration_seconds=duration,
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        else:
            _job_report(jm, job_id, stage="failed", progress=10.0, message="重建失敗")
            jm.update_job(
                job_id,
                status="failed",
                progress=0.0,
                result={
                    "success": False,
                    "time_taken_seconds": duration,
                    "documents_processed": 0,
                },
                error="rebuild_index returned False",
                duration_seconds=duration,
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        job = jm.get_job(job_id, auto_progress=False) or {"job_id": job_id}
        if session_id:
            try:
                stage_events = job.get("stage_events")
                if not isinstance(stage_events, list):
                    stage_events = []
                snapshot = {
                    "job_id": job_id,
                    "job_type": "rag_rebuild",
                    "status": job.get("status"),
                    "stage": job.get("stage"),
                    "stage_message": job.get("stage_message"),
                    "progress": job.get("progress"),
                    "started_at": job.get("started_at") or job.get("created_at"),
                    "duration_seconds": job.get("duration_seconds"),
                    "stage_events": [
                        {
                            "ts": e.get("ts"),
                            "stage": e.get("stage"),
                            "progress": e.get("progress"),
                            "message": e.get("message"),
                            "meta": e.get("meta"),
                        }
                        for e in (stage_events or [])[-30:]
                        if isinstance(e, dict)
                    ],
                }
                _patch_story_session_rag_rebuild_job(
                    session_id=session_id, turn=turn_int, job_snapshot=snapshot
                )
            except Exception:
                pass
        return job
    except Exception as exc:  # noqa: BLE001
        duration = time.time() - started
        _job_report(jm, job_id, stage="failed", progress=0.0, message="重建失敗", meta={"error": str(exc)})
        jm.update_job(
            job_id,
            status="failed",
            progress=0.0,
            error=str(exc),
            result={"success": False, "time_taken_seconds": duration},
            duration_seconds=duration,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        job = jm.get_job(job_id, auto_progress=False) or {"job_id": job_id}
        if session_id:
            try:
                stage_events = job.get("stage_events")
                if not isinstance(stage_events, list):
                    stage_events = []
                snapshot = {
                    "job_id": job_id,
                    "job_type": "rag_rebuild",
                    "status": job.get("status"),
                    "stage": job.get("stage"),
                    "stage_message": job.get("stage_message"),
                    "progress": job.get("progress"),
                    "started_at": job.get("started_at") or job.get("created_at"),
                    "duration_seconds": job.get("duration_seconds"),
                    "stage_events": [
                        {
                            "ts": e.get("ts"),
                            "stage": e.get("stage"),
                            "progress": e.get("progress"),
                            "message": e.get("message"),
                            "meta": e.get("meta"),
                        }
                        for e in (stage_events or [])[-30:]
                        if isinstance(e, dict)
                    ],
                }
                _patch_story_session_rag_rebuild_job(
                    session_id=session_id, turn=turn_int, job_snapshot=snapshot
                )
            except Exception:
                pass
        raise


def run_rag_upload_job(
    job_id: str,
    payload: Dict[str, Any],
    *,
    job_manager: Optional[TrainJobManager] = None,
) -> Dict[str, Any]:
    job_id = str(job_id or "").strip()
    if not job_id:
        raise ValueError("job_id is required")

    jm = job_manager or TrainJobManager()
    existing = jm.get_job(job_id, auto_progress=False) or {}
    if str(existing.get("status") or "").lower() == "cancelled":
        return existing

    world_id = str(payload.get("world_id") or "default").strip() or "default"
    tags_str = str(payload.get("tags") or "").strip()
    tags = [t.strip() for t in tags_str.split(",") if t and t.strip()]

    file_specs = payload.get("files")
    if isinstance(file_specs, list) and file_specs:
        specs = [s for s in file_specs if isinstance(s, dict)]
    else:
        file_path = str(payload.get("file_path") or "").strip()
        if not file_path:
            jm.update_job(job_id, status="failed", progress=0.0, error="file_path/files is required")
            raise ValueError("file_path/files is required")
        specs = [
            {
                "file_path": file_path,
                "original_filename": payload.get("original_filename"),
                "content_type": payload.get("content_type"),
            }
        ]

    started = time.time()
    cleanup_paths: list[str] = []
    temp_dirs: list[tempfile.TemporaryDirectory] = []
    try:
        if not _job_report(jm, job_id, stage="load", progress=1.0, message="開始處理"):
            return jm.get_job(job_id, auto_progress=False) or {"job_id": job_id}

        allowed_suffixes = {
            ".txt",
            ".md",
            ".markdown",
            ".pdf",
            ".docx",
            ".doc",
            ".csv",
            ".tsv",
            ".json",
            ".xml",
            ".html",
            ".htm",
            ".yml",
            ".yaml",
            ".toml",
            ".ini",
            ".cfg",
            ".log",
            ".rst",
            ".tex",
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".css",
            ".scss",
        }

        ingest_items: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        def _is_zip(path: str, name: str, ctype: Optional[str]) -> bool:
            if str(name or "").lower().endswith(".zip"):
                return True
            if str(path or "").lower().endswith(".zip"):
                return True
            raw = str(ctype or "").lower()
            return raw in {"application/zip", "application/x-zip-compressed"}

        def _accept_file(name: str) -> bool:
            suffix = Path(str(name or "")).suffix.lower()
            if not suffix:
                return True
            if suffix == ".zip":
                return True
            return suffix in allowed_suffixes

        def _queue_file(*, path: str, original_name: str, ctype: Optional[str]) -> None:
            if not path:
                return
            if not os.path.exists(path):
                skipped.append({"original_filename": original_name, "reason": "missing_file"})
                return
            if not _accept_file(original_name or path):
                skipped.append({"original_filename": original_name, "reason": "unsupported_type"})
                return
            ingest_items.append({"file_path": path, "original_filename": original_name, "content_type": ctype})

        max_zip_files = _safe_int_env("RAG_ZIP_MAX_FILES", 200)
        max_zip_total_bytes = _safe_int_env("RAG_ZIP_MAX_TOTAL_BYTES", 200 * 1024 * 1024)  # 200MB
        max_zip_file_bytes = _safe_int_env("RAG_ZIP_MAX_FILE_BYTES", 50 * 1024 * 1024)  # 50MB

        if not _job_report(jm, job_id, stage="load", progress=4.0, message="準備批次清單"):
            return jm.get_job(job_id, auto_progress=False) or {"job_id": job_id}

        for spec in specs:
            src_path = str((spec or {}).get("file_path") or "").strip()
            if not src_path:
                continue
            cleanup_paths.append(src_path)
            original_name = str((spec or {}).get("original_filename") or "").strip() or os.path.basename(src_path)
            ctype = (spec or {}).get("content_type")

            if _is_zip(src_path, original_name, ctype):
                if not _job_report(jm, job_id, stage="extract", progress=6.0, message=f"解壓縮：{original_name}"):
                    return jm.get_job(job_id, auto_progress=False) or {"job_id": job_id}
                try:
                    tmp_dir = tempfile.TemporaryDirectory(prefix="rag_zip_")
                    temp_dirs.append(tmp_dir)
                    extract_root = Path(tmp_dir.name)

                    with zipfile.ZipFile(src_path) as zf:
                        total_bytes = 0
                        extracted_count = 0
                        for info in zf.infolist():
                            if extracted_count >= max_zip_files:
                                skipped.append({"original_filename": original_name, "reason": "zip_too_many_files"})
                                break
                            if getattr(info, "is_dir", lambda: False)():
                                continue

                            member_name = str(info.filename or "").replace("\\", "/").lstrip("/").strip()
                            if not member_name:
                                continue
                            member_path = Path(member_name)
                            if any(part == ".." for part in member_path.parts):
                                skipped.append({"original_filename": member_name, "reason": "zip_path_traversal"})
                                continue

                            try:
                                mode = (int(info.external_attr or 0) >> 16) & 0o170000
                                if mode == stat.S_IFLNK:
                                    skipped.append({"original_filename": member_name, "reason": "zip_symlink"})
                                    continue
                            except Exception:
                                pass

                            if int(getattr(info, "file_size", 0) or 0) > max_zip_file_bytes:
                                skipped.append({"original_filename": member_name, "reason": "zip_file_too_large"})
                                continue

                            total_bytes += int(getattr(info, "file_size", 0) or 0)
                            if max_zip_total_bytes and total_bytes > max_zip_total_bytes:
                                skipped.append({"original_filename": original_name, "reason": "zip_total_too_large"})
                                break

                            dest = (extract_root / member_path).resolve()
                            if extract_root.resolve() not in dest.parents and dest != extract_root.resolve():
                                skipped.append({"original_filename": member_name, "reason": "zip_outside_target"})
                                continue

                            dest.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                with zf.open(info) as src, open(dest, "wb") as dst:
                                    shutil.copyfileobj(src, dst)
                                extracted_count += 1
                                _queue_file(path=str(dest), original_name=member_name, ctype=None)
                            except Exception:
                                skipped.append({"original_filename": member_name, "reason": "zip_extract_failed"})
                                continue
                except Exception as exc:  # noqa: BLE001
                    skipped.append({"original_filename": original_name, "reason": f"zip_error:{exc}"})
                finally:
                    try:
                        os.unlink(src_path)
                    except Exception:
                        pass
            else:
                _queue_file(path=src_path, original_name=original_name, ctype=ctype)

        if not ingest_items:
            jm.update_job(
                job_id,
                status="failed",
                progress=0.0,
                stage="failed",
                stage_message="沒有可匯入的文件",
                error="no ingestable files",
                result={"skipped": skipped},
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            return jm.get_job(job_id, auto_progress=False) or {"job_id": job_id}

        if not _job_report(jm, job_id, stage="load", progress=8.0, message=f"解析文件（{len(ingest_items)}）"):
            return jm.get_job(job_id, auto_progress=False) or {"job_id": job_id}

        processor = DocumentProcessor()
        rag_engine = get_rag_engine()

        added_docs: list[dict[str, Any]] = []
        failed_docs: list[dict[str, Any]] = []

        total = len(ingest_items)
        for idx, item in enumerate(ingest_items):
            current = jm.get_job(job_id, auto_progress=False) or {}
            if str(current.get("status") or "").lower() == "cancelled" or bool(current.get("cancel_requested")):
                return current

            p = 10.0 + (idx / max(1, total)) * 75.0
            display_name = str(item.get("original_filename") or "").strip() or os.path.basename(
                str(item.get("file_path") or "")
            )
            if not _job_report(jm, job_id, stage="ingest", progress=p, message=f"({idx + 1}/{total}) {display_name}"):
                return jm.get_job(job_id, auto_progress=False) or {"job_id": job_id}

            file_path = str(item.get("file_path") or "").strip()
            try:
                processed_doc = processor.process_file(
                    file_path=file_path,
                    metadata={
                        "world_id": world_id,
                        "original_filename": str(item.get("original_filename") or "").strip() or display_name,
                        "content_type": item.get("content_type"),
                        "tags": tags,
                    },
                )

                if not _job_report(jm, job_id, stage="index", progress=p + 2.0, message=f"寫入知識庫：{display_name}"):
                    return jm.get_job(job_id, auto_progress=False) or {"job_id": job_id}

                added = rag_engine.add_document(
                    doc_id=processed_doc.doc_id,
                    content=processed_doc.content,
                    metadata=processed_doc.metadata,
                )
                if added:
                    added_docs.append({"doc_id": processed_doc.doc_id, "original_filename": display_name, "world_id": world_id})
                else:
                    failed_docs.append({"doc_id": processed_doc.doc_id, "original_filename": display_name, "world_id": world_id})
            except Exception as exc:  # noqa: BLE001
                failed_docs.append({"original_filename": display_name, "error": str(exc)})

            _job_report(jm, job_id, stage="save", progress=min(96.0, p + 5.0), message="保存索引")
            try:
                rag_engine.save_index()
            except Exception:
                pass

        current = jm.get_job(job_id, auto_progress=False) or {}
        if str(current.get("status") or "").lower() == "cancelled":
            return current

        duration = time.time() - started
        ok = len(added_docs) > 0
        status = "completed" if ok else "failed"
        message = f"完成（成功 {len(added_docs)} / 失敗 {len(failed_docs)} / 略過 {len(skipped)}）"

        jm.update_job(
            job_id,
            status=status,
            progress=100.0 if ok else float(current.get("progress") or 0.0),
            stage="done" if ok else "failed",
            stage_message=message,
            duration_seconds=duration,
            result={
                "success": ok,
                "world_id": world_id,
                "documents_added": len(added_docs),
                "documents_failed": len(failed_docs),
                "documents_skipped": len(skipped),
                "added": added_docs[:50],
                "failed": failed_docs[:50],
                "skipped": skipped[:50],
            },
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return jm.get_job(job_id, auto_progress=False) or {"job_id": job_id}
    except Exception as exc:  # noqa: BLE001
        jm.update_job(
            job_id,
            status="failed",
            progress=0.0,
            stage="failed",
            stage_message="失敗",
            error=str(exc),
            result={"success": False},
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        raise
    finally:
        try:
            for p in cleanup_paths:
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    continue
        except Exception:
            pass
        try:
            for td in temp_dirs:
                try:
                    td.cleanup()
                except Exception:
                    continue
        except Exception:
            pass

