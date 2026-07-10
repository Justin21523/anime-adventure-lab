#!/usr/bin/env python3
"""Run the deterministic, API-level portfolio scenario and emit auditable JSON."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx


def wait_job(client: httpx.Client, base: str, job_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"{base}/jobs/{job_id}").raise_for_status().json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.25)
    raise TimeoutError(f"Job did not finish: {job_id}")


def compose(project: str, *args: str) -> None:
    subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            project,
            "-f",
            "docker-compose.prod.yml",
            "-f",
            "docker-compose.demo.yml",
            *args,
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v2")
    parser.add_argument("--compose-project")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    api_key = os.getenv("DEMO_API_KEY")
    if not api_key:
        raise SystemExit("DEMO_API_KEY is required")

    marker = uuid.uuid4().hex[:10]
    world_id = f"evidence-{marker}"
    base = args.base_url.rstrip("/")
    checks: dict[str, bool] = {}
    worker_stopped = False
    started = time.perf_counter()
    with httpx.Client(headers={"X-API-Key": api_key}, timeout=20) as client:
        try:
            world = (
                client.post(
                    f"{base}/worlds",
                    json={
                        "world_id": world_id,
                        "name": "Evidence Archive",
                        "pack": {"rules": {"human_review_required": True}},
                    },
                )
                .raise_for_status()
                .json()
            )
            checks["world_created"] = world["version"] == 1
            lore = "The silver archive opens only after the third moon bell."
            upload = (
                client.post(
                    f"{base}/worlds/{world_id}/documents",
                    files={"file": ("verified-lore.md", lore, "text/markdown")},
                )
                .raise_for_status()
                .json()
            )
            index_job = wait_job(client, base, upload["job"]["job_id"], args.timeout)
            checks["document_indexed"] = index_job["status"] == "completed"
            session = (
                client.post(
                    f"{base}/story-sessions",
                    json={"world_id": world_id, "player_name": "Evidence Runner"},
                )
                .raise_for_status()
                .json()
            )

            if args.compose_project:
                compose(args.compose_project, "stop", "worker")
                worker_stopped = True
            turn = (
                client.post(
                    f"{base}/story-sessions/{session['session_id']}/turns",
                    headers={"Idempotency-Key": f"e2e-{marker}"},
                    json={
                        "player_input": "How does the silver archive open?",
                        "rag_mode": "on",
                    },
                )
                .raise_for_status()
                .json()
            )
            queued = (
                client.get(f"{base}/jobs/{turn['job_id']}").raise_for_status().json()
            )
            if args.compose_project:
                checks["durable_while_worker_stopped"] = queued["status"] == "queued"
            if worker_stopped:
                compose(args.compose_project, "start", "worker")
                worker_stopped = False
            completed = wait_job(client, base, turn["job_id"], args.timeout)
            checks[
                (
                    "story_completed_after_worker_restart"
                    if args.compose_project
                    else "story_completed"
                )
            ] = (completed["status"] == "completed")
            turns = (
                client.get(f"{base}/story-sessions/{session['session_id']}/turns")
                .raise_for_status()
                .json()
            )
            citations = turns[-1]["citations"]
            checks["rag_citation_persisted"] = bool(
                citations and citations[0]["filename"] == "verified-lore.md"
            )

            proposals = (
                client.get(f"{base}/review-proposals?world_id={world_id}")
                .raise_for_status()
                .json()
            )
            pending = next(item for item in proposals if item["status"] == "pending")
            approved = (
                client.post(
                    f"{base}/review-proposals/{pending['proposal_id']}/approve",
                    headers={"If-Match": '"1"'},
                )
                .raise_for_status()
                .json()
            )
            checks["human_review_versioned"] = approved["world"]["version"] == 2
            events = (
                client.get(f"{base}/jobs/{turn['job_id']}/events")
                .raise_for_status()
                .json()
            )
            event_types = [event["event_type"] for event in events]
            checks["event_chain_persisted"] = all(
                name in event_types for name in ("queued", "claimed", "completed")
            )
            report = {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "profile": (
                    "deterministic-compose"
                    if args.compose_project
                    else "deterministic-api"
                ),
                "passed": all(checks.values()),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "checks": checks,
                "evidence": {
                    "job_id": turn["job_id"],
                    "attempt_count": completed["attempt_count"],
                    "event_types": event_types,
                    "citation": (
                        {
                            "filename": citations[0]["filename"],
                            "chunk_id": citations[0]["chunk_id"],
                            "score": citations[0]["score"],
                        }
                        if citations
                        else None
                    ),
                    "world_version_after_review": approved["world"]["version"],
                },
            }
        finally:
            if worker_stopped and args.compose_project:
                compose(args.compose_project, "start", "worker")
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
