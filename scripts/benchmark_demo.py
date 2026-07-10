#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import uuid

import httpx


def wait_job(client: httpx.Client, base: str, job_id: str) -> tuple[dict, float]:
    started = time.perf_counter()
    while time.perf_counter() - started < 60:
        job = client.get(f"{base}/jobs/{job_id}").raise_for_status().json()
        if job["status"] in {"completed", "failed"}:
            return job, time.perf_counter() - started
        time.sleep(0.2)
    raise TimeoutError(job_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the deterministic portfolio flow"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v2")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    key = os.getenv("DEMO_API_KEY")
    if not key:
        raise SystemExit("DEMO_API_KEY is required")
    base = args.base_url.rstrip("/")
    headers = {"X-API-Key": key}
    turn_times: list[float] = []
    index_times: list[float] = []
    with httpx.Client(headers=headers, timeout=20) as client:
        world_response = client.get(f"{base}/worlds/moon-archive")
        if world_response.status_code == 404:
            client.post(
                f"{base}/worlds",
                json={
                    "world_id": "moon-archive",
                    "name": "Benchmark Archive",
                    "pack": {"profile": "deterministic-benchmark"},
                },
            ).raise_for_status()
        else:
            world_response.raise_for_status()
        session = (
            client.post(
                f"{base}/story-sessions",
                json={"world_id": "moon-archive", "player_name": "Benchmark"},
            )
            .raise_for_status()
            .json()
        )
        for index in range(max(1, args.iterations)):
            marker = uuid.uuid4().hex
            uploaded = (
                client.post(
                    f"{base}/worlds/moon-archive/documents",
                    files={
                        "file": (
                            f"benchmark-{marker}.md",
                            f"benchmark lore {marker}",
                            "text/markdown",
                        )
                    },
                )
                .raise_for_status()
                .json()
            )
            index_job, index_time = wait_job(client, base, uploaded["job"]["job_id"])
            if index_job["status"] != "completed":
                raise RuntimeError(index_job)
            index_times.append(index_time)
            turn = (
                client.post(
                    f"{base}/story-sessions/{session['session_id']}/turns",
                    headers={"Idempotency-Key": f"benchmark-{marker}"},
                    json={
                        "player_input": f"benchmark turn {index}",
                        "rag_mode": "auto",
                    },
                )
                .raise_for_status()
                .json()
            )
            turn_job, turn_time = wait_job(client, base, turn["job_id"])
            if turn_job["status"] != "completed":
                raise RuntimeError(turn_job)
            turn_times.append(turn_time)
    report = {
        "iterations": len(turn_times),
        "story_turn_median_ms": round(statistics.median(turn_times) * 1000, 1),
        "document_index_median_ms": round(statistics.median(index_times) * 1000, 1),
        "profile": "deterministic",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
