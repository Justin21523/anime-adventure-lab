#!/usr/bin/env python3
"""Build public portfolio facts only from machine-generated reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_GLOBS = (
    "api/**/*.py",
    "core/**/*.py",
    "schemas/**/*.py",
    "workers/**/*.py",
    "migrations/**/*.py",
    "tests/**/*.py",
    "frontend/react/src/**/*.tsx",
    "frontend/react/src/**/*.jsx",
    "frontend/react/src/**/*.ts",
    "frontend/react/src/**/*.js",
)


def load_json(path: Path | None) -> dict[str, Any] | None:
    return (
        json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else None
    )


def fingerprint() -> str:
    digest = hashlib.sha256()
    files = sorted(
        {
            path
            for pattern in SOURCE_GLOBS
            for path in ROOT.glob(pattern)
            if path.is_file()
        }
    )
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def junit_summary(path: Path | None) -> dict[str, int] | None:
    if not path or not path.exists():
        return None
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--e2e", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "portfolio-web/data/evidence.json"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "docs/verification-report.md"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current_fingerprint = fingerprint()
    if args.check:
        existing = load_json(args.output)
        if not existing or existing.get("schema_version") != 1:
            raise SystemExit("Evidence file missing or schema version is unsupported")
        if existing.get("source_fingerprint") != current_fingerprint:
            raise SystemExit("Evidence is stale: source fingerprint changed")
        if existing.get("verified") is not True:
            raise SystemExit("Evidence is incomplete and cannot be published")
        print("Portfolio evidence is current and verified")
        return 0

    coverage = load_json(args.coverage)
    e2e = load_json(args.e2e)
    benchmark = load_json(args.benchmark)
    tests = junit_summary(args.junit)
    migration = (
        subprocess.run(
            ["alembic", "heads"], cwd=ROOT, text=True, capture_output=True, check=True
        )
        .stdout.strip()
        .split()[0]
    )
    coverage_percent = (
        round(float(coverage["totals"]["percent_covered"]), 2) if coverage else None
    )
    verified = bool(
        coverage
        and tests
        and e2e
        and e2e.get("passed")
        and benchmark
        and tests["failures"] == 0
        and tests["errors"] == 0
    )
    evidence = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": current_fingerprint,
        "verified": verified,
        "tests": tests,
        "coverage": (
            {"percent": coverage_percent, "scope": "refactored v2 core"}
            if coverage
            else None
        ),
        "migration_head": migration,
        "e2e": e2e,
        "benchmark": benchmark,
        "artifacts": {
            "screenshots": len(list((ROOT / "portfolio-web/assets").glob("*.png"))),
            "demo_video": (ROOT / "portfolio-web/assets/demo-recording.mp4").exists(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Verification report\n\n"
        f"Generated: `{evidence['generated_at']}`<br>\nSource fingerprint: `{current_fingerprint}`\n\n"
        f"- Publishable evidence: **{'yes' if verified else 'no — required reports are missing'}**\n"
        f"- Tests: `{tests or 'not supplied'}`\n- Refactored v2 core coverage: `{coverage_percent if coverage_percent is not None else 'not supplied'}`\n"
        f"- Migration head: `{migration}`\n- Compose E2E: `{'passed' if e2e and e2e.get('passed') else 'not supplied or failed'}`\n"
        f"- Deterministic benchmark: `{'recorded' if benchmark else 'not supplied'}`\n\n"
        "This document is generated. Missing evidence remains explicitly unavailable; no placeholder metrics are substituted.\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
