# ADR 0005: Generate portfolio claims from verification artifacts

Status: accepted — 2026-07-10

## Context

Hardcoded latency, test, and retrieval numbers become stale and can misrepresent the implementation. A reviewer needs reproducible evidence and an explicit unavailable state when verification has not run.

## Decision

The public case study reads `portfolio-web/data/evidence.json`. The generator accepts coverage JSON, JUnit XML, deterministic benchmark JSON, and Compose E2E JSON, records the migration head and a source fingerprint, and marks evidence verified only when all required reports pass. The web page has no numeric fallback.

## Consequences

- Source changes invalidate `--check` until reports are regenerated.
- Missing reports remain visible as unavailable and cannot silently become portfolio claims.
- Deterministic measurements describe the reproducible profile only; they are not presented as real-model inference performance.
