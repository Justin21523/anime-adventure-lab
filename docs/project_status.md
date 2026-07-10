# Project status

Updated: 2026-07-10

SagaForge is in an architecture-transition beta. The supported product is a
single-creator, Story-first workbench; it is not accurate to describe every
legacy multimodal endpoint as production-ready.

## Supported core

- FastAPI starts in a CPU-light profile without importing local Transformers,
  Diffusers, VLM, or training runtimes.
- React uses the `/api/v2` Story workbench for worlds, sessions, durable turns,
  jobs, and lore uploads.
- Postgres is the source of truth for worlds, sessions, turns, documents,
  chunks, jobs, artifacts, and review proposals.
- Story turns require an idempotency key and serialize state changes per
  session. Redis dispatch failure does not lose the durable job; claimed jobs
  use execution IDs and leases so stale delivery attempts cannot overwrite a
  newer attempt.
- Lore uploads are stored in MinIO and indexed into 1024-dimensional pgvector
  chunks by a worker. Story turns retrieve only their world's chunks and persist
  source, excerpt, position, and score as citations.
- Private deployments support Argon2 admin login, signed HttpOnly cookies,
  CSRF protection, API keys, rate limits, request IDs, and Problem Details.
- JSON WorldPack and Story session import supports dry-run and idempotent apply.
- Jobs expose request, dispatch, execution, attempt, lease, duration, and error
  details. Celery beat reconciles deferred dispatches and expired leases.
- Story-generated WorldPack patches enter a human Review Queue and require an
  optimistic-lock approval before changing authoritative data.

## Compatibility and experimental surface

- v1 Story, World, RAG, and job endpoints remain for limited migration
  compatibility, but v2 is the target contract.
- VLM, training, model export, model management, direct T2I, ControlNet, LoRA,
  performance tuning, and legacy safety endpoints are disabled unless
  `ENABLE_EXPERIMENTAL_ROUTERS=1`.
- Legacy file-based session flows remain in the repository and should not be
  mixed with v2 sessions.
- T2I artifact persistence and review-proposal APIs have schema foundations but
  are not yet complete v2 user flows.

## Verified gates

- Foundation, persistence, object-store, auth/CSRF, v2 API, and smoke tests.
- Alembic offline PostgreSQL SQL generation.
- React type-check, focused ESLint, and production build.
- Production Compose configuration validation.
- Static portfolio asset validation and GitHub Pages deployment workflow.

## Next milestones

1. Add Postgres/MinIO service-container integration tests in CI.
2. Complete the generated-artifact user flow while keeping it outside the core
   Story demo until separately verified.
3. Add hosted Postgres/MinIO service-container integration tests.
4. Add server-side session revocation before any multi-user expansion.
5. Measure model-mode quality and latency without weakening deterministic gates.
