# SagaForge — Anime Adventure Lab

> A Story-first AI authoring workbench that keeps narrative state, world-scoped knowledge, asynchronous jobs, and AI-proposed changes observable and reviewable.

[Public portfolio demo](https://justin21523.github.io/anime-adventure-lab/) · [Architecture](docs/architecture-v2.md) · [Demo runbook](docs/demo-runbook.md) · [Project status](docs/project_status.md)

![SagaForge Story workbench](portfolio-web/assets/screenshot-story.png)

## Why this project exists

Story generation is easy to prototype as a chat box, but difficult to operate as a dependable product. A useful authoring system must preserve continuity, retrieve only the current world's lore, survive worker failures, and prevent an AI response from silently overwriting authoritative data.

SagaForge demonstrates those boundaries through a private creator workbench:

- Story turns are transactional, serialized per session, and idempotent.
- Uploaded lore moves through MinIO, Celery, deterministic/model embeddings, and pgvector.
- Every retrieved chunk is persisted as a citation with its source and score.
- Jobs expose dispatch, attempt, lease, progress, request ID, result, and failure state.
- AI world changes enter a Review Queue and require optimistic-lock approval.
- The CPU-safe core runs without PyTorch; model-heavy capabilities are explicit experimental profiles.

## Core demo flow

1. Sign in with the private admin session.
2. Select the `moon-archive` world and continue Rin's story.
3. Inspect the indexed lore document and pgvector chunk count.
4. Ask how the silver archive box opens.
5. Follow the durable job from queued to completed.
6. Expand the RAG evidence and verify the answer against the uploaded lore.
7. Review the AI-proposed WorldPack patch, then approve or reject it.
8. Open Runtime & Services to show Postgres, pgvector, Redis, MinIO, Celery, migrations, and runtime profiles.

The public GitHub Pages site is static and anonymous. The complete mutable application is a separate authenticated Docker Compose deployment; no private API credentials are embedded in the public demo.

## Architecture

```mermaid
flowchart LR
    Browser[React 19 + TypeScript] -->|HttpOnly session + CSRF| API[FastAPI v2]
    API --> DB[(Postgres + pgvector)]
    API --> Store[(MinIO)]
    API --> Queue[Redis]
    Queue --> Worker[Celery core worker]
    Beat[Celery beat reconciler] --> Queue
    Worker --> DB
    Worker --> Store
    Worker -. optional .-> LLM[llama.cpp / model runtime]
```

### Story turn lifecycle

```mermaid
sequenceDiagram
    participant UI as React
    participant API as FastAPI
    participant DB as Postgres
    participant Q as Redis/Celery
    participant W as Worker
    UI->>API: POST turn + Idempotency-Key
    API->>DB: create turn + durable job transaction
    API->>Q: dispatch after commit
    W->>DB: claim job + execution lease
    W->>DB: world-scoped pgvector retrieval
    W->>W: deterministic or LLM generation
    W->>DB: atomic turn, citation, state, proposal, job completion
    UI->>API: poll observable job
    API-->>UI: completed + evidence
```

## Technology and decisions

| Area | Technology | Why it is used |
| --- | --- | --- |
| UI | React 19, TypeScript, Vite, TanStack Query | Typed server-state UI with explicit loading, error, and recovery states |
| API | FastAPI, Pydantic, OpenAPI | Contract-first v2 endpoints and generated frontend types |
| Persistence | PostgreSQL, SQLAlchemy, Alembic | Transactional source of truth and additive schema migrations |
| Retrieval | pgvector, deterministic/model embeddings | World-isolated semantic retrieval with reproducible demo mode |
| Objects | MinIO | Uploaded lore and recoverable artifacts stay outside relational rows |
| Jobs | Redis, Celery, Celery beat | Durable asynchronous execution, leases, attempts, and reconciliation |
| Security | Argon2, signed HttpOnly cookies, CSRF, API keys | Appropriate single-admin private-workbench boundary |
| Delivery | Docker Compose, Nginx, GitHub Actions, GitHub Pages | Reproducible private stack and isolated public portfolio |

Important decisions are recorded in [docs/adr](docs/adr). The supported core intentionally does not claim that every retained VLM, T2I, LoRA, or training prototype is production-ready.

## Quick start

### Local development

```bash
conda create -n ai_env python=3.10 -y
conda activate ai_env
pip install -r requirements.txt -r requirements-test.txt

uvicorn api.main:app --reload
cd frontend/react && npm install && npm run dev
```

The React dev server runs at `http://localhost:3000`; FastAPI health is available at `http://localhost:8000/healthz`.

### Private Docker demo

Create `.env` from `.env.example` and set unique values for:

```dotenv
POSTGRES_PASSWORD=...
DATABASE_URL=postgresql://saga:...@postgres:5432/sagaforge
MINIO_USER=...
MINIO_PASSWORD=...
API_SECRET_KEY=...
API_SESSION_SECRET=...
API_ADMIN_PASSWORD_HASH=...
```

Generate the password hash without writing plaintext to the repository:

```bash
python -c "from argon2 import PasswordHasher; print(PasswordHasher().hash(input('Password: ')))"
```

Start the reproducible deterministic profile:

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml up --build -d
docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml exec api \
  python scripts/seed_demo.py --apply --reset
```

Production Compose runs migrations and bucket initialization before API startup. Postgres, Redis, and MinIO are internal-only; TLS must terminate at a trusted reverse proxy.

## Runtime profiles

```dotenv
# Reliable interview/demo profile
STORY_RUNTIME_MODE=deterministic
RAG_RUNTIME_MODE=deterministic
WORKER_PROFILE=core

# Optional real local AI profile
STORY_RUNTIME_MODE=llm
RAG_RUNTIME_MODE=model
LLAMA_SERVER_URL=http://host.docker.internal:8080
```

The same application contract is used in both profiles. Deterministic mode exists so an interview does not depend on GPU availability or a remote model service.

## Validation

```bash
pytest -q
pytest -q -m smoke
ruff check api core schemas workers tests
black --check api core schemas workers tests
alembic upgrade head --sql

cd frontend/react
npm run type-check
npm test
npm run lint
npm run build
npm audit --audit-level=high
```

The maintained gate covers authentication, CSRF, Problem Details, transactional worlds and sessions, idempotent turns, pgvector evidence, review proposals, job leases/reconciliation, object storage, API contracts, and React states. Retired v1 contracts remain inspectable with `pytest -m legacy` and `npm run lint:legacy`.

The private workbench also includes an **Engineering Evidence** view. It reads
the persistent Job API and chronological Job Event API, so API, worker,
scheduler, retry, and recovery transitions can be inspected without relying on
transient broker logs.

## Demo media

The committed screenshots and video are generated from the real deterministic workbench, not hand-drawn mockups:

```bash
cd frontend/react
DEMO_ADMIN_PASSWORD='temporary-demo-password' npm run capture:demo
```

The capture requires local Chromium and ffmpeg. It produces desktop/mobile screenshots and a silent H.264 walkthrough under `portfolio-web/assets/`.

Measure the currently running deterministic stack without publishing invented
numbers:

```bash
DEMO_API_KEY='temporary-api-key' python scripts/benchmark_demo.py --iterations 3
```

Run the complete reproducible evidence flow and rebuild the public case study:

```bash
make demo-up
make demo-e2e
make demo-benchmark
make portfolio-evidence
python scripts/generate_portfolio_evidence.py --check
make demo-down
```

The generated [verification report](docs/verification-report.md) combines
JUnit, scoped coverage, migration head, deterministic benchmark, Compose
recovery E2E, and a source fingerprint. Missing inputs remain explicitly
unavailable; the portfolio page does not use fallback metrics.

## Security boundary

- Browser sessions are HMAC signed, HttpOnly, SameSite cookies with CSRF protection.
- Passwords are verified with Argon2; login has a separate lower rate limit.
- API keys are accepted only in headers and hashed before rate-limit identification.
- OpenAPI is private unless `PUBLIC_API_DOCS=1` is explicitly set.
- Generic Agent execution and filesystem tools are disabled by default.
- Forwarded client IP headers are ignored unless a trusted proxy is explicitly configured.

See [docs/security.md](docs/security.md) for limitations and rotation guidance.

## Supported and experimental

### Supported v2 core

- World and Story Session persistence
- Durable Story turns and observable jobs
- Transactional, sanitized Job lifecycle event timeline
- MinIO lore upload and pgvector indexing/retrieval
- RAG citations and runtime trace
- Human-reviewed WorldPack proposals
- Admin session/API-key security
- Legacy JSON dry-run/idempotent import
- CPU-safe Docker Compose deployment

### Experimental compatibility surface

- VLM caption/VQA
- Direct T2I and ControlNet
- LoRA/fine-tuning
- Model export and management
- Legacy file-backed Story and RAG endpoints

Experimental routers require `ENABLE_EXPERIMENTAL_ROUTERS=1` and model-capable workers require `WORKER_PROFILE=experimental` with a separately provisioned AI image.

## Known limitations

- The private workbench is deliberately single-admin, not multi-tenant identity software.
- Signed sessions have no server-side revocation list; secret rotation forces logout.
- The public site is a deterministic portfolio, not a shared writable playground.
- Generated image artifacts remain outside the supported v2 Demo flow.
- Real screenshots and benchmark numbers must be regenerated when the UI or deployment profile materially changes.

## Repository map

```text
api/                 FastAPI composition, middleware, v2 endpoints
core/application/    Story, RAG, job, review, and status use cases
core/persistence/    SQLAlchemy models and Unit of Work
workers/tasks/       Durable Story, indexing, and reconciliation tasks
frontend/react/      Story-first React workbench
migrations/          Alembic PostgreSQL/pgvector migrations
docs/                Architecture, ADRs, security, deployment, demo
portfolio-web/       Public static case study and generated media
```

## Roadmap

- Add generated-image artifacts as a separately reviewed v2 capability.
- Add server-side session revocation if the product becomes multi-user.
- Extend service-container integration tests to hosted CI.
- Add measured model-mode quality and latency comparisons without weakening deterministic gates.

License: Apache-2.0 (project metadata pending final confirmation).
