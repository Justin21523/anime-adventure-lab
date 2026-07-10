# Testing strategy

The default `pytest` gate covers the supported Story-first architecture:

- security boundaries, browser sessions, CSRF, and Problem Details;
- Story DI and CPU-light import behavior;
- transactional worlds, sessions, turns, leased jobs, documents, chunks, and
  review proposals;
- world-scoped retrieval evidence, job reconciliation, and superseded worker
  execution protection;
- object-store path and credential behavior;
- maintained mock-mode Story/RAG/job smoke flows.

Run it with:

```bash
pytest -q
cd frontend/react && npm test && npm run type-check && npm run build
```

Tests for retired v1 URLs, generic Agent/file tools, direct model-management
routers, old file-backed sessions, VLM, training, and performance prototypes
are marked `legacy`. They remain inspectable and runnable:

```bash
pytest -m legacy
```

They are not part of the production gate because several assert behavior that
is now intentionally unsafe or unsupported (for example, repo-root file tools
and anonymous generic tool execution). A legacy failure is migration evidence,
not a claim that the v2 core passed.

The React suite uses Vitest + Testing Library for auth and Story empty states.
`npm run lint` gates the mounted v2 surface; `npm run lint:legacy` exposes the
retained, unmounted v1 UI findings as migration inventory. CI additionally
checks the supported ESLint scope, OpenAPI-generated types, offline Alembic
SQL, production Compose, and the static portfolio asset graph. Scheduled AI
compatibility runs install the optional AI dependency group and never download
models during tests.
