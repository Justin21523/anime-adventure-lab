# README structure for the portfolio release

The existing root README has user-owned edits and should be migrated
deliberately. The release README should use this order:

1. One-sentence product statement and hero screenshot.
2. The authoring problem and target user.
3. Public static demo URL, private-demo note, and 90-second video.
4. Three core flows: durable Story turn, world-scoped lore, observable job.
5. Architecture diagram showing React → FastAPI → Postgres/pgvector, MinIO,
   Redis/Celery, and external AI workers.
6. Security boundary and public/private deployment separation.
7. Technology table with reasons, not only package names.
8. Local quick start, required environment variables, migrations, and bucket
   initialization.
9. Test, lint, type-check, build, and smoke commands.
10. Technical decisions with links to `docs/adr/`.
11. Demo seed data and screenshots.
12. Honest supported/experimental matrix and known limitations.
13. Roadmap focused on generated artifacts, lease reconciliation, and broader
    browser tests.

Avoid claiming that every legacy router is production-ready. Do not publish a
real test password, API key, local model path, or private warehouse layout.
