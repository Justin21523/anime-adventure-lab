# ADR 0002: Postgres, pgvector, and MinIO

Status: Accepted

JSON files and process-local dictionaries cannot safely coordinate API and
worker updates. Postgres therefore becomes the authoritative state store,
pgvector becomes the production retrieval index, and MinIO stores binary
artifacts. JSON remains an import/export format only.

This adds infrastructure to a single-user app, but it is justified by atomic
Story turns, idempotent jobs, migration history, retrieval traceability, and
recoverable generated artifacts.
