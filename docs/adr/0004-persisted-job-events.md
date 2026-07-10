# ADR 0004: Persist job lifecycle events in the application transaction

Status: accepted — 2026-07-10

## Context

The `jobs` row exposes current state but cannot explain how a task reached that state. Broker logs are transient and do not provide a product-safe audit trail for retries, lease recovery, or worker fencing.

## Decision

Store append-only `job_events` beside each job. Application services append the event in the same database transaction as the state mutation. Events capture actor, transition, progress, attempt, execution/request correlation, sanitized details, and timestamp. The API returns events in chronological order with a bounded limit.

## Consequences

- A committed state transition always has its corresponding evidence; rolled-back work exposes neither.
- API, worker, scheduler, and admin actions can be distinguished without depending on Celery logs.
- Sensitive detail keys are redacted before persistence and payload size/depth is bounded.
- This is application-level observability, not a replacement for infrastructure metrics or distributed tracing.
