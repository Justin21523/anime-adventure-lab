# Deployment guide

SagaForge has two intentionally separate deployment targets.

## Public portfolio demo

`portfolio-web/` is deterministic, anonymous, static, and contains no API
credentials. `.github/workflows/pages.yml` deploys only that directory to
GitHub Pages. Never point the public demo at the private API.

## Private Story workbench

The private stack is defined in `docker-compose.prod.yml`:

- `frontend`: Nginx-served React application;
- `api`: CPU-light FastAPI composition root;
- `worker`: CPU-safe durable Story and RAG jobs by default;
- `scheduler`: Celery beat reconciliation for deferred and expired jobs;
- `postgres`: authoritative relational state plus pgvector;
- `redis`: task transport and transient progress;
- `minio`: uploads, generated artifacts, and exports;
- one-shot `migrate` and `minio-init` services.

### Required configuration

Copy `.env.example` to `.env` and set unique values for:

```dotenv
POSTGRES_PASSWORD=...
DATABASE_URL=postgresql://saga:...@postgres:5432/sagaforge
MINIO_USER=...
MINIO_PASSWORD=...
API_SECRET_KEY=...
API_SESSION_SECRET=...
API_ADMIN_PASSWORD_HASH=...
```

Generate the Argon2 password hash without putting plaintext into `.env`:

```bash
python -c "from argon2 import PasswordHasher; print(PasswordHasher().hash(input('Password: ')))"
```

Set the remote LLM endpoint or use the deterministic runtime for a smoke
deployment:

```dotenv
LLM_BACKEND=llamacpp
LLAMA_SERVER_URL=http://host.docker.internal:8080
STORY_RUNTIME_MODE=llm
RAG_RUNTIME_MODE=model
```

### Start and verify

```bash
docker compose -f docker-compose.prod.yml config --quiet
docker compose -f docker-compose.prod.yml up --build -d
docker compose -f docker-compose.prod.yml ps
curl --fail http://localhost:8000/healthz
```

For the reproducible interview profile, layer the demo override and seed only
the fixed `moon-archive` namespace:

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml up --build -d
docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml exec api \
  python scripts/seed_demo.py --apply --reset
```

The seed command defaults to dry-run. `--reset` never deletes non-demo worlds.

The API publishes only port 8000; Postgres, Redis, and MinIO remain internal.
The frontend is available at `http://localhost` by default. Production TLS must
terminate at a reverse proxy; keep `API_COOKIE_SECURE=1`.

The default `WORKER_PROFILE=core` does not import PyTorch. Provision a separate
image with the AI dependencies and set `WORKER_PROFILE=experimental` only when
legacy VLM, T2I, or training queues are deliberately enabled. Flower is also
opt-in and loopback-only:

```bash
docker compose -f docker-compose.prod.yml --profile ops up -d flower
```

### Migrations and import

Compose runs `alembic upgrade head` before starting the API. For a manual
deployment:

```bash
alembic upgrade head
python scripts/import_legacy_data.py
python scripts/import_legacy_data.py --apply
```

Always inspect the dry-run report and back up Postgres and MinIO before import.

### Backup and recovery

Back up all authoritative stores together:

```bash
docker compose -f docker-compose.prod.yml exec -T postgres \
  pg_dump -U saga -Fc sagaforge > sagaforge.dump
```

Mirror the MinIO buckets (`uploads`, `generated`, `exports`) with `mc mirror`.
Redis is not authoritative and does not replace those backups.

### Rollback

Deploy the previous application image first. Run Alembic downgrade only after
reviewing the generated SQL; schema rollback can discard new-version data.
Never use `Base.metadata.drop_all()` in deployment automation.
# Complete local deployment

The production-like local deployment includes PostgreSQL/pgvector, Redis,
MinIO, migration job, FastAPI, Celery worker, Celery scheduler, the authenticated
React workbench, and a separate static portfolio container.

Create an ignored `.env.deploy` from `.env.example`, configure strong local
secrets and a valid Argon2 `API_ADMIN_PASSWORD_HASH`, then run:

```bash
make deploy-up
make deploy-status
```

Default endpoints (override their port variables when occupied):

- Workbench: `http://localhost:3000` or `FRONTEND_PORT`
- API health: `http://localhost:8000/healthz` or `API_PORT`
- Public case study: `http://localhost:8081` or `PORTFOLIO_PORT`

Operational commands:

```bash
make deploy-logs
make deploy-down        # preserves named data volumes
make deploy-up          # reruns migration safely and restores persisted data
docker compose --env-file .env.deploy -f docker-compose.prod.yml -f docker-compose.demo.yml down -v  # destructive reset
```

Do not expose the deterministic demo overlay directly to the internet. For a
remote production deployment, terminate TLS at a trusted reverse proxy, set
secure cookies, restrict CORS/hosts, rotate all `.env.deploy` credentials, and
use managed secret storage rather than an environment file.
