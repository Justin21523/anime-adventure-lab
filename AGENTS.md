# Repository Guidelines

## Project Structure & Module Organization
- API lives in `api/` (FastAPI routers, dependencies, middleware) with entrypoint `api/main.py`.
- Core logic sits in `core/` (LLM, RAG, story, T2I, VLM, training). Shared schemas live in `schemas/`.
- Celery tasks and workers are in `workers/`; configs (model, RAG, presets) in `configs/`.
- Frontend demos reside in `frontend/` (Gradio app at `frontend/gradio/app.py`).
- Tests are in `tests/`; scripts in `scripts/`; docs in `docs/`; infra/docker assets under `docker/`.
- Models and large assets are external—set `AI_CACHE_ROOT` to your mounted warehouse (see `.env.example`).

## Build, Test, and Development Commands
- Environment: `conda create -n ai_env python=3.10 -y && conda activate ai_env && pip install -r requirements.txt && pip install -r requirements-test.txt`.
- Run API locally: `uvicorn api.main:app --reload` (health at `http://localhost:8000/healthz`).
- Worker: `REDIS_URL=redis://localhost:6379/0 celery -A workers.celery_app:celery_app worker -l INFO`.
- Gradio demo: `python frontend/gradio/app.py` (`http://localhost:7860`).
- Docker (dev): `docker compose up --build`.
- Test helpers: `make test`, `make test-smoke`, or `./scripts/test_runner.sh smoke|unit|integration|e2e`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, type hints expected. Prefer `async` endpoints where applicable.
- Format and lint with `black`, `ruff`, and `isort` (`make format` / `make lint`).
- Naming: modules/packages `snake_case`, classes `PascalCase`, functions/vars `snake_case`. Keep routers and tasks scoped by domain (e.g., `api/routers/rag.py`, `workers/tasks/t2i.py`).
- Keep cache/bootstrap logic centralized; do not hardcode model paths—respect `AI_CACHE_ROOT`.

## Testing Guidelines
- Framework: `pytest` (config in `pytest.ini`). Markers available: `unit`, `integration`, `e2e`, `smoke`, `slow`.
- Naming: files `test_*.py`, classes `Test*`, functions `test_*`.
- Run all: `pytest -q` or `make test`. Targeted: `pytest -m smoke`, `pytest tests/test_api_endpoints.py -m integration`.
- Use lightweight fixtures/mocks; avoid pulling large models in tests. If GPU/large models are required, mark as `slow`.

## Commit & Pull Request Guidelines
- Follow conventional commit style seen in history: `feat(scope): ...`, `fix(scope): ...`, `chore/tests/docs/...`.
- Keep commits scoped and reviewable; include brief rationale in the body when touching infra or configs.
- PRs: include summary, linked issue, test evidence (`make test-smoke` or specific `pytest` runs), and note any model/config changes or new env vars. Add screenshots/GIFs for UI tweaks.

## Security & Configuration Tips
- Never commit secrets; use `.env` and document new keys in PRs. Ensure `AI_CACHE_ROOT` points to a mounted warehouse before running heavy tasks.
- Default to low-VRAM settings (bf16/fp16, device_map="auto") and prefer LoRA over full fine-tuning unless justified.
