# Phase 2 吸收記錄 — charaforge-T2I-Lab → anime-adventure-lab

> 執行日期：2026-05-09

## 吸收清單

| 來源 (charaforge) | 目的地 (anime-adventure-lab) | 狀態 | 備註 |
|---|---|---|---|
| `api/jwt_tokens.py` | `api/jwt_tokens.py` | ✅ | 純函數模組，直接複製 |
| `api/t2i_tokens.py` | `api/t2i_tokens.py` | ✅ | 改用 `~/.anime-adventure-lab` 取代 `get_cache_paths()` |
| `api/security.py` | `api/security.py` | ✅ | 新增 RateLimiter + API key 解析，Redis prefix 改為 `anime_adventure:` |
| `api/train_access.py` | `api/train_access.py` | ✅ | 改用 `os.getenv("DATA_DIR")` 取代 `get_app_paths()` |
| `api/t2i_cost.py` | `api/t2i_cost.py` | ✅ | 純函數，直接複製 |
| `api/ws_tickets.py` | `api/ws_tickets.py` | ✅ | 直接複製 |
| `api/routers/ws.py` | `api/routers/ws.py` | ✅ | 簡化：改用 `get_config()` 取代 `get_settings()`，Redis channel 改為 `anime_adventure:train:` |
| `workers/celery_app.py` (TaskProgress) | `workers/celery_app.py` (追加) | ✅ | 加入 TaskProgress 類別 + `postprocess` queue + task events |
| `core/train/lora_trainer.py` | *(不吸收)* | ⏭️ | anime-adventure-lab 已有自己的 `core.train.lora_trainer` + `executor` + `sdxl_lora_trainer` |
| `workers/tasks/training.py` | *(不吸收)* | ⏭️ | anime-adventure-lab 已有自己的 Celery training task 架構 |

## 路由註冊

- `api/routers/__init__.py` — 新增 `ws_router`
- `api/main.py` — 新增 `ws_router` import + `app.include_router(ws_router)`

## Docker Compose 變更

- 新增 `worker-training` 服務（獨立 training queue，concurrency=1）
- 新增 `flower` 服務（Celery 監控 UI，port 5555）

## 依賴

- JWT/t2i tokens：純 Python 標準庫（無新增依賴）
- WebSocket：需 `redis[asyncio]`（已含於 `redis>=5.0.0`）
- RateLimiter：需 `redis`（已有）

## 備份

- `docker-compose.yml.bak.pre-merge` — Phase 1 備份仍有效
