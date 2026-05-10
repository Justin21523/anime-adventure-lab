# Phase 1 吸收記錄 — sd-multimodal-platform → anime-adventure-lab

> 執行日期：2026-05-08

## 吸收清單

| 來源 | 目的地 | 狀態 |
|------|-------|------|
| `sd-multimodal-platform/deployment/k8s/` | `deployment/k8s/` | ✅ |
| `sd-multimodal-platform/deployment/docker/` | `deployment/docker/` | ✅ |
| `sd-multimodal-platform/app/services/postprocess/` | `core/postprocess/` | ✅ |
| `sd-multimodal-platform/app/core/queue_manager.py` | `core/queue/queue_manager.py` | ✅ |
| `sd-multimodal-platform/app/schemas/queue_requests.py` | `core/queue/queue_requests.py` | ✅ |
| `sd-multimodal-platform/app/api/v1/queue.py` | `api/routers/queue.py` | ✅ |
| `sd-multimodal-platform/frontend/gradio/` | `frontend/gradio/` | ✅ |
| `sd-multimodal-platform/frontend/gradio_app/` | `frontend/gradio/` (合併) | ✅ |
| `sd-multimodal-platform/frontend/desktop/` | `frontend/desktop/` | ✅ |

## Docker Compose 變更

- 原 `worker` 服務 → 拆分為 `worker-generation` + `worker-postprocess`
- 加入 `CELERY_WORKER_CONCURRENCY` 環境變數
- 加入 `frontend_node_modules` volume
- 保留 anime-adventure-lab 原有的 YAML anchors 架構

## 備份

- `docker-compose.yml.bak.pre-merge` — 原始 docker-compose 備份

## 已完成修正

- [x] 重寫 `core/postprocess/` — 原 sd-multimodal 是 shim，已重寫為獨立實作（lazy-load optional deps）
- [x] 重寫 `core/queue/queue_manager.py` — 改用 `core.queue` 路徑，移除 `app.config` 依賴
- [x] 重寫 `core/queue/queue_requests.py` — 移除 `app.schemas` 依賴
- [x] 重寫 `api/routers/queue.py` — 改用 `core.queue` import
- [x] 註冊 queue router — `api/main.py` + `api/routers/__init__.py` 已加入
- [x] requirements.txt 加入 postprocess 可選依賴
- [x] 所有 Python 檔案語法檢查通過

## 需後續處理

- [ ] 實際環境中測試 postprocess 功能（需安裝 realesrgan/gfpgan）
- [ ] 測試 queue API 端點
