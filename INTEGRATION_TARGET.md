# 🏠 此專案為統一平台主力

**狀態：** ✅ 主力保留
**標記日期：** 2026-05-08

## 為什麼選擇這個專案

- 功能最全：25 個 API 路由，涵蓋 LLM/RAG/VLM/T2I/LoRA/Agent/Story/Worlds
- 代碼最多：255 個 Python 檔案，144 個前端組件
- 架構最完整：`core/agents/`、`core/train/`、`core/vlm/`、`core/worldpacks/`
- 有 Docker Compose、Redis/Celery、React 前端

## 待吸收的外部功能

| 來源 | 待吸收模組 | 預計放置位置 |
|------|-----------|-------------|
| sd-multimodal-platform | upscale_service, face_restore_service | `core/postprocess/` |
| sd-multimodal-platform | queue_manager | `core/queue/` |
| sd-multimodal-platform | deployment/k8s | `deployment/k8s/` |
| sd-multimodal-platform | frontend/gradio | `frontend/gradio/` |
| charaforge-T2I-Lab | workers/ (Celery) | `workers/` |
| charaforge-T2I-Lab | jwt_tokens, security | `core/security/` |
| charaforge-T2I-Lab | train_access | `core/train/` |
| charaforge-T2I-Lab | lora_trainer (專業版) | 替換 `core/train/lora_trainer.py` |
| charaforge-T2I-Lab | evaluators | `core/train/evaluators.py` |
| charaforge-T2I-Lab | WebSocket | `api/routers/ws.py` |

## 現有弱點需補強

- 訓練執行器預設模擬模式 → 需要 charaforge 的 Celery 真實訓練
- 認證系統只有基本版 → 需要 charaforge 的 JWT
- Docker Compose 比 sd-multimodal 簡單 → 需要整合

**完整分析：** 見 `/mnt/c/ai_projects/PROJECT_POSITIONING_ANALYSIS.md`
