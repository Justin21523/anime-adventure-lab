# 前端操作指南（React / Vite）

本專案前端已收斂為 **React（Vite）**，並以 **Story（主流程）** 為唯一主畫面；RAG / T2I / Batch / Agent 以「故事工作台」的方式內嵌在故事畫面中，而不是各自獨立頁面。

## 環境與啟動

### 1) 啟動後端（FastAPI）

```bash
conda activate ai_env
cp .env.example .env

# 建議先用 mock 模式避免下載大模型
VLM_MOCK=1 T2I_MOCK=1 uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- API base：`http://localhost:8000/api/v1`
- 健康檢查：`http://localhost:8000/api/v1/health`

### 2) 啟動前端（React / Vite）

```bash
cd frontend/react
npm install
npm run dev
```

- 前端：`http://localhost:3000`
- Vite proxy：`/api/v1/* -> http://localhost:8000`（見 `frontend/react/vite.config.ts`）

## UI 流程（Story-first）

1) 進入「故事」列表 → 建立新故事 → 進入故事畫面  
2) 在故事畫面右上角點「工作台」：
   - **場景生成**：T2I 生成 + 歷史（會帶 `session_id`）
   - **RAG**：儀表板 / 搜索 / 上傳 / 文件列表（會帶 `world_id`）
   - **批次**：任務監控、取消、下載結果
   - **Agent**：任務執行與工具瀏覽

## Celery / Redis（可選）

批次任務/訓練等需要 Redis + Celery worker：

```bash
redis-server --port 6379
REDIS_URL=redis://localhost:6379/0 celery -A workers.celery_app:celery_app worker -l INFO
```

## 常見注意事項

- 無網路或無 GPU：請用 `VLM_MOCK=1`、`T2I_MOCK=1` 避免模型下載/初始化失敗。
- RAG/批次/訓練/匯出會寫入 `AI_OUTPUT_ROOT`（以及可選的 `AI_RAG_ROOT` / `AI_DATASETS_ROOT`），請確認路徑可寫。
