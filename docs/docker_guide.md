# Docker 部署指南

## 概述

本指南說明如何使用 Docker 部署 Anime-Adventure-Lab 項目。

## 前置要求

### 系統要求

**最低配置**:
- CPU: 4 cores
- RAM: 8GB
- 硬碟: 50GB 可用空間
- Docker: 20.10+
- Docker Compose: 2.0+

**推薦配置**:
- CPU: 8+ cores
- RAM: 16GB+
- 硬碟: 100GB+ SSD
- GPU: NVIDIA GPU (CUDA 11.8+)
- Docker with NVIDIA Container Toolkit

### 軟體安裝

#### 1. 安裝 Docker

**Ubuntu/Debian**:
```bash
# 更新套件索引
sudo apt-get update

# 安裝依賴
sudo apt-get install ca-certificates curl gnupg

# 添加 Docker 官方 GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 添加 Docker 倉庫
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安裝 Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 驗證安裝
sudo docker run hello-world
```

**Windows/macOS**:
- 下載並安裝 [Docker Desktop](https://www.docker.com/products/docker-desktop)

#### 2. 安裝 NVIDIA Container Toolkit (GPU 支持)

```bash
# 添加 NVIDIA 倉庫
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安裝
sudo apt-get update && sudo apt-get install -y nvidia-docker2

# 重啟 Docker
sudo systemctl restart docker

# 測試 GPU 支持
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 快速開始

### 1. 克隆項目

```bash
git clone https://github.com/your-org/anime-adventure-lab.git
cd anime-adventure-lab
```

### 2. 配置環境變數

```bash
# 複製環境變數範本
cp .env.example .env

# 編輯配置（必須設置 AI_CACHE_ROOT）
nano .env
```

**最小必要配置**:
```bash
# .env
AI_CACHE_ROOT=/mnt/c/AI_LLM_projects/ai_warehouse
AI_WAREHOUSE_HOST=/mnt/c/AI_LLM_projects/ai_warehouse
POSTGRES_PASSWORD=your_secure_password
MINIO_PASSWORD=your_minio_password
```

### 3. 創建模型倉儲目錄

```bash
# 創建倉儲目錄
mkdir -p /mnt/c/AI_LLM_projects/ai_warehouse/cache

# 設置權限
chmod -R 755 /mnt/c/AI_LLM_projects/ai_warehouse
```

### 4. 啟動服務

**開發環境（最簡化）**:
```bash
docker compose up --build
```

**生產環境（完整堆棧）**:
```bash
docker compose -f docker-compose.prod.yml up -d
```

**批次處理環境**:
```bash
docker compose -f docker-compose.batch.yml up -d
```

### 5. 驗證部署

```bash
# 檢查服務狀態
docker compose ps

# 測試 API
curl http://localhost:8000/healthz

# 測試 Gradio UI（如果啟動）
open http://localhost:7860
```

## Docker Compose 配置說明

### docker-compose.yml（開發）

**最簡配置，適合快速開發和測試**:

```yaml
services:
  api:     # FastAPI 後端
  worker:  # Celery 背景任務
  redis:   # 快取和訊息佇列
```

**啟動**:
```bash
docker compose up --build
```

**特點**:
- 輕量化，啟動快速
- 僅包含核心服務
- 適合開發和單機測試

### docker-compose.prod.yml（生產）

**完整堆棧，適合生產部署**:

```yaml
services:
  postgres:  # 資料庫（含 pgvector）
  redis:     # 快取
  minio:     # 物件儲存
  api:       # API 服務器
  worker:    # 背景任務
  webui:     # Gradio UI（可選）
  flower:    # Celery 監控（可選）
```

**啟動**:
```bash
docker compose -f docker-compose.prod.yml up -d
```

**特點**:
- 完整功能
- 資料持久化
- 健康檢查
- 自動重啟
- GPU 支持

### docker-compose.batch.yml（批次處理）

**針對大規模批次任務優化**:

```yaml
services:
  # ... 基礎服務 ...
  worker-t2i:     # T2I 專用 worker
  worker-rag:     # RAG 專用 worker
  worker-train:   # 訓練專用 worker
```

**啟動**:
```bash
docker compose -f docker-compose.batch.yml up -d
```

**特點**:
- 多個專用 worker
- 分離的任務隊列
- 資源隔離

## 常用操作

### 查看日誌

```bash
# 查看所有服務日誌
docker compose logs -f

# 查看特定服務日誌
docker compose logs -f api
docker compose logs -f worker

# 查看最近 100 行
docker compose logs --tail=100 api
```

### 重啟服務

```bash
# 重啟所有服務
docker compose restart

# 重啟特定服務
docker compose restart api
docker compose restart worker
```

### 停止服務

```bash
# 停止所有服務（保留資料）
docker compose stop

# 停止並移除容器（保留資料卷）
docker compose down

# 完全清理（包括資料卷）
docker compose down -v
```

### 擴展 Worker

```bash
# 增加 worker 數量
docker compose up -d --scale worker=4

# 生產環境
docker compose -f docker-compose.prod.yml up -d --scale worker=4
```

### 更新映像

```bash
# 拉取最新映像
docker compose pull

# 重新構建並啟動
docker compose up -d --build

# 強制重建（不使用快取）
docker compose build --no-cache
docker compose up -d
```

### 進入容器

```bash
# 進入 API 容器
docker compose exec api bash

# 進入 worker 容器
docker compose exec worker bash

# 執行 Python 命令
docker compose exec api python -c "from core.rag import get_rag_engine; print('OK')"
```

### 數據備份與恢復

**備份**:
```bash
# 備份 PostgreSQL
docker compose exec postgres pg_dump -U saga sagaforge > backup.sql

# 備份 Redis
docker compose exec redis redis-cli SAVE
docker cp $(docker compose ps -q redis):/data/dump.rdb backup_redis.rdb

# 備份倉儲
tar -czf warehouse_backup.tar.gz /mnt/c/AI_LLM_projects/ai_warehouse
```

**恢復**:
```bash
# 恢復 PostgreSQL
cat backup.sql | docker compose exec -T postgres psql -U saga sagaforge

# 恢復 Redis
docker cp backup_redis.rdb $(docker compose ps -q redis):/data/dump.rdb
docker compose restart redis
```

## 環境變數配置

### 核心配置

```bash
# 倉儲路徑（必須）
AI_CACHE_ROOT=/warehouse  # 容器內路徑
AI_WAREHOUSE_HOST=/mnt/c/AI_LLM_projects/ai_warehouse  # 主機路徑

# 資料庫
DATABASE_URL=postgresql://saga:password@postgres:5432/sagaforge

# Redis
REDIS_URL=redis://redis:6379/0
```

### GPU 配置

```bash
# 指定使用的 GPU
CUDA_VISIBLE_DEVICES=0  # 單 GPU
CUDA_VISIBLE_DEVICES=0,1  # 多 GPU

# 在 docker-compose.yml 中配置
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1  # 或 all
              capabilities: [gpu]
```

### 性能調優

```bash
# 量化模式
QUANTIZATION_MODE=int8  # none, int8, int4

# 批次大小
MAX_BATCH_SIZE=32
OPTIMAL_BATCH_SIZE=8

# 緩存
EMBEDDING_CACHE_TTL=604800  # 7 天
ENABLE_KV_CACHE=true
```

## 監控和維護

### 健康檢查

```bash
# API 健康
curl http://localhost:8000/healthz

# 監控端點
curl http://localhost:8000/monitoring/health
curl http://localhost:8000/monitoring/metrics

# 查看容器健康狀態
docker compose ps
```

### 資源監控

```bash
# Docker 統計
docker stats

# 特定容器
docker stats anime-adventure-lab-api-1

# GPU 使用率
nvidia-smi

# 在容器內查看
docker compose exec api nvidia-smi
```

### Celery 監控

**使用 Flower**:

```bash
# 添加到 docker-compose.prod.yml
services:
  flower:
    image: mher/flower
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    ports:
      - "5555:5555"
    depends_on:
      - redis

# 訪問
open http://localhost:5555
```

**命令行監控**:
```bash
# 查看活躍任務
docker compose exec worker celery -A workers.celery_app inspect active

# 查看統計
docker compose exec worker celery -A workers.celery_app inspect stats
```

## 故障排除

### 容器無法啟動

**問題**: 端口被佔用
```bash
# 檢查端口使用
sudo lsof -i :8000
sudo lsof -i :6379

# 修改端口（docker-compose.yml）
ports:
  - "8001:8000"  # 使用不同端口
```

**問題**: 權限錯誤
```bash
# 檢查目錄權限
ls -la /mnt/c/AI_LLM_projects/ai_warehouse

# 修復權限
sudo chown -R $USER:$USER /mnt/c/AI_LLM_projects/ai_warehouse
chmod -R 755 /mnt/c/AI_LLM_projects/ai_warehouse
```

### GPU 不可用

```bash
# 檢查 NVIDIA 驅動
nvidia-smi

# 檢查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 檢查容器內 GPU
docker compose exec api nvidia-smi
```

### 記憶體不足

**解決方案**:

1. **啟用量化**:
```bash
# .env
QUANTIZATION_MODE=int8  # 或 int4
LOW_VRAM_MODE=true
```

2. **減少 Worker 數量**:
```bash
docker compose up -d --scale worker=1
```

3. **增加 Swap**:
```bash
# 創建 8GB swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 連接錯誤

**Redis 連接失敗**:
```bash
# 檢查 Redis 狀態
docker compose exec redis redis-cli ping

# 檢查連接
docker compose exec api python -c "import redis; r=redis.from_url('redis://redis:6379/0'); print(r.ping())"
```

**資料庫連接失敗**:
```bash
# 檢查 PostgreSQL
docker compose exec postgres pg_isready -U saga -d sagaforge

# 測試連接
docker compose exec api python -c "from sqlalchemy import create_engine; e=create_engine('postgresql://saga:saga123@postgres:5432/sagaforge'); print(e.connect())"
```

## 生產部署檢查清單

### 部署前

- [ ] 設置強密碼（資料庫、Redis、MinIO）
- [ ] 配置正確的 CORS 來源
- [ ] 啟用 API 認證（如需要）
- [ ] 配置 SSL/TLS（使用 Nginx 反向代理）
- [ ] 設置日誌輪替
- [ ] 配置備份策略
- [ ] 測試健康檢查端點

### 部署後

- [ ] 驗證所有服務正常運行
- [ ] 測試 API 端點
- [ ] 檢查日誌無錯誤
- [ ] 監控資源使用
- [ ] 設置告警（Sentry, 監控系統）
- [ ] 測試備份恢復流程

### 安全

- [ ] 使用環境變數管理敏感信息
- [ ] 限制容器網絡訪問
- [ ] 啟用防火牆規則
- [ ] 定期更新映像
- [ ] 掃描安全漏洞
- [ ] 限制容器特權

## 高級配置

### 多節點部署

使用 Docker Swarm 或 Kubernetes 進行多節點部署：

**Docker Swarm**:
```bash
# 初始化 swarm
docker swarm init

# 部署堆棧
docker stack deploy -c docker-compose.prod.yml sagaforge

# 擴展服務
docker service scale sagaforge_worker=4
```

### 自定義網絡

```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # 不對外

services:
  api:
    networks:
      - frontend
      - backend
  postgres:
    networks:
      - backend  # 僅內部網絡
```

### 資源限制

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 8G
        reservations:
          cpus: '1.0'
          memory: 4G
```

## 參考資源

- [Docker 官方文檔](https://docs.docker.com/)
- [Docker Compose 文檔](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [項目 GitHub](https://github.com/your-org/anime-adventure-lab)
