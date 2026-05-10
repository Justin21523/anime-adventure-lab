#!/bin/bash
# run_local.sh - Direct native startup (no Docker)

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
START_WORKER="${START_WORKER:-0}"
START_REDIS="${START_REDIS:-0}"
CLEAN_PORTS="${CLEAN_PORTS:-1}"

echo -e "${GREEN}Launching Anime Adventure Lab in native mode...${NC}"

load_env_file() {
    local env_file="$1"
    [ -f "$env_file" ] || return 0
    echo -e "${GREEN}Loading ${env_file}...${NC}"
    while IFS= read -r line || [[ -n "$line" ]]; do
        trimmed=$(echo "$line" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if [[ "$trimmed" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            key=$(echo "$trimmed" | cut -d'=' -f1)
            val=$(echo "$trimmed" | cut -d'=' -f2- | sed 's/[[:space:]]*#.*$//' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            if [[ "$val" =~ ^\".*\"$ ]] || [[ "$val" =~ ^\'.*\'$ ]]; then
                val="${val:1:${#val}-2}"
            fi
            export "$key"="$val"
        fi
    done < "$env_file"
}

load_env_file ".env"

API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
export VITE_PROXY_TARGET="${VITE_PROXY_TARGET:-http://127.0.0.1:${API_PORT}}"
export VITE_API_BASE="${VITE_API_BASE:-/api/v1}"
export T2I_MOCK="${T2I_MOCK:-0}"
export VLM_MOCK="${VLM_MOCK:-0}"
export LLM_MOCK="${LLM_MOCK:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_DEVICE="${MODEL_DEVICE:-cuda}"
export MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-auto}"
export MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE:-float16}"
export MODEL_USE_4BIT_LOADING="${MODEL_USE_4BIT_LOADING:-true}"

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo -e "${YELLOW}.venv not found; using current Python environment.${NC}"
fi

python - <<'PY'
import os
import sys
import torch

real_modes = [
    name for name in ("T2I_MOCK", "VLM_MOCK", "LLM_MOCK")
    if os.getenv(name, "0").strip().lower() in {"0", "false", "no", "off"}
]

print(
    "[GPU] cuda_available=%s device_count=%s current_device=%s"
    % (
        torch.cuda.is_available(),
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    )
)
print(
    "[AI] T2I_MOCK=%s VLM_MOCK=%s LLM_MOCK=%s MODEL_DEVICE=%s MODEL_DEVICE_MAP=%s MODEL_TORCH_DTYPE=%s"
    % (
        os.getenv("T2I_MOCK", "0"),
        os.getenv("VLM_MOCK", "0"),
        os.getenv("LLM_MOCK", "0"),
        os.getenv("MODEL_DEVICE", "cuda"),
        os.getenv("MODEL_DEVICE_MAP", "auto"),
        os.getenv("MODEL_TORCH_DTYPE", "float16"),
    )
)

if real_modes and not torch.cuda.is_available():
    print(
        "ERROR: real AI mode is enabled (%s), but CUDA is unavailable. "
        "Fix CUDA/PyTorch or set *_MOCK=1 explicitly for test mode."
        % ", ".join(real_modes),
        file=sys.stderr,
    )
    sys.exit(1)
PY

cleanup() {
    echo -e "\n${RED}Stopping native services...${NC}"
    kill ${BACKEND_PID:-} ${WORKER_PID:-} ${FRONTEND_PID:-} ${REDIS_PID:-} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

stop_port_processes() {
    local port="$1"
    if [ "$CLEAN_PORTS" != "1" ]; then
        return 0
    fi
    # 1. Clean local processes
    if command -v lsof >/dev/null 2>&1; then
        local pids
        pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo -e "${YELLOW}Port ${port} is in use; stopping local process(es): ${pids}${NC}"
            kill -9 $pids 2>/dev/null || true
            sleep 1
        fi
    fi
    # 2. Clean Docker containers
    if command -v docker >/dev/null 2>&1; then
        local cont
        cont=$(docker ps -q --filter "publish=$port")
        if [ -n "$cont" ]; then
            echo -e "${YELLOW}Port ${port} is in use by Docker container ${cont}; removing it...${NC}"
            docker rm -f "$cont" >/dev/null
        fi
    fi
}

stop_port_processes "$API_PORT"
stop_port_processes "$FRONTEND_PORT"
stop_port_processes "11434"

if [ "$START_REDIS" = "1" ]; then
    if redis-cli ping >/dev/null 2>&1; then
        echo -e "${GREEN}Local Redis is already running.${NC}"
    elif command -v redis-server >/dev/null 2>&1; then
        echo -e "${GREEN}Starting local redis-server...${NC}"
        redis-server --port 6379 --save "" --appendonly no &
        REDIS_PID=$!
    else
        echo -e "${YELLOW}START_REDIS=1 was set, but redis-server is not installed. Worker will be skipped.${NC}"
        START_WORKER=0
    fi
fi

echo -e "${GREEN}Starting Backend API on http://127.0.0.1:${API_PORT}...${NC}"
python -m uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" &
BACKEND_PID=$!

echo -e "${GREEN}Waiting for Backend API health check...${NC}"
for i in {1..60}; do
    if curl -fsS "http://127.0.0.1:${API_PORT}/healthz" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${RED}Backend API exited before becoming healthy.${NC}"
        wait "$BACKEND_PID"
    fi
    sleep 1
    if [ "$i" = "60" ]; then
        echo -e "${RED}Backend API did not become healthy within 60 seconds.${NC}"
        exit 1
    fi
done

if [ "$START_WORKER" = "1" ]; then
    if python -c "import celery, redis" >/dev/null 2>&1; then
        echo -e "${GREEN}Starting Celery worker...${NC}"
        python scripts/start_worker.py --loglevel INFO &
        WORKER_PID=$!
    else
        echo -e "${YELLOW}Celery/redis Python packages are not available; skipping worker.${NC}"
    fi
else
    echo -e "${YELLOW}Worker disabled. Set START_WORKER=1 to run it with a local Redis server.${NC}"
fi

echo -e "${GREEN}Starting React dev server on http://localhost:${FRONTEND_PORT}...${NC}"
cd frontend/react
npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" &
FRONTEND_PID=$!
cd ../..

echo -e "\n${GREEN}Native services are running.${NC}"
echo -e "Frontend:   ${YELLOW}http://localhost:${FRONTEND_PORT}${NC}"
echo -e "Backend:    ${YELLOW}http://127.0.0.1:${API_PORT}${NC}"
echo -e "API health: ${YELLOW}http://127.0.0.1:${API_PORT}/healthz${NC}"
echo -e "Proxy:      ${YELLOW}/api/v1 -> ${VITE_PROXY_TARGET}${NC}"
echo -e "Press ${RED}Ctrl+C${NC} to stop."

wait
