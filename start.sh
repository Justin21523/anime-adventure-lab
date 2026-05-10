#!/bin/bash
# start.sh - One-key startup script for Anime Adventure Lab (Robust Edition)

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Anime Adventure Lab System...${NC}"

# 1. Initialize .env
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env missing. Creating from example...${NC}"
    cp .env.example .env
fi

# 2. Auto-Detect C Drive Paths
if [ -d "/mnt/c/ai_models" ] && ! grep -q "AI_MODELS_ROOT=/mnt/c/ai_models" .env; then
    echo -e "${GREEN}✨ Found C: Drive Models. Auto-configuring...${NC}"
    sed -i '/AI_MODELS_ROOT/d' .env
    echo "AI_MODELS_ROOT=/mnt/c/ai_models" >> .env
fi
if [ -d "/mnt/c/ai_cache" ] && ! grep -q "AI_CACHE_ROOT=/mnt/c/ai_cache" .env; then
    echo -e "${GREEN}✨ Found C: Drive Cache. Auto-configuring...${NC}"
    sed -i '/AI_CACHE_ROOT/d' .env
    echo "AI_CACHE_ROOT=/mnt/c/ai_cache" >> .env
fi

# 3. Clean invalid paths (like /mnt/c/AI_LLM_projects/ai_warehouse)
sed -i '/ai_warehouse/d' .env

# 4. Load .env variables safely
echo -e "${GREEN}📖 Parsing configuration...${NC}"
while IFS= read -r line || [[ -n "$line" ]]; do
    trimmed=$(echo "$line" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [[ "$trimmed" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
        key=$(echo "$trimmed" | cut -d'=' -f1)
        val=$(echo "$trimmed" | cut -d'=' -f2- | sed 's/[[:space:]]*#.*$//' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        export "$key"="$val"
    fi
done < .env

# 5. Port Conflict Resolution
echo -e "${GREEN}🔍 Cleaning port conflicts (6379, 8000, 3000, 11434)...${NC}"
for port in 6379 8000 3000 11434; do
    # Docker containers
    CONT=$(docker ps -q --filter "publish=$port")
    [ ! -z "$CONT" ] && docker rm -f $CONT >/dev/null
    # Local processes
    PID=$(lsof -t -i:$port 2>/dev/null || true)
    [ ! -z "$PID" ] && kill -9 $PID 2>/dev/null || true
done

# 6. Mode Selection
echo -e "\n${YELLOW}🛠️  Select Mode:${NC}"
echo "1) Mock Mode (Fast, CPU only)"
echo "2) Production Mode (Real Models, GPU Required)"
read -p "Choice [1-2, default 1]: " mode_choice

if [ "$mode_choice" == "2" ]; then
    echo -e "${GREEN}🔥 Launching PRODUCTION Mode (Real AI)...${NC}"
    export T2I_MOCK=0 VLM_MOCK=0 LLM_MOCK=0
else
    echo -e "${GREEN}🧪 Launching MOCK Mode...${NC}"
    export T2I_MOCK=1 VLM_MOCK=1 LLM_MOCK=1
fi

# 7. Start
docker compose up -d --build
echo -e "${GREEN}✅ System is coming UP. Wait for Healthy status.${NC}"
echo -e "💡 Use 'docker compose logs -f api' to monitor model loading."
