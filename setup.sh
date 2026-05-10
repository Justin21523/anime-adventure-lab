#!/bin/bash
# setup.sh - Initial environment setup for Anime Adventure Lab

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔧 Setting up Anime Adventure Lab environment...${NC}"

# 1. Python Environment
echo -e "${GREEN}🐍 Setting up Python virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo -e "${GREEN}📦 Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-test.txt

# 2. Frontend Environment
echo -e "${GREEN}📦 Installing Frontend dependencies...${NC}"
cd frontend/react
npm install
cd ../..

# 3. .env Setup
if [ ! -f .env ]; then
    echo -e "${GREEN}📝 Creating .env from .env.example...${NC}"
    cp .env.example .env
fi

# Auto-configure local drive paths if missing
if ! grep -q "^AI_MODELS_ROOT=" .env && [ -d "/mnt/c/ai_models" ]; then
    echo -e "${GREEN}✨ Detected /mnt/c/ai_models. Configuring in .env...${NC}"
    echo "AI_MODELS_ROOT=/mnt/c/ai_models" >> .env
fi
if ! grep -q "^AI_CACHE_ROOT=" .env && [ -d "/mnt/c/ai_cache" ]; then
    echo -e "${GREEN}✨ Detected /mnt/c/ai_cache. Configuring in .env...${NC}"
    echo "AI_CACHE_ROOT=/mnt/c/ai_cache" >> .env
fi

echo -e "${YELLOW}📢 Please review .env file and adjust paths if necessary.${NC}"

# 4. Warehouse Setup
echo -e "${GREEN}📁 Creating local warehouse directories...${NC}"
mkdir -p warehouse/ai_cache warehouse/ai_models warehouse/ai_output

echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo -e "🚀 You can now start the project using:"
echo -e "   - ${YELLOW}./start.sh${NC} (Docker Mode - Recommended)"
echo -e "   - ${YELLOW}./dev.sh${NC}   (Local Dev Mode)"
