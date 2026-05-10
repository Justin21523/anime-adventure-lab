#!/bin/bash
# Simple script to verify Mock mode is active and working
# Usage: ./scripts/verify_mock_mode.sh

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000/api/v1"

echo -e "${BOLD}${BLUE}========================================${NC}"
echo -e "${BOLD}${BLUE}  Mock Mode Verification${NC}"
echo -e "${BOLD}${BLUE}========================================${NC}\n"

# Check if backend is running
echo -e "${BOLD}Checking backend status...${NC}"
if ! curl -s -f "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}✗ Backend is not running!${NC}"
    echo -e "  Start it with: ${YELLOW}conda run -n ai_env python api/main.py${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Backend is running${NC}\n"

# Get health status
echo -e "${BOLD}Fetching system status...${NC}"
HEALTH=$(curl -s "$API_URL/health")

# Check CUDA status
CUDA_AVAILABLE=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin)['cache']['gpu_info']['cuda_available'])" 2>/dev/null || echo "false")
DEVICE=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin)['config']['model']['device_map'])" 2>/dev/null || echo "unknown")

echo -e "\n${BOLD}Current Configuration:${NC}"
echo -e "  CUDA Available: ${CUDA_AVAILABLE}"
echo -e "  Device:         ${DEVICE}"

# Check .env settings
echo -e "\n${BOLD}Environment Variables:${NC}"
if [ -f ".env" ]; then
    T2I_MOCK=$(grep "^T2I_MOCK=" .env | cut -d'=' -f2 || echo "not set")
    VLM_MOCK=$(grep "^VLM_MOCK=" .env | cut -d'=' -f2 || echo "not set")
    LLM_MOCK=$(grep "^LLM_MOCK=" .env | cut -d'=' -f2 || echo "not set")
    MODEL_DEVICE=$(grep "^MODEL_DEVICE=" .env | cut -d'=' -f2 || echo "not set")

    echo -e "  T2I_MOCK:      ${T2I_MOCK}"
    echo -e "  VLM_MOCK:      ${VLM_MOCK}"
    echo -e "  LLM_MOCK:      ${LLM_MOCK}"
    echo -e "  MODEL_DEVICE:  ${MODEL_DEVICE}"
else
    echo -e "  ${YELLOW}.env file not found${NC}"
fi

# Verify Mock Mode
echo -e "\n${BOLD}Mock Mode Verification:${NC}"
IS_MOCK=true

if [ "$CUDA_AVAILABLE" = "true" ]; then
    echo -e "  ${YELLOW}⚠ CUDA is available (may load real models)${NC}"
    IS_MOCK=false
fi

if [ "$DEVICE" != "cpu" ] && [ "$DEVICE" != "unknown" ]; then
    echo -e "  ${YELLOW}⚠ Device is not CPU: $DEVICE${NC}"
    IS_MOCK=false
fi

if [ "${T2I_MOCK}" = "0" ] || [ "${VLM_MOCK}" = "0" ] || [ "${LLM_MOCK}" = "0" ]; then
    echo -e "  ${YELLOW}⚠ Some mock flags are disabled${NC}"
    IS_MOCK=false
fi

if [ "$IS_MOCK" = "true" ]; then
    echo -e "\n${GREEN}${BOLD}✓ MOCK MODE IS ACTIVE${NC}"
    echo -e "${GREEN}  No real AI models will be loaded${NC}"
else
    echo -e "\n${YELLOW}${BOLD}⚠ REAL MODE MAY BE ACTIVE${NC}"
    echo -e "${YELLOW}  Real AI models may be loaded${NC}"
    echo -e "\n${BOLD}To enable Mock mode:${NC}"
    echo -e "  1. Edit .env file"
    echo -e "  2. Set: T2I_MOCK=1, VLM_MOCK=1, LLM_MOCK=1"
    echo -e "  3. Set: MODEL_DEVICE=\"cpu\""
    echo -e "  4. Restart backend"
fi

# Test a few endpoints
echo -e "\n${BOLD}Testing API Endpoints:${NC}"
test_endpoint() {
    local name="$1"
    local endpoint="$2"
    echo -n "  ${name}... "
    if curl -s -f "$API_URL$endpoint" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
}

test_endpoint "Health" "/health"
test_endpoint "Story Sessions" "/story/sessions"
test_endpoint "Story Personas" "/story/personas"
test_endpoint "RAG Stats" "/rag/stats"
test_endpoint "T2I Generations" "/t2i/generations"
test_endpoint "Agent Tools" "/agent/tools"
test_endpoint "Batch Jobs" "/batch/jobs"

echo -e "\n${BOLD}${BLUE}========================================${NC}"
echo -e "${BOLD}${BLUE}  Verification Complete${NC}"
echo -e "${BOLD}${BLUE}========================================${NC}\n"
