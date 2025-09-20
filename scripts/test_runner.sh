# scripts/test_runner.sh
#!/bin/bash
# 測試執行腳本

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_DIR="tests"
BACKEND_DIR="backend"
CACHE_DIR="${AI_CACHE_ROOT:-/tmp/test_cache}"

echo -e "${BLUE}🧪 Multi-Modal Lab Backend Test Runner${NC}"
echo "=================================================="

# Create test cache if not exists
if [ ! -d "$CACHE_DIR" ]; then
    echo -e "${YELLOW}📁 Creating test cache directory: $CACHE_DIR${NC}"
    mkdir -p "$CACHE_DIR"/{hf,torch,models,datasets,outputs}
fi

# Export test environment
export AI_CACHE_ROOT="$CACHE_DIR"
export PYTEST_CURRENT_TEST=""

# Function to run tests with timeout
run_tests() {
    local test_type="$1"
    local test_files="$2"
    local timeout_minutes="${3:-10}"

    echo -e "\n${BLUE}🔄 Running $test_type...${NC}"

    if timeout "${timeout_minutes}m" python -m pytest $test_files -v --tb=short; then
        echo -e "${GREEN}✅ $test_type passed!${NC}"
        return 0
    else
        echo -e "${RED}❌ $test_type failed!${NC}"
        return 1
    fi
}

# Main test execution
main() {
    local test_mode="${1:-all}"
    local total_failures=0

    case "$test_mode" in
        "smoke")
            echo -e "${YELLOW}💨 Running smoke tests only${NC}"
            run_tests "Smoke Tests" "-k 'test_health_check or test_caption_endpoint or test_rag_add_document'" 5
            total_failures=$?
            ;;
        "unit")
            echo -e "${YELLOW}🧪 Running unit tests only${NC}"
            run_tests "Unit Tests" "$TEST_DIR/test_core_modules.py" 8
            total_failures=$?
            ;;
        "integration")
            echo -e "${YELLOW}🔗 Running integration tests only${NC}"
            run_tests "Integration Tests" "$TEST_DIR/test_api_endpoints.py" 15
            total_failures=$?
            ;;
        "e2e")
            echo -e "${YELLOW}🎯 Running E2E tests only${NC}"
            run_tests "E2E Tests" "$TEST_DIR/test_e2e_workflows.py" 20
            total_failures=$?
            ;;
        "all"|*)
            echo -e "${YELLOW}🎯 Running all tests${NC}"

            # Run in sequence with proper error handling
            run_tests "Unit Tests" "$TEST_DIR/test_core_modules.py" 8
            unit_result=$?

            run_tests "Integration Tests" "$TEST_DIR/test_api_endpoints.py" 15
            integration_result=$?

            run_tests "E2E Tests" "$TEST_DIR/test_e2e_workflows.py" 20
            e2e_result=$?

            total_failures=$((unit_result + integration_result + e2e_result))
            ;;
    esac

    # Final summary
    echo -e "\n=================================================="
    if [ $total_failures -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests completed successfully!${NC}"
        echo -e "${GREEN}✅ Backend system is ready for deployment${NC}"
    else
        echo -e "${RED}💥 $total_failures test suite(s) failed${NC}"
        echo -e "${RED}❌ Please fix issues before proceeding${NC}"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up test artifacts...${NC}"
    # Clean test cache if it's in tmp
    if [[ "$CACHE_DIR" == "/tmp/"* ]]; then
        rm -rf "$CACHE_DIR"
        echo -e "${GREEN}✅ Test cache cleaned${NC}"
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Run with arguments
main "$@"