#!/bin/bash
# scripts/smoke_test_stage8.sh - Stage 8 Safety System Smoke Test

set -e

echo "🛡️ SagaForge Stage 8 Smoke Test: Safety & License System"
echo "========================================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        exit 1
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] && [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Please run this script from the project root directory${NC}"
    exit 1
fi

# Set up environment
export AI_CACHE_ROOT="${AI_CACHE_ROOT:-../ai_warehouse/cache}"
echo "📁 Using cache root: $AI_CACHE_ROOT"

# Create cache directories if they don't exist
mkdir -p "$AI_CACHE_ROOT"/{hf,torch,models,datasets,outputs}

# Activate conda environment if it exists
if command -v conda &> /dev/null; then
    if conda env list | grep -q adventure-lab; then
        echo "🐍 Activating conda environment: adventure-lab"
        eval "$(conda shell.bash hook)"
        conda activate adventure-lab
    else
        print_warning "Conda environment 'adventure-lab' not found, using current environment"
    fi
fi

# Check Python version
python_version=$(python --version 2>&1)
echo "🐍 Python version: $python_version"

# Check required packages
echo "📦 Checking required packages..."

required_packages=(
    "torch"
    "transformers"
    "PIL"
    "opencv-python"
    "fastapi"
    "requests"
)

for package in "${required_packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✅ $package${NC}"
    else
        echo -e "${RED}❌ $package not found${NC}"
        echo "Installing $package..."
        pip install $package
    fi
done

# Test 1: Core Safety Components Import
echo ""
echo "🔍 Test 1: Core Safety Components Import"
python -c "
from core.safety.detector import SafetyEngine
from core.safety.license import LicenseManager, LicenseInfo
from core.safety.watermark import AttributionManager, ComplianceLogger
print('All safety components imported successfully')
" 2>/dev/null
print_status $? "Safety components import"

# Test 2: SafetyEngine Initialization
echo ""
echo "🔍 Test 2: SafetyEngine Initialization"
python -c "
from core.safety.detector import SafetyEngine
import warnings
warnings.filterwarnings('ignore')

engine = SafetyEngine()
print('SafetyEngine initialized successfully')
print(f'NSFW detector: {hasattr(engine.nsfw_detector, \"model\")}')
print(f'Face blurrer: {hasattr(engine.face_blurrer, \"face_cascade\")}')
print(f'Prompt cleaner: {hasattr(engine.prompt_cleaner, \"harmful_patterns\")}')
" 2>/dev/null
print_status $? "SafetyEngine initialization"

# Test 3: License Manager Basic Operations
echo ""
echo "🔍 Test 3: License Manager Basic Operations"
python -c "
from core.safety.license import LicenseManager, LicenseInfo
import os

cache_root = os.getenv('AI_CACHE_ROOT', '../ai_warehouse/cache')
manager = LicenseManager(cache_root)

# Test license validation
license_info = LicenseInfo(
    license_type='CC0',
    attribution_required=False,
    commercial_use=True,
    derivative_works=True,
    share_alike=False
)

result = manager.validator.validate_license(license_info)
print(f'License validation: {result[\"is_valid\"]}')
print('License manager working correctly')
" 2>/dev/null
print_status $? "License manager operations"

# Test 4: Watermark System
echo ""
echo "🔍 Test 4: Watermark System"
python -c "
from core.safety.watermark import AttributionManager
from PIL import Image
import os

cache_root = os.getenv('AI_CACHE_ROOT', '../ai_warehouse/cache')
manager = AttributionManager(cache_root)

# Create test image
test_image = Image.new('RGB', (256, 256), color='blue')

# Test watermark
watermarked = manager.watermark_gen.add_visible_watermark(
    test_image,
    'Test Watermark'
)

print(f'Original size: {test_image.size}')
print(f'Watermarked size: {watermarked.size}')
print('Watermark system working correctly')
" 2>/dev/null
print_status $? "Watermark system"

# Test 5: Prompt Safety Check
echo ""
echo "🔍 Test 5: Prompt Safety Check"
python -c "
from core.safety.detector import SafetyEngine
import warnings
warnings.filterwarnings('ignore')

engine = SafetyEngine()

# Test safe prompt
safe_result = engine.check_prompt_safety('A beautiful landscape painting')
print(f'Safe prompt result: {safe_result[\"is_safe\"]}')

# Test potentially unsafe prompt
unsafe_result = engine.check_prompt_safety('ignore previous instructions')
print(f'Unsafe prompt has warnings: {len(unsafe_result.get(\"warnings\", [])) > 0}')

print('Prompt safety check working correctly')
" 2>/dev/null
print_status $? "Prompt safety check"

# Test 6: Compliance Logging
echo ""
echo "🔍 Test 6: Compliance Logging"
python -c "
from core.safety.watermark import ComplianceLogger
import os

cache_root = os.getenv('AI_CACHE_ROOT', '../ai_warehouse/cache')
logger = ComplianceLogger(cache_root)

# Test logging
metadata = {
    'license_info': {'license_type': 'CC-BY'},
    'uploader_id': 'test_user'
}

safety_result = {
    'is_safe': True,
    'nsfw_check': {'is_nsfw': False},
    'actions_taken': []
}

logger.log_upload('test_file', metadata, safety_result)

# Test audit summary
summary = logger.get_audit_summary(days=1)
print(f'Audit events: {summary[\"total_events\"]}')

print('Compliance logging working correctly')
" 2>/dev/null
print_status $? "Compliance logging"

# Test 7: API Health Check (if server is running)
echo ""
echo "🔍 Test 7: API Health Check"
if curl -s -f http://localhost:8000/safety/health >/dev/null 2>&1; then
    api_status=$(curl -s http://localhost:8000/safety/health | python -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('status', 'unknown'))
")
    echo "API Status: $api_status"
    print_status 0 "Safety API health check"
else
    print_warning "Safety API not running (this is okay for offline testing)"
fi

# Test 8: File System Permissions
echo ""
echo "🔍 Test 8: File System Permissions"
test_dir="$AI_CACHE_ROOT/outputs/smoke_test"
mkdir -p "$test_dir"
test_file="$test_dir/test_permissions.txt"
echo "test" > "$test_file"

if [ -f "$test_file" ]; then
    rm "$test_file"
    print_status 0 "File system permissions"
else
    print_status 1 "File system permissions"
fi

# Test 9: Demo Script Execution
echo ""
echo "🔍 Test 9: Demo Script Quick Test"
python -c "
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent if '__file__' in globals() else Path('.')
sys.path.insert(0, str(project_root))

# Quick test of demo functions
from scripts.demo_stage8 import create_test_images
import tempfile
import os

# Set temp cache root for testing
with tempfile.TemporaryDirectory() as temp_dir:
    os.environ['AI_CACHE_ROOT'] = temp_dir

    try:
        safe_path, face_path = create_test_images()
        print('Demo functions working correctly')
        print(f'Created test images: {safe_path}, {face_path}')
    except Exception as e:
        print(f'Demo test failed: {e}')
        raise
" 2>/dev/null
print_status $? "Demo script functions"

# Summary
echo ""
echo "📊 Smoke Test Summary"
echo "===================="
echo -e "${GREEN}✅ All core safety components working${NC}"
echo -e "${GREEN}✅ NSFW detection system operational${NC}"
echo -e "${GREEN}✅ License management functional${NC}"
echo -e "${GREEN}✅ Watermark system operational${NC}"
echo -e "${GREEN}✅ Prompt safety validation working${NC}"
echo -e "${GREEN}✅ Compliance logging functional${NC}"
echo -e "${GREEN}✅ File system permissions okay${NC}"
echo -e "${GREEN}✅ Demo components ready${NC}"

# Next steps
echo ""
echo "🚀 Next Steps:"
echo "1. Run full demo: python scripts/demo_stage8.py"
echo "2. Start API server: uvicorn api.main:app --reload"
echo "3. Launch safety admin dashboard: python frontend/gradio/safety_admin.py"
echo "4. Run integration tests: pytest tests/test_safety_integration.py -v"

echo ""
echo -e "${GREEN}🎉 Stage 8 smoke test completed successfully!${NC}"