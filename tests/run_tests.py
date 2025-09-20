# tests/run_tests.py
"""
測試執行腳本
"""

import pytest
import sys
from pathlib import Path


def run_all_tests():
    """執行所有測試"""
    test_dir = Path(__file__).parent

    # Basic unit tests
    print("🧪 Running Core Module Tests...")
    result1 = pytest.main(["-v", str(test_dir / "test_core_modules.py")])

    # API integration tests
    print("🔗 Running API Integration Tests...")
    result2 = pytest.main(["-v", str(test_dir / "test_api_endpoints.py")])

    # E2E workflow tests
    print("🎯 Running E2E Workflow Tests...")
    result3 = pytest.main(["-v", str(test_dir / "test_e2e_workflows.py")])

    # Summary
    total_result = result1 + result2 + result3
    if total_result == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {total_result} test(s) failed")
        sys.exit(1)


def run_smoke_tests():
    """執行快速煙霧測試"""
    test_dir = Path(__file__).parent

    print("💨 Running Smoke Tests...")
    result = pytest.main(
        [
            "-v",
            "-k",
            "test_health_check or test_caption_endpoint or test_rag_add_document",
            str(test_dir),
        ]
    )

    if result == 0:
        print("✅ Smoke tests passed!")
    else:
        print("❌ Smoke tests failed")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run backend tests")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")

    args = parser.parse_args()

    if args.coverage:
        # Install: pip install pytest-cov
        pytest.main(["--cov=backend", "--cov-report=html", "--cov-report=term"])
    elif args.smoke:
        run_smoke_tests()
    else:
        run_all_tests()
