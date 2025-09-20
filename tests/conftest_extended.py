# tests/conftest_extended.py
"""
擴展的測試配置
"""

import pytest
import asyncio
from pathlib import Path
import json
import tempfile
from datetime import datetime


@pytest.fixture(scope="session")
def test_results_dir():
    """創建測試結果目錄"""
    results_dir = Path("test_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


@pytest.fixture(scope="session")
def performance_metrics():
    """性能指標收集"""
    metrics = {
        "start_time": time.time(),
        "requests": [],
        "errors": [],
        "memory_usage": [],
    }
    return metrics


@pytest.fixture(autouse=True)
def collect_metrics(request, performance_metrics):
    """自動收集測試指標"""
    start_time = time.time()

    yield

    end_time = time.time()
    duration = end_time - start_time

    # 記錄測試執行時間
    performance_metrics["requests"].append(
        {
            "test_name": request.node.name,
            "duration": duration,
            "status": (
                "passed"
                if not hasattr(request.node, "rep_call") or request.node.rep_call.passed
                else "failed"
            ),
        }
    )


@pytest.fixture(scope="session", autouse=True)
def generate_test_report(test_results_dir, performance_metrics):
    """生成測試報告"""
    yield

    # 測試結束後生成報告
    end_time = time.time()
    total_duration = end_time - performance_metrics["start_time"]

    report = {
        "summary": {
            "total_duration": total_duration,
            "total_tests": len(performance_metrics["requests"]),
            "passed_tests": len(
                [r for r in performance_metrics["requests"] if r["status"] == "passed"]
            ),
            "failed_tests": len(
                [r for r in performance_metrics["requests"] if r["status"] == "failed"]
            ),
            "average_duration": (
                sum(r["duration"] for r in performance_metrics["requests"])
                / len(performance_metrics["requests"])
                if performance_metrics["requests"]
                else 0
            ),
        },
        "details": performance_metrics["requests"],
        "errors": performance_metrics["errors"],
        "timestamp": datetime.now().isoformat(),
    }

    # 保存報告
    report_file = test_results_dir / "test_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Test Report Generated: {report_file}")
    print(f"   Total Tests: {report['summary']['total_tests']}")
    print(f"   Passed: {report['summary']['passed_tests']}")
    print(f"   Failed: {report['summary']['failed_tests']}")
    print(f"   Duration: {report['summary']['total_duration']:.2f}s")
