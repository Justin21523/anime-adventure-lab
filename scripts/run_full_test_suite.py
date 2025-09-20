#!/usr/bin/env python3
# scripts/run_full_test_suite.py
"""
完整測試套件執行器
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time
import json


def run_command(cmd, timeout=300):
    """執行命令並返回結果"""
    print(f"🔄 Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "returncode": -1,
        }


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive backend tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument(
        "--performance", action="store_true", help="Include performance tests"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument(
        "--output-dir", default="test_results", help="Output directory for results"
    )

    args = parser.parse_args()

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🧪 Multi-Modal Lab Comprehensive Test Suite")
    print("=" * 50)

    # 測試階段配置
    test_phases = [
        {
            "name": "健康檢查",
            "command": [sys.executable, "scripts/health_check.py"],
            "required": True,
        },
        {
            "name": "代碼品質檢查",
            "command": ["ruff", "check", "backend/", "tests/"],
            "required": True,
        },
        {
            "name": "格式檢查",
            "command": ["black", "--check", "backend/", "tests/"],
            "required": True,
        },
        {
            "name": "單元測試",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_core_modules.py",
                "-v",
            ],
            "required": True,
        },
        {
            "name": "整合測試",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_api_endpoints.py",
                "-v",
            ],
            "required": True,
        },
    ]

    if not args.quick:
        test_phases.extend(
            [
                {
                    "name": "端到端測試",
                    "command": [
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_e2e_workflows.py",
                        "-v",
                    ],
                    "required": False,
                },
                {
                    "name": "可靠性測試",
                    "command": [
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_reliability.py",
                        "-v",
                    ],
                    "required": False,
                },
            ]
        )

    if args.performance:
        test_phases.append(
            {
                "name": "性能測試",
                "command": [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_performance.py",
                    "-v",
                    "-m",
                    "slow",
                ],
                "required": False,
            }
        )

    # 並行測試選項
    if args.parallel:
        for phase in test_phases:
            if "pytest" in phase["command"]:
                phase["command"].extend(["-n", "auto"])

    # 覆蓋率選項
    if args.coverage:
        for phase in test_phases:
            if "pytest" in phase["command"]:
                phase["command"].extend(
                    ["--cov=backend", "--cov-report=html", "--cov-report=term"]
                )

    # 執行測試階段
    results = {}
    total_start = time.time()

    for phase in test_phases:
        phase_start = time.time()
        print(f"\n🔍 {phase['name']}")
        print("-" * 30)

        result = run_command(phase["command"], timeout=600)
        phase_duration = time.time() - phase_start

        results[phase["name"]] = {
            "success": result["success"],
            "duration": phase_duration,
            "required": phase["required"],
            "output": result["stdout"],
            "errors": result["stderr"],
        }

        if result["success"]:
            print(f"✅ {phase['name']} 通過 ({phase_duration:.1f}s)")
        else:
            print(f"❌ {phase['name']} 失敗 ({phase_duration:.1f}s)")
            if phase["required"]:
                print(f"🚨 必要階段失敗，停止測試")
                break
            else:
                print(f"⚠️  可選階段失敗，繼續執行")

        # 輸出階段結果到文件
        phase_output_file = output_dir / f"{phase['name'].replace(' ', '_')}.log"
        with open(phase_output_file, "w", encoding="utf-8") as f:
            f.write(f"=== {phase['name']} ===\n")
            f.write(f"Command: {' '.join(phase['command'])}\n")
            f.write(f"Duration: {phase_duration:.2f}s\n")
            f.write(f"Return Code: {result['returncode']}\n")
            f.write(f"Success: {result['success']}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result["stdout"])
            f.write("\n\n=== STDERR ===\n")
            f.write(result["stderr"])

    total_duration = time.time() - total_start

    # 生成總結報告
    print(f"\n{'='*50}")
    print("📊 測試總結報告")
    print(f"{'='*50}")

    passed_count = sum(1 for r in results.values() if r["success"])
    total_count = len(results)
    required_failed = any(not r["success"] and r["required"] for r in results.values())

    print(f"總執行時間: {total_duration:.1f}s")
    print(f"測試階段: {passed_count}/{total_count} 通過")

    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        required_mark = "🔴" if result["required"] else "🔵"
        print(f"  {status} {required_mark} {name}: {result['duration']:.1f}s")

    # 保存詳細報告
    detailed_report = {
        "summary": {
            "total_duration": total_duration,
            "total_phases": total_count,
            "passed_phases": passed_count,
            "failed_phases": total_count - passed_count,
            "required_failed": required_failed,
        },
        "phases": results,
        "timestamp": time.time(),
        "config": {
            "quick": args.quick,
            "performance": args.performance,
            "coverage": args.coverage,
            "parallel": args.parallel,
        },
    }

    report_file = output_dir / "comprehensive_test_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(detailed_report, f, ensure_ascii=False, indent=2)

    print(f"\n📋 詳細報告已保存: {report_file}")

    # 最終結果
    if required_failed:
        print(f"\n❌ 測試失敗: 必要階段未通過")
        return 1
    elif passed_count == total_count:
        print(f"\n🎉 所有測試通過! 系統就緒")
        return 0
    else:
        print(f"\n⚠️  部分可選測試失敗，但系統基本功能正常")
        return 0


if __name__ == "__main__":
    main()
