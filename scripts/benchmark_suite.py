#!/usr/bin/env python3
# scripts/benchmark_suite.py
"""
性能基準測試套件
"""

import time
import statistics
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import sys
import requests

# Add backend path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from tests.mocks.sample_data import get_sample_data


class BenchmarkSuite:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        self.session = requests.Session()

    def time_function(self, func, *args, iterations=5, **kwargs):
        """測量函數執行時間"""
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "iterations": iterations,
            "raw_times": times,
        }

    def benchmark_api_endpoint(self, endpoint, method="GET", data=None, files=None):
        """基準測試API端點"""

        def make_request():
            url = f"{self.base_url}{endpoint}"

            if method == "GET":
                response = self.session.get(url)
            elif method == "POST":
                if files:
                    response = self.session.post(url, data=data, files=files)
                else:
                    response = self.session.post(url, json=data)

            response.raise_for_status()
            return response.json()

        return self.time_function(make_request, iterations=10)

    def benchmark_concurrent_requests(self, endpoint, num_requests=50, max_workers=10):
        """並發請求基準測試"""

        def make_request():
            response = self.session.get(f"{self.base_url}{endpoint}")
            return response.status_code == 200

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]

        end_time = time.perf_counter()

        success_rate = sum(results) / len(results)
        total_time = end_time - start_time
        throughput = num_requests / total_time

        return {
            "total_time": total_time,
            "success_rate": success_rate,
            "throughput": throughput,
            "requests_per_second": throughput,
        }

    def benchmark_memory_usage(self):
        """記憶體使用基準測試"""
        process = psutil.Process()

        initial_memory = process.memory_info()

        # 執行一些操作
        sample_docs = get_sample_data("documents")

        for i in range(20):
            # 模擬RAG操作
            self.benchmark_api_endpoint(
                "/api/v1/rag/add",
                method="POST",
                data={
                    "content": sample_docs[i % len(sample_docs)]["content"],
                    "metadata": sample_docs[i % len(sample_docs)]["metadata"],
                },
            )

        final_memory = process.memory_info()

        return {
            "initial_rss": initial_memory.rss / 1024 / 1024,  # MB
            "final_rss": final_memory.rss / 1024 / 1024,  # MB
            "memory_growth": (final_memory.rss - initial_memory.rss) / 1024 / 1024,
            "initial_vms": initial_memory.vms / 1024 / 1024,
            "final_vms": final_memory.vms / 1024 / 1024,
        }

    def run_full_benchmark(self):
        """執行完整基準測試"""
        print("🚀 Starting Multi-Modal Lab Benchmark Suite")
        print("=" * 50)

        # 健康檢查
        print("📋 Health Check...")
        health_result = self.benchmark_api_endpoint("/api/v1/health")
        self.results["health_check"] = health_result
        print(f"   平均響應時間: {health_result['mean']:.3f}s")

        # 狀態端點
        print("📊 Status Endpoint...")
        status_result = self.benchmark_api_endpoint("/api/v1/status")
        self.results["status_check"] = status_result
        print(f"   平均響應時間: {status_result['mean']:.3f}s")

        # 聊天端點
        print("💬 Chat Endpoint...")
        chat_result = self.benchmark_api_endpoint(
            "/api/v1/chat",
            method="POST",
            data={"message": "測試訊息", "conversation_id": "bench-test"},
        )
        self.results["chat"] = chat_result
        print(f"   平均響應時間: {chat_result['mean']:.3f}s")

        # RAG操作
        print("🔍 RAG Operations...")

        # 添加文檔
        sample_docs = get_sample_data("documents")
        add_doc_result = self.benchmark_api_endpoint(
            "/api/v1/rag/add",
            method="POST",
            data={
                "content": sample_docs[0]["content"],
                "metadata": sample_docs[0]["metadata"],
            },
        )
        self.results["rag_add"] = add_doc_result
        print(f"   文檔添加平均時間: {add_doc_result['mean']:.3f}s")

        # 文檔搜尋
        search_result = self.benchmark_api_endpoint(
            "/api/v1/rag/search", method="POST", data={"query": "人工智慧", "top_k": 5}
        )
        self.results["rag_search"] = search_result
        print(f"   搜尋平均時間: {search_result['mean']:.3f}s")

        # 並發測試
        print("⚡ Concurrent Load Test...")
        concurrent_result = self.benchmark_concurrent_requests(
            "/api/v1/health", num_requests=100
        )
        self.results["concurrent_load"] = concurrent_result
        print(f"   吞吐量: {concurrent_result['throughput']:.1f} req/s")
        print(f"   成功率: {concurrent_result['success_rate']:.1%}")

        # 記憶體使用測試
        print("🧠 Memory Usage Test...")
        memory_result = self.benchmark_memory_usage()
        self.results["memory_usage"] = memory_result
        print(f"   記憶體增長: {memory_result['memory_growth']:.1f}MB")

        return self.results

    def generate_report(self, output_file="benchmark_report.json"):
        """生成基準測試報告"""
        report = {
            "timestamp": time.time(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total
                / 1024
                / 1024
                / 1024,  # GB
                "python_version": sys.version,
            },
            "results": self.results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n📈 基準測試報告已保存: {output_file}")

        # 生成摘要
        print("\n📋 基準測試摘要:")
        print("-" * 30)

        if "health_check" in self.results:
            print(f"健康檢查: {self.results['health_check']['mean']:.3f}s")

        if "chat" in self.results:
            print(f"聊天響應: {self.results['chat']['mean']:.3f}s")

        if "rag_search" in self.results:
            print(f"RAG搜尋: {self.results['rag_search']['mean']:.3f}s")

        if "concurrent_load" in self.results:
            print(
                f"並發吞吐: {self.results['concurrent_load']['throughput']:.1f} req/s"
            )

        if "memory_usage" in self.results:
            print(f"記憶體增長: {self.results['memory_usage']['memory_growth']:.1f}MB")

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Modal Lab Benchmark Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="benchmark_report.json", help="Output file")
    parser.add_argument(
        "--quick", action="store_true", help="Quick benchmark (fewer iterations)"
    )

    args = parser.parse_args()

    # 檢查API可用性
    try:
        response = requests.get(f"{args.url}/api/v1/health", timeout=5)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ API不可用: {args.url}")
        print(f"   錯誤: {e}")
        print("   請確保API服務正在運行")
        return 1

    # 運行基準測試
    benchmark = BenchmarkSuite(args.url)

    try:
        results = benchmark.run_full_benchmark()
        report = benchmark.generate_report(args.output)

        print(f"\n🎉 基準測試完成!")
        return 0

    except Exception as e:
        print(f"\n❌ 基準測試失敗: {e}")
        return 1


if __name__ == "__main__":
    main()
