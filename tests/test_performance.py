# tests/test_performance.py
"""
性能和負載測試
"""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc


class TestPerformance:
    """性能測試"""

    @pytest.mark.slow
    def test_concurrent_requests(self, client):
        """測試並發請求處理"""

        def make_request():
            response = client.get("/api/v1/health")
            return response.status_code == 200

        # 並發執行50個請求
        with ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [f.result() for f in futures]
            end_time = time.time()

        # 驗證結果
        assert all(results), "Some requests failed"
        duration = end_time - start_time
        assert duration < 30, f"Requests took too long: {duration:.2f}s"

        print(f"✅ 50 concurrent requests completed in {duration:.2f}s")

    @pytest.mark.slow
    def test_memory_usage(self, client, mock_models, sample_documents):
        """測試記憶體使用"""
        import psutil
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 執行多個操作
        for i in range(10):
            # RAG操作
            payload = {
                "content": sample_documents[0]["content"] * 10,  # 擴大內容
                "metadata": sample_documents[0]["metadata"],
            }
            response = client.post("/api/v1/rag/add", json=payload)
            assert response.status_code == 200

            # 搜尋操作
            search_payload = {"query": "測試查詢", "top_k": 5}
            response = client.post("/api/v1/rag/search", json=search_payload)
            assert response.status_code == 200

        # 檢查記憶體增長
        gc.collect()  # 強制垃圾回收
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        print(
            f"📊 Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_growth:.1f}MB)"
        )

        # 記憶體增長不應過大（根據實際情況調整閾值）
        assert memory_growth < 100, f"Memory growth too high: {memory_growth:.1f}MB"

    def test_response_times(self, client, mock_models, sample_image_data):
        """測試各端點響應時間"""
        endpoints = [
            ("GET", "/api/v1/health", None, 1.0),  # 健康檢查應該很快
            (
                "POST",
                "/api/v1/chat",
                {"message": "測試", "conversation_id": "perf-test"},
                3.0,
            ),
            ("POST", "/api/v1/rag/search", {"query": "測試", "top_k": 3}, 2.0),
        ]

        results = {}

        for method, endpoint, payload, threshold in endpoints:
            start_time = time.time()

            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint, json=payload)

            duration = time.time() - start_time
            results[endpoint] = duration

            assert response.status_code == 200, f"Endpoint {endpoint} failed"
            assert (
                duration < threshold
            ), f"Endpoint {endpoint} too slow: {duration:.2f}s > {threshold}s"

            print(f"⚡ {endpoint}: {duration:.3f}s (< {threshold}s)")

        return results


class TestStressTest:
    """壓力測試"""

    @pytest.mark.slow
    def test_rag_bulk_operations(self, client, sample_documents):
        """RAG批量操作壓力測試"""

        # 批量添加文檔
        doc_ids = []
        start_time = time.time()

        for i in range(100):  # 添加100個文檔
            payload = {
                "content": f"{sample_documents[i % len(sample_documents)]['content']} - 文檔{i}",
                "metadata": {"title": f"測試文檔{i}", "batch_id": "stress_test"},
            }
            response = client.post("/api/v1/rag/add", json=payload)
            assert response.status_code == 200
            doc_ids.append(response.json()["document_id"])

        add_duration = time.time() - start_time
        print(f"📝 Added 100 documents in {add_duration:.2f}s")

        # 批量搜尋
        queries = ["人工智慧", "機器學習", "深度學習", "電腦視覺", "自然語言"]
        search_start = time.time()

        for query in queries * 20:  # 每個查詢執行20次
            search_payload = {"query": query, "top_k": 5}
            response = client.post("/api/v1/rag/search", json=search_payload)
            assert response.status_code == 200

        search_duration = time.time() - search_start
        print(f"🔍 Completed 100 searches in {search_duration:.2f}s")

        # 性能斷言
        assert add_duration < 60, f"Document addition too slow: {add_duration:.2f}s"
        assert (
            search_duration < 30
        ), f"Search operations too slow: {search_duration:.2f}s"
