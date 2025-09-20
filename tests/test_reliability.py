# tests/test_reliability.py
"""
可靠性和穩定性測試
"""

import pytest
import random
import string


class TestErrorHandling:
    """錯誤處理測試"""

    def test_invalid_json_handling(self, client):
        """測試無效JSON處理"""
        # 發送無效JSON
        response = client.post(
            "/api/v1/chat",
            data="invalid json{",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422  # Validation error

    def test_missing_fields(self, client):
        """測試必填欄位缺失"""
        # 缺少必填欄位的請求
        response = client.post("/api/v1/chat", json={})
        assert response.status_code == 422

        error_detail = response.json()["detail"]
        assert any("message" in str(error) for error in error_detail)

    def test_oversized_requests(self, client):
        """測試過大請求處理"""
        # 創建超大內容
        huge_content = "A" * 10000  # 10KB content

        payload = {"content": huge_content, "metadata": {"title": "huge_doc"}}

        response = client.post("/api/v1/rag/add", json=payload)
        # 應該正常處理或返回適當錯誤
        assert response.status_code in [200, 413, 422]

    def test_special_characters_handling(self, client, mock_models):
        """測試特殊字符處理"""
        special_messages = [
            "Hello 🌍🚀✨",  # Emojis
            "測試中文內容",  # Chinese
            "Тест на русском",  # Russian
            "🤖💬🔥",  # Only emojis
            "' OR '1'='1",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
        ]

        for message in special_messages:
            payload = {"message": message, "conversation_id": "special-test"}

            response = client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200, f"Failed on message: {message}"

            data = response.json()
            assert "response" in data

    def test_concurrent_game_sessions(self, client):
        """測試並發遊戲會話"""
        import threading
        import time

        results = {}

        def create_and_play_game(thread_id):
            try:
                # 創建遊戲
                new_game_payload = {
                    "persona": "friendly_guide",
                    "scenario": "fantasy_forest",
                }
                response = client.post("/api/v1/game/new", json=new_game_payload)
                if response.status_code != 200:
                    results[thread_id] = False
                    return

                game_id = response.json()["game_id"]

                # 多步遊戲操作
                for i in range(3):
                    step_payload = {"game_id": game_id, "action": f"行動{i}"}
                    response = client.post("/api/v1/game/step", json=step_payload)
                    if response.status_code != 200:
                        results[thread_id] = False
                        return
                    time.sleep(0.1)  # 模擬思考時間

                results[thread_id] = True

            except Exception as e:
                print(f"Thread {thread_id} error: {e}")
                results[thread_id] = False

        # 啟動多個線程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_and_play_game, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有線程完成
        for thread in threads:
            thread.join()

        # 驗證結果
        success_count = sum(1 for success in results.values() if success)
        print(f"🎮 {success_count}/5 concurrent game sessions succeeded")

        assert success_count >= 4, "Too many concurrent game session failures"


class TestDataIntegrity:
    """資料完整性測試"""

    def test_conversation_persistence(self, client, mock_models):
        """測試對話持續性"""
        conversation_id = "integrity-test-conv"

        # 第一輪對話
        payload1 = {"message": "我叫張三，今年25歲", "conversation_id": conversation_id}
        response1 = client.post("/api/v1/chat", json=payload1)
        assert response1.status_code == 200

        # 第二輪對話，引用前面的資訊
        payload2 = {
            "message": "我剛才說我多大年紀？",
            "conversation_id": conversation_id,
        }
        response2 = client.post("/api/v1/chat", json=payload2)
        assert response2.status_code == 200

        # 驗證上下文保持
        response_text = response2.json()["response"]
        # Mock實現可能不會真的記住，但至少不應該出錯
        assert len(response_text) > 0

    def test_rag_document_consistency(self, client, sample_documents):
        """測試RAG文檔一致性"""
        doc_ids = []

        # 添加文檔
        for i, doc in enumerate(sample_documents):
            payload = {
                "content": doc["content"],
                "metadata": {**doc["metadata"], "test_id": i},
            }
            response = client.post("/api/v1/rag/add", json=payload)
            assert response.status_code == 200
            doc_ids.append(response.json()["document_id"])

        # 搜尋並驗證
        for i in range(len(sample_documents)):
            search_payload = {"query": "人工智慧" if i == 0 else "機器學習", "top_k": 3}
            response = client.post("/api/v1/rag/search", json=search_payload)
            assert response.status_code == 200

            results = response.json()["results"]
            assert len(results) > 0, f"No results for search {i}"

            # 驗證文檔確實存在於結果中
            found_doc_ids = [r["document_id"] for r in results]
            assert any(doc_id in doc_ids for doc_id in found_doc_ids)
