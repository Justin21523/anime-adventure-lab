# tests/test_api_endpoints.py
"""
API 端點整合測試
"""

import pytest
import json
from io import BytesIO


class TestHealthEndpoints:
    """測試健康檢查端點"""

    def test_health_check(self, client):
        """測試健康檢查"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_status_endpoint(self, client):
        """測試狀態端點"""
        response = client.get("/api/v1/status")
        assert response.status_code == 200

        data = response.json()
        assert "system_info" in data
        assert "models" in data


class TestCaptionEndpoints:
    """測試圖片描述端點"""

    def test_caption_endpoint(self, client, mock_models, sample_image_data):
        """測試圖片描述生成"""
        files = {"image": ("test.jpg", BytesIO(sample_image_data), "image/jpeg")}

        response = client.post("/api/v1/caption", files=files)
        assert response.status_code == 200

        data = response.json()
        assert "caption" in data
        assert "confidence" in data

    def test_caption_no_image(self, client):
        """測試無圖片的錯誤處理"""
        response = client.post("/api/v1/caption")
        assert response.status_code == 422  # Validation error


class TestVQAEndpoints:
    """測試視覺問答端點"""

    def test_vqa_endpoint(self, client, mock_models, sample_image_data):
        """測試視覺問答"""
        files = {"image": ("test.jpg", BytesIO(sample_image_data), "image/jpeg")}
        data = {"question": "這張圖片裡有什麼？"}

        response = client.post("/api/v1/vqa", files=files, data=data)
        assert response.status_code == 200

        result = response.json()
        assert "answer" in result
        assert "confidence" in result


class TestChatEndpoints:
    """測試聊天端點"""

    def test_chat_endpoint(self, client, mock_models):
        """測試聊天對話"""
        payload = {
            "message": "你好，請介紹一下人工智慧",
            "conversation_id": "test-conv-1",
        }

        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "response" in data
        assert "conversation_id" in data

    def test_chat_history(self, client, mock_models):
        """測試對話歷史"""
        # First message
        payload1 = {"message": "我的名字是張三", "conversation_id": "test-conv-2"}
        response1 = client.post("/api/v1/chat", json=payload1)
        assert response1.status_code == 200

        # Second message referencing history
        payload2 = {
            "message": "我剛才說我叫什麼名字？",
            "conversation_id": "test-conv-2",
        }
        response2 = client.post("/api/v1/chat", json=payload2)
        assert response2.status_code == 200


class TestRAGEndpoints:
    """測試 RAG 端點"""

    def test_rag_add_document(self, client, sample_documents):
        """測試文檔添加"""
        payload = {
            "content": sample_documents[0]["content"],
            "metadata": sample_documents[0]["metadata"],
        }

        response = client.post("/api/v1/rag/add", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "document_id" in data
        assert data["success"] is True

    def test_rag_search(self, client, sample_documents):
        """測試 RAG 搜尋"""
        # First add some documents
        for doc in sample_documents:
            payload = {"content": doc["content"], "metadata": doc["metadata"]}
            client.post("/api/v1/rag/add", json=payload)

        # Then search
        search_payload = {"query": "人工智慧是什麼", "top_k": 2}

        response = client.post("/api/v1/rag/search", json=search_payload)
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 2

    def test_rag_ask(self, client, mock_models, sample_documents):
        """測試 RAG 問答"""
        # Add documents first
        for doc in sample_documents:
            payload = {"content": doc["content"], "metadata": doc["metadata"]}
            client.post("/api/v1/rag/add", json=payload)

        # Ask question
        ask_payload = {"question": "什麼是機器學習？", "top_k": 3}

        response = client.post("/api/v1/rag/ask", json=ask_payload)
        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "sources" in data


class TestAgentEndpoints:
    """測試 Agent 端點"""

    def test_agent_call_tool(self, client):
        """測試工具調用"""
        payload = {"tool_name": "calculator", "parameters": {"expression": "2 + 3"}}

        response = client.post("/api/v1/agent/call", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "result" in data
        assert data["success"] is True

    def test_agent_task(self, client):
        """測試複合任務"""
        payload = {"task_description": "計算今天的日期加上30天", "parameters": {}}

        response = client.post("/api/v1/agent/task", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "result" in data
        assert "tools_used" in data


class TestGameEndpoints:
    """測試遊戲端點"""

    def test_game_new(self, client):
        """測試新遊戲創建"""
        payload = {"persona": "friendly_guide", "scenario": "fantasy_adventure"}

        response = client.post("/api/v1/game/new", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "game_id" in data
        assert "initial_scene" in data

    def test_game_step(self, client):
        """測試遊戲步驟"""
        # Create game first
        new_game_payload = {
            "persona": "friendly_guide",
            "scenario": "fantasy_adventure",
        }
        new_response = client.post("/api/v1/game/new", json=new_game_payload)
        game_id = new_response.json()["game_id"]

        # Take action
        step_payload = {"game_id": game_id, "action": "探索森林"}

        response = client.post("/api/v1/game/step", json=step_payload)
        assert response.status_code == 200

        data = response.json()
        assert "scene_description" in data
        assert "available_actions" in data


class TestAdminEndpoints:
    """測試管理端點"""

    def test_admin_system_info(self, client, mock_models):
        """測試系統資訊"""
        response = client.get("/api/v1/admin/system")
        assert response.status_code == 200

        data = response.json()
        assert "cache_stats" in data
        assert "loaded_models" in data
        assert "system_resources" in data

    def test_admin_model_control(self, client, mock_models):
        """測試模型控制"""
        payload = {"action": "unload_all"}

        response = client.post("/api/v1/admin/models/control", json=payload)
        assert response.status_code == 200
