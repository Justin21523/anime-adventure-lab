# tests/test_e2e_workflows.py
"""
端到端工作流測試
"""

import pytest
import json
from io import BytesIO


class TestE2EWorkflows:
    """端到端工作流測試"""

    def test_multimodal_workflow(
        self, client, mock_models, sample_image_data, sample_documents
    ):
        """測試多模態工作流：圖片→描述→RAG→聊天"""

        # Step 1: Generate caption
        files = {"image": ("test.jpg", BytesIO(sample_image_data), "image/jpeg")}
        caption_response = client.post("/api/v1/caption", files=files)
        assert caption_response.status_code == 200
        caption = caption_response.json()["caption"]

        # Step 2: Add documents to RAG
        for doc in sample_documents:
            payload = {"content": doc["content"], "metadata": doc["metadata"]}
            add_response = client.post("/api/v1/rag/add", json=payload)
            assert add_response.status_code == 200

        # Step 3: RAG question about the caption
        rag_payload = {
            "question": f"根據這個圖片描述：{caption}，請解釋相關的AI技術",
            "top_k": 2,
        }
        rag_response = client.post("/api/v1/rag/ask", json=rag_payload)
        assert rag_response.status_code == 200
        rag_answer = rag_response.json()["answer"]

        # Step 4: Continue with chat
        chat_payload = {
            "message": f"請進一步解釋：{rag_answer}",
            "conversation_id": "multimodal-test",
        }
        chat_response = client.post("/api/v1/chat", json=chat_payload)
        assert chat_response.status_code == 200

    def test_game_workflow(self, client, mock_models):
        """測試遊戲工作流：創建→多步驟→狀態持續"""

        # Create new game
        new_game_payload = {"persona": "wise_mentor", "scenario": "cyberpunk_city"}
        new_response = client.post("/api/v1/game/new", json=new_game_payload)
        assert new_response.status_code == 200
        game_id = new_response.json()["game_id"]

        # Multiple game steps
        actions = ["觀察周圍環境", "與NPC對話", "搜索線索", "做出決定"]

        for action in actions:
            step_payload = {"game_id": game_id, "action": action}
            step_response = client.post("/api/v1/game/step", json=step_payload)
            assert step_response.status_code == 200

            data = step_response.json()
            assert "scene_description" in data
            assert len(data["available_actions"]) > 0

    def test_batch_processing_workflow(self, client, mock_models, sample_documents):
        """測試批次處理工作流"""

        # Add multiple documents
        document_ids = []
        for doc in sample_documents:
            payload = {"content": doc["content"], "metadata": doc["metadata"]}
            response = client.post("/api/v1/rag/add", json=payload)
            assert response.status_code == 200
            document_ids.append(response.json()["document_id"])

        # Batch search multiple queries
        queries = ["人工智慧", "機器學習", "深度學習"]
        results = []

        for query in queries:
            search_payload = {"query": query, "top_k": 1}
            response = client.post("/api/v1/rag/search", json=search_payload)
            assert response.status_code == 200
            results.append(response.json())

        # Verify all searches returned results
        assert len(results) == len(queries)
        assert all(len(result["results"]) > 0 for result in results)
