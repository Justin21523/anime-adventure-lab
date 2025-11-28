# tests/test_integration_end_to_end.py
"""
完整的端到端整合測試場景
"""

import io
import pytest


class TestCompleteWorkflows:
    """完整工作流測試"""

    @pytest.mark.e2e
    def test_full_multimodal_pipeline(
        self, client, mock_models, sample_image_data, sample_documents
    ):
        """完整多模態流水線測試"""

        # Step 1: 系統健康檢查
        health_response = client.get("/api/v1/health")
        assert health_response.status_code == 200

        # Step 2: 圖片上傳和描述
        files = {"image": ("test.jpg", io.BytesIO(sample_image_data), "image/jpeg")}
        caption_response = client.post("/api/v1/caption", files=files)
        assert caption_response.status_code == 200
        caption = caption_response.json()["caption"]
        print(f"📷 Image caption: {caption}")

        # Step 3: 視覺問答
        files = {"image": ("test.jpg", io.BytesIO(sample_image_data), "image/jpeg")}
        vqa_response = client.post(
            "/api/v1/vqa", files=files, data={"question": "這張圖片裡有什麼顏色？"}
        )
        assert vqa_response.status_code == 200
        vqa_answer = vqa_response.json()["answer"]
        print(f"❓ VQA answer: {vqa_answer}")

        # Step 4: 建立知識庫
        doc_ids = []
        for doc in sample_documents:
            payload = {"content": doc["content"], "metadata": doc["metadata"]}
            response = client.post("/api/v1/rag/add", json=payload)
            assert response.status_code == 200
            doc_ids.append(response.json()["document_id"])

        print(f"📚 Added {len(doc_ids)} documents to knowledge base")

        # Step 5: RAG增強的對話
        conversation_id = "e2e-multimodal-conv"

        # 基於圖片描述提問
        chat_payload = {
            "message": f"根據這個圖片描述「{caption}」和VQA回答「{vqa_answer}」，請解釋相關的AI技術原理",
            "conversation_id": conversation_id,
        }
        chat_response = client.post("/api/v1/chat", json=chat_payload)
        assert chat_response.status_code == 200
        chat_answer = chat_response.json()["response"]
        print(f"💬 Chat response: {chat_answer[:100]}...")

        # Step 6: RAG增強查詢
        rag_payload = {"question": "請詳細解釋電腦視覺技術在AI中的應用", "top_k": 3}
        rag_response = client.post("/api/v1/rag/ask", json=rag_payload)
        assert rag_response.status_code == 200
        rag_answer = rag_response.json()["answer"]
        print(f"🔍 RAG answer: {rag_answer[:100]}...")

        # Step 7: Agent工具調用
        agent_payload = {
            "tool_name": "calculator",
            "parameters": {"expression": "100 * 0.85"},  # 計算準確率
        }
        agent_response = client.post("/api/v1/agent/call", json=agent_payload)
        assert agent_response.status_code == 200
        calculation_result = agent_response.json()["result"]["value"]
        print(f"🧮 Agent calculation: {calculation_result}")

        # Step 8: 遊戲互動
        game_payload = {"persona": "wise_mentor", "scenario": "cyberpunk_city"}
        game_response = client.post("/api/v1/game/new", json=game_payload)
        assert game_response.status_code == 200
        game_id = game_response.json()["game_id"]

        # 遊戲步驟
        step_payload = {"game_id": game_id, "action": "探索這個充滿AI技術的城市"}
        step_response = client.post("/api/v1/game/step", json=step_payload)
        assert step_response.status_code == 200
        game_scene = step_response.json()["scene_description"]
        print(f"🎮 Game scene: {game_scene[:100]}...")

        # Step 9: 系統狀態檢查
        status_response = client.get("/api/v1/status")
        assert status_response.status_code == 200
        system_status = status_response.json()
        print(f"⚡ System status: {system_status.get('models', {}).keys()}")

        print("✅ Complete multimodal pipeline test passed!")

    @pytest.mark.e2e
    def test_batch_processing_workflow(self, client, mock_models, sample_documents):
        """批次處理工作流測試"""

        # 批量文檔處理
        batch_docs = sample_documents * 10  # 擴展到40個文檔
        processed_docs = []

        print(f"📦 Starting batch processing of {len(batch_docs)} documents...")

        # 分批處理（模擬真實批次場景）
        batch_size = 5
        for i in range(0, len(batch_docs), batch_size):
            batch = batch_docs[i : i + batch_size]
            batch_results = []

            for j, doc in enumerate(batch):
                payload = {
                    "content": f"{doc['content']} - Batch {i//batch_size + 1}, Doc {j+1}",
                    "metadata": {
                        **doc["metadata"],
                        "batch_id": f"batch_{i//batch_size + 1}",
                        "doc_index": i + j,
                    },
                }

                response = client.post("/api/v1/rag/add", json=payload)
                assert response.status_code == 200
                batch_results.append(response.json()["document_id"])

            processed_docs.extend(batch_results)
            print(
                f"  ✅ Processed batch {i//batch_size + 1}: {len(batch_results)} docs"
            )

        # 批量搜尋測試
        search_queries = [
            "人工智慧的發展",
            "機器學習算法",
            "深度學習應用",
            "電腦視覺技術",
            "自然語言處理",
        ]

        search_results = {}
        for query in search_queries:
            search_payload = {"query": query, "top_k": 5}
            response = client.post("/api/v1/rag/search", json=search_payload)
            assert response.status_code == 200

            results = response.json()["results"]
            search_results[query] = len(results)
            print(f"  🔍 Query '{query}': {len(results)} results")

        # 驗證批次處理結果
        assert len(processed_docs) == len(batch_docs)
        assert all(count > 0 for count in search_results.values())

        print(
            f"✅ Batch processing workflow completed: {len(processed_docs)} docs processed"
        )

    @pytest.mark.e2e
    def test_user_session_lifecycle(self, client, mock_models):
        """用戶會話生命週期測試"""

        # 模擬完整用戶會話
        session_id = f"user_session_{int(time.time())}"

        # 1. 用戶登入（簡化版）
        print(f"👤 Starting user session: {session_id}")

        # 2. 多輪對話
        conversation_topics = [
            "你好，我想了解AI技術",
            "什麼是機器學習？",
            "深度學習和機器學習有什麼區別？",
            "AI在日常生活中有哪些應用？",
            "未來AI會如何發展？",
        ]

        conversation_id = f"conv_{session_id}"
        for i, topic in enumerate(conversation_topics):
            payload = {"message": topic, "conversation_id": conversation_id}

            response = client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200

            answer = response.json()["response"]
            print(f"  💭 Round {i+1}: {topic[:30]}... -> {answer[:50]}...")

        # 3. 創建個人知識庫
        personal_docs = [
            {
                "content": "我的AI學習筆記：今天學習了監督學習的基本概念...",
                "metadata": {
                    "type": "personal_note",
                    "session": session_id,
                    "topic": "supervised_learning",
                },
            },
            {
                "content": "實踐項目記錄：使用Python實現了一個簡單的線性回歸模型...",
                "metadata": {
                    "type": "project_log",
                    "session": session_id,
                    "topic": "linear_regression",
                },
            },
        ]

        for doc in personal_docs:
            payload = {"content": doc["content"], "metadata": doc["metadata"]}
            response = client.post("/api/v1/rag/add", json=payload)
            assert response.status_code == 200

        # 4. 個人化查詢
        personal_query = {"question": "我之前學過什麼AI概念？", "top_k": 5}
        response = client.post("/api/v1/rag/ask", json=personal_query)
        assert response.status_code == 200
        personal_answer = response.json()["answer"]
        print(f"  🔍 Personal query result: {personal_answer[:100]}...")

        # 5. 遊戲娛樂
        game_payload = {"persona": "friendly_guide", "scenario": "fantasy_forest"}
        response = client.post("/api/v1/game/new", json=game_payload)
        assert response.status_code == 200
        game_id = response.json()["game_id"]

        # 進行幾步遊戲
        game_actions = ["觀察周圍環境", "與導師對話", "探索新區域"]
        for action in game_actions:
            step_payload = {"game_id": game_id, "action": action}
            response = client.post("/api/v1/game/step", json=step_payload)
            assert response.status_code == 200

        print(f"  🎮 Completed game session with {len(game_actions)} actions")

        # 6. 會話總結
        summary_payload = {
            "message": "請總結我們今天的對話內容",
            "conversation_id": conversation_id,
        }
        response = client.post("/api/v1/chat", json=summary_payload)
        assert response.status_code == 200

        print(f"✅ User session lifecycle test completed for {session_id}")
