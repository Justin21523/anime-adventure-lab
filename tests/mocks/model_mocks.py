# tests/mocks/model_mocks.py
"""
模型相關的 Mock 類別
"""

from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional
import torch
from PIL import Image
import io
import base64


class MockLLMAdapter:
    """模擬 LLM 適配器"""

    def __init__(self):
        self.loaded_models = ["mock-qwen-7b", "mock-llama-7b"]
        self.current_model = "mock-qwen-7b"

    def generate(self, prompt: str, **kwargs) -> str:
        """模擬文字生成"""
        # 簡單的規則回應
        prompt_lower = prompt.lower()

        if "人工智慧" in prompt or "ai" in prompt_lower:
            return "人工智慧是模擬人類智能的技術，包含機器學習、深度學習等領域。"
        elif "機器學習" in prompt or "machine learning" in prompt_lower:
            return "機器學習讓電腦能夠從資料中學習，無需明確程式化。"
        elif "你好" in prompt or "hello" in prompt_lower:
            return "你好！我是AI助手，很高興為您服務。"
        elif "計算" in prompt or "calculate" in prompt_lower:
            return "這需要進行數學計算，讓我使用計算工具來幫您。"
        else:
            return f"根據您的問題「{prompt[:50]}...」，這是一個模擬回應。"

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """模擬對話"""
        last_message = messages[-1]["content"] if messages else ""
        return self.generate(last_message, **kwargs)

    def list_loaded_models(self) -> List[str]:
        """列出已載入模型"""
        return self.loaded_models

    def load_model(self, model_name: str) -> bool:
        """模擬載入模型"""
        self.current_model = model_name
        return True

    def unload_all(self) -> bool:
        """模擬卸載所有模型"""
        self.loaded_models = []
        return True


class MockVLMEngine:
    """模擬視覺語言模型引擎"""

    def __init__(self):
        self.loaded_models = ["mock-llava-7b", "mock-blip2"]
        self.current_model = "mock-llava-7b"

    def caption(self, image: Any, **kwargs) -> str:
        """模擬圖片描述"""
        # 根據圖片特徵返回不同描述
        if hasattr(image, "size"):
            w, h = image.size
            if w == h:
                return "一個正方形的彩色圖片，包含幾何圖形。"
            else:
                return "一張長方形的圖片，顯示某個場景。"
        return "一張圖片，包含各種視覺元素。"

    def vqa(self, image: Any, question: str, **kwargs) -> str:
        """模擬視覺問答"""
        question_lower = question.lower()

        if "顏色" in question or "color" in question_lower:
            return "圖片中主要包含紅色和藍色。"
        elif "什麼" in question or "what" in question_lower:
            return "圖片中顯示了一些幾何圖形和色彩。"
        elif "多少" in question or "how many" in question_lower:
            return "圖片中大約有2-3個主要元素。"
        else:
            return f"根據圖片內容，關於「{question}」的答案是：這是一個模擬回應。"

    def get_status(self) -> Dict[str, Any]:
        """獲取引擎狀態"""
        return {
            "loaded": self.loaded_models,
            "current": self.current_model,
            "memory_usage": "2.1GB",
        }

    def unload_models(self) -> bool:
        """卸載模型"""
        self.loaded_models = []
        return True


class MockRAGEngine:
    """模擬 RAG 引擎"""

    def __init__(self):
        self.documents = []
        self.doc_counter = 0

    def add_document(self, content: str, metadata: Dict) -> str:
        """添加文檔"""
        self.doc_counter += 1
        doc_id = f"doc_{self.doc_counter}"

        self.documents.append(
            {
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "embedding": [0.1] * 768,  # Mock embedding
            }
        )

        return doc_id

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜尋文檔"""
        # 簡單的關鍵字匹配
        query_lower = query.lower()
        results = []

        for doc in self.documents:
            content_lower = doc["content"].lower()

            # 計算簡單相似度分數
            score = 0.5  # 基礎分數
            for word in query_lower.split():
                if word in content_lower:
                    score += 0.2

            if score > 0.6:  # 閾值過濾
                results.append(
                    {
                        "document_id": doc["id"],
                        "content": (
                            doc["content"][:200] + "..."
                            if len(doc["content"]) > 200
                            else doc["content"]
                        ),
                        "metadata": doc["metadata"],
                        "score": min(score, 1.0),
                    }
                )

        # 按分數排序並返回前 top_k 個
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def ask(self, question: str, top_k: int = 3) -> str:
        """RAG問答"""
        # 先搜尋相關文檔
        results = self.search(question, top_k)

        if not results:
            return "很抱歉，我在知識庫中沒有找到相關資訊。"

        # 基於搜尋結果生成回答
        context = "\n".join([r["content"] for r in results])
        question_lower = question.lower()

        if "人工智慧" in question or "ai" in question_lower:
            return "根據資料，人工智慧是電腦科學的分支，旨在創造能夠模擬人類智能的機器和系統。"
        elif "機器學習" in question or "machine learning" in question_lower:
            return "機器學習是人工智慧的子集，使用統計方法讓電腦從資料中學習和改進。"
        elif "深度學習" in question or "deep learning" in question_lower:
            return "深度學習是機器學習的分支，使用多層神經網路來處理複雜問題。"
        else:
            return f"根據知識庫的資料，{question}的相關資訊包含在找到的{len(results)}個文檔中。"

    def get_stats(self) -> Dict[str, Any]:
        """獲取統計資訊"""
        return {
            "total_documents": len(self.documents),
            "total_embeddings": len(self.documents),
            "index_size": f"{len(self.documents) * 0.1:.1f}MB",
        }


class MockAgentExecutor:
    """模擬 Agent 執行器"""

    def __init__(self):
        self.available_tools = ["calculator", "web_search", "file_reader"]

    def call_tool(self, tool_name: str, parameters: Dict) -> Dict[str, Any]:
        """調用工具"""
        if tool_name == "calculator":
            expression = parameters.get("expression", "0")
            try:
                # 安全的數學表達式計算
                import ast
                import operator as op

                # 支援的操作
                operators = {
                    ast.Add: op.add,
                    ast.Sub: op.sub,
                    ast.Mult: op.mul,
                    ast.Div: op.truediv,
                    ast.Pow: op.pow,
                    ast.BitXor: op.xor,
                    ast.USub: op.neg,
                }

                def eval_expr(expr):
                    return eval_(ast.parse(expr, mode="eval").body)

                def eval_(node):
                    if isinstance(node, ast.Num):
                        return node.n
                    elif isinstance(node, ast.BinOp):
                        return operators[type(node.op)](
                            eval_(node.left), eval_(node.right)
                        )
                    elif isinstance(node, ast.UnaryOp):
                        return operators[type(node.op)](eval_(node.operand))
                    else:
                        raise TypeError(node)

                result = eval_expr(expression)
                return {
                    "success": True,
                    "result": {"value": result, "expression": expression},
                }
            except:
                return {"success": False, "error": "Invalid mathematical expression"}

        elif tool_name == "web_search":
            query = parameters.get("query", "")
            num_results = parameters.get("num_results", 3)

            # 模擬搜尋結果
            mock_results = [
                {
                    "title": f"關於{query}的搜尋結果1",
                    "url": "https://example1.com",
                    "snippet": f"{query}相關內容...",
                },
                {
                    "title": f"關於{query}的搜尋結果2",
                    "url": "https://example2.com",
                    "snippet": f"更多{query}資訊...",
                },
                {
                    "title": f"關於{query}的搜尋結果3",
                    "url": "https://example3.com",
                    "snippet": f"{query}詳細說明...",
                },
            ]

            return {
                "success": True,
                "result": {
                    "query": query,
                    "results": mock_results[:num_results],
                    "total_found": len(mock_results),
                },
            }

        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    def execute_task(
        self, task_description: str, parameters: Dict = None
    ) -> Dict[str, Any]:
        """執行複合任務"""
        task_lower = task_description.lower()

        if "計算" in task_description or "calculate" in task_lower:
            tools_used = ["calculator"]
            result = "任務完成：已使用計算工具處理數學問題。"
        elif "搜尋" in task_description or "search" in task_lower:
            tools_used = ["web_search"]
            result = "任務完成：已使用搜尋工具查找資訊。"
        else:
            tools_used = ["llm"]
            result = f"任務完成：{task_description}（模擬執行）"

        return {
            "task_description": task_description,
            "result": result,
            "tools_used": tools_used,
            "steps_taken": len(tools_used) + 1,
        }


class MockGameEngine:
    """模擬遊戲引擎"""

    def __init__(self):
        self.active_games = {}
        self.game_counter = 0

    def new_game(self, persona: str, scenario: str) -> Dict[str, Any]:
        """創建新遊戲"""
        self.game_counter += 1
        game_id = f"game_{self.game_counter}"

        # 根據人設和場景生成初始場景
        persona_data = {
            "friendly_guide": "艾麗絲微笑著向你揮手",
            "wise_mentor": "智者靜靜地看著你，眼中閃爍著智慧的光芒",
        }.get(persona, "一個神秘的角色出現了")

        scenario_data = {
            "fantasy_forest": "你站在魔法森林的邊緣，古樹參天，神秘的光芒在樹葉間舞動。",
            "cyberpunk_city": "霓虹燈閃爍的街道上，機器與人類交織在一起。",
            "space_station": "太空站的走廊中，透過窗戶可以看到無垠的星空。",
        }.get(scenario, "你來到了一個未知的地方。")

        initial_scene = f"{scenario_data} {persona_data}。"

        # 遊戲狀態
        game_state = {
            "game_id": game_id,
            "persona": persona,
            "scenario": scenario,
            "current_scene": initial_scene,
            "turn_count": 0,
            "player_status": {"health": 100, "energy": 100},
            "inventory": [],
            "history": [],
        }

        self.active_games[game_id] = game_state

        return {
            "game_id": game_id,
            "initial_scene": initial_scene,
            "available_actions": self._get_available_actions(scenario),
            "player_status": game_state["player_status"],
        }

    def game_step(self, game_id: str, action: str) -> Dict[str, Any]:
        """遊戲步驟"""
        if game_id not in self.active_games:
            raise ValueError(f"Game {game_id} not found")

        game = self.active_games[game_id]
        game["turn_count"] += 1
        game["history"].append({"turn": game["turn_count"], "action": action})

        # 根據行動生成新場景
        action_lower = action.lower()

        if "探索" in action or "explore" in action_lower:
            new_scene = "你小心翼翼地向前探索，發現了一些有趣的線索..."
        elif "對話" in action or "talk" in action_lower:
            new_scene = "你與遇到的角色展開了深入的對話，獲得了寶貴的資訊..."
        elif "戰鬥" in action or "fight" in action_lower:
            new_scene = "一場激烈的戰鬥展開了！你需要運用智慧和技能..."
            game["player_status"]["energy"] -= 10
        elif "休息" in action or "rest" in action_lower:
            new_scene = "你找了個安全的地方休息，恢復了體力..."
            game["player_status"]["energy"] = min(
                100, game["player_status"]["energy"] + 20
            )
        else:
            new_scene = f"你選擇了「{action}」，周圍的環境隨之發生了變化..."

        game["current_scene"] = new_scene

        return {
            "game_id": game_id,
            "turn_count": game["turn_count"],
            "scene_description": new_scene,
            "available_actions": self._get_available_actions(game["scenario"]),
            "player_status": game["player_status"],
            "game_state": "ongoing",
        }

    def _get_available_actions(self, scenario: str) -> List[str]:
        """獲取可用行動"""
        base_actions = ["觀察周圍", "休息", "檢查狀態"]

        scenario_actions = {
            "fantasy_forest": ["探索森林", "尋找魔法物品", "與精靈對話"],
            "cyberpunk_city": ["黑市交易", "資料駭入", "接取任務"],
            "space_station": ["檢查控制台", "聯繫地面", "修理設備"],
        }

        return base_actions + scenario_actions.get(scenario, ["繼續探索", "尋找線索"])
