# tests/fixtures/sample_data.py
"""
測試用樣本資料
"""

import json
from pathlib import Path
from typing import Dict, List, Any

# Sample images (base64 encoded small test images)
SAMPLE_IMAGES = {
    "red_square": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAFYSURBVBiVY2RgYPgPBQwMDAwMjCCamZGBkYWBgfE/NjBShwGGQQqRASRGNUAqGBgYGP7//w8SMzIyMv7//x+kGKcBRkZGBgYGhv//QRqwgxhzVrJNBskxMTExMHInJBgY",
    "blue_circle": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAFXSURBVBiVY2RgYPgPBQwMDAwMjCCamZGBkYGRgYEBJsAwKQMyEKsIDAwMDIwMDAz//4M0//8PEjMyMjL+//8fpBinAUZGRgYGBgYGBoZGRgYGRgYGRkZGBgYGRkZGhv//QRqwgxhzVrJNBskxMTExMHInJBgY",
}

# Sample documents for RAG testing
SAMPLE_DOCUMENTS = [
    {
        "content": """人工智慧（Artificial Intelligence, AI）是電腦科學的一個分支，
        旨在創造能夠模擬人類智能的機器和系統。AI包含機器學習、深度學習、
        自然語言處理等多個子領域。""",
        "metadata": {
            "title": "人工智慧基礎介紹",
            "source": "ai_basics.txt",
            "author": "AI專家",
            "category": "基礎知識",
            "language": "zh-TW",
        },
    },
    {
        "content": """機器學習（Machine Learning, ML）是人工智慧的一個重要子集，
        它使用統計方法讓電腦系統能夠從資料中學習和改進，而無需明確程式化。
        常見的ML算法包括監督學習、無監督學習和強化學習。""",
        "metadata": {
            "title": "機器學習概述",
            "source": "ml_overview.txt",
            "author": "ML研究員",
            "category": "進階概念",
            "language": "zh-TW",
        },
    },
    {
        "content": """深度學習（Deep Learning, DL）是機器學習的一個分支，
        使用具有多層的人工神經網路來模擬人腦的工作方式。
        它在影像識別、語音處理和自然語言理解方面表現優異。""",
        "metadata": {
            "title": "深度學習原理",
            "source": "dl_principles.txt",
            "author": "DL專家",
            "category": "專業技術",
            "language": "zh-TW",
        },
    },
    {
        "content": """Computer vision is a field of AI that enables machines to interpret
        and understand visual information from the world. It involves techniques like
        image classification, object detection, and semantic segmentation.""",
        "metadata": {
            "title": "Computer Vision Basics",
            "source": "cv_intro.txt",
            "author": "CV Researcher",
            "category": "Technical",
            "language": "en",
        },
    },
]

# Sample conversation history
SAMPLE_CONVERSATIONS = [
    {
        "conversation_id": "test-conv-1",
        "messages": [
            {"role": "user", "content": "你好，請介紹一下什麼是人工智慧？"},
            {
                "role": "assistant",
                "content": "人工智慧是電腦科學的分支，旨在創造能模擬人類智能的系統...",
            },
            {"role": "user", "content": "那機器學習和AI有什麼關係？"},
            {
                "role": "assistant",
                "content": "機器學習是人工智慧的重要子集，專注於讓電腦從資料中學習...",
            },
        ],
    },
    {
        "conversation_id": "test-conv-2",
        "messages": [
            {"role": "user", "content": "我想了解深度學習"},
            {
                "role": "assistant",
                "content": "深度學習使用多層神經網路來處理複雜問題...",
            },
        ],
    },
]

# Game scenarios and personas
SAMPLE_GAME_DATA = {
    "personas": {
        "friendly_guide": {
            "name": "艾麗絲",
            "description": "友善的冒險嚮導",
            "personality": "樂觀、好奇、樂於助人",
            "speaking_style": "溫和親切，喜歡用比喻",
        },
        "wise_mentor": {
            "name": "智者",
            "description": "充滿智慧的導師",
            "personality": "深思熟慮、博學、耐心",
            "speaking_style": "深奧但清晰，常引用古語",
        },
    },
    "scenarios": {
        "fantasy_forest": {
            "name": "魔法森林探險",
            "description": "在充滿神秘生物的魔法森林中探險",
            "initial_scene": "你站在古老森林的邊緣，陽光透過樹葉灑下斑駁的光影...",
            "available_actions": ["進入森林", "觀察周圍", "呼喊同伴", "檢查裝備"],
        },
        "cyberpunk_city": {
            "name": "賽博朋克都市",
            "description": "在高科技低生活的未來都市中生存",
            "initial_scene": "霓虹燈閃爍的街道上，你穿梭在人群和機器人之間...",
            "available_actions": ["前往酒吧", "接取任務", "黑市交易", "資料駭入"],
        },
    },
}

# Agent tools configuration
SAMPLE_AGENT_TOOLS = {
    "calculator": {
        "name": "calculator",
        "description": "執行數學計算",
        "parameters": {"expression": {"type": "string", "description": "數學表達式"}},
        "examples": [
            {"input": "2 + 3 * 4", "output": 14},
            {"input": "sqrt(16)", "output": 4},
        ],
    },
    "web_search": {
        "name": "web_search",
        "description": "搜尋網路資訊",
        "parameters": {
            "query": {"type": "string", "description": "搜尋關鍵字"},
            "num_results": {"type": "integer", "default": 5},
        },
        "examples": [{"input": "人工智慧最新發展", "output": "找到5個相關結果..."}],
    },
}


def get_sample_data(data_type: str) -> Any:
    """獲取指定類型的樣本資料"""
    mapping = {
        "documents": SAMPLE_DOCUMENTS,
        "conversations": SAMPLE_CONVERSATIONS,
        "game_data": SAMPLE_GAME_DATA,
        "agent_tools": SAMPLE_AGENT_TOOLS,
        "images": SAMPLE_IMAGES,
    }
    return mapping.get(data_type, [])
