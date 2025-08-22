# core/llm/prompt_templates.py
from typing import Dict, Any
from ..story.data_structures import Persona, GameState


class PromptTemplates:
    """Traditional Chinese prompt templates for story generation"""

    SYSTEM_PROMPT = """你是一個專業的互動小說引擎，專門創作沉浸式的故事體驗。

核心規則：
1. 以繁體中文回應，保持優美的敘事風格
2. 嚴格遵循角色設定和當前遊戲狀態
3. 必須以 JSON 格式回應，包含 narration、dialogues、choices 欄位
4. 每回合推進故事情節，但不要過於快速
5. 選擇項要有意義且影響劇情發展
6. 對話要符合角色性格和說話風格

JSON 格式要求：
{
  "narration": "故事敘述文字，描述場景、氛圍、動作",
  "dialogues": [
    {"speaker": "角色名", "text": "對話內容", "emotion": "情緒描述"}
  ],
  "choices": [
    {"id": "choice_1", "text": "選擇文字", "description": "選擇結果預覽"}
  ]
}

記住：
- 敘述要生動具體，營造畫面感
- 對話要自然流暢，符合角色特色
- 選擇要平衡，避免過於明顯的「正確答案」
- 保持適度的懸念和情感張力"""

    @staticmethod
    def build_user_prompt(
        player_input: str,
        persona: Persona,
        game_state: GameState,
        choice_id: str = None,  # type: ignore
    ) -> str:
        """Build user prompt with context"""

        prompt_parts = []

        # Character context
        prompt_parts.append("=== 角色設定 ===")
        prompt_parts.append(persona.to_prompt_context())

        # Game state context
        prompt_parts.append("\n=== 遊戲狀態 ===")
        prompt_parts.append(f"當前場景: {game_state.scene_id}")
        prompt_parts.append(f"回合數: {game_state.turn_count}")
        prompt_parts.append(f"位置: {game_state.current_location}")

        if game_state.flags:
            prompt_parts.append("重要標記:")
            for key, value in game_state.flags.items():
                prompt_parts.append(f"  - {key}: {value}")

        if game_state.inventory:
            prompt_parts.append(f"持有物品: {', '.join(game_state.inventory)}")

        if game_state.relationships:
            prompt_parts.append("角色關係:")
            for rel in game_state.relationships[-3:]:  # Show last 3 relationships
                prompt_parts.append(
                    f"  - {rel.character_a} 與 {rel.character_b}: {rel.relation_type.value} ({rel.strength:.1f})"
                )

        # Recent history
        if game_state.choice_history:
            prompt_parts.append("\n=== 近期選擇 ===")
            for choice in game_state.choice_history[-2:]:  # Last 2 choices
                prompt_parts.append(f"- {choice.get('text', '')}")

        if game_state.timeline_notes:
            prompt_parts.append("\n=== 劇情要點 ===")
            for note in game_state.timeline_notes[-3:]:  # Last 3 notes
                prompt_parts.append(f"- {note}")

        # Current input
        prompt_parts.append("\n=== 玩家輸入 ===")
        if choice_id:
            prompt_parts.append(f"選擇了: {choice_id}")

        prompt_parts.append(f"輸入: {player_input}")

        # Request
        prompt_parts.append("\n=== 請求 ===")
        prompt_parts.append(
            "請根據以上情境，以角色視角推進故事。回應必須是有效的 JSON 格式。"
        )

        return "\n".join(prompt_parts)

    @staticmethod
    def build_few_shot_examples() -> str:
        """Build few-shot examples for better JSON generation"""
        return """
範例回應格式：

{
  "narration": "夕陽西下，古老的圖書館裡瀰漫著書香和灰塵的味道。你走過一排排高聳的書架，腳步聲在寂靜中格外清晰。",
  "dialogues": [
    {"speaker": "圖書管理員", "text": "年輕人，你在找什麼特別的書嗎？", "emotion": "好奇"}
  ],
  "choices": [
    {"id": "ask_about_magic", "text": "詢問魔法書籍", "description": "可能獲得魔法知識，但也可能引起懷疑"},
    {"id": "browse_quietly", "text": "安靜地瀏覽", "description": "安全但可能錯過重要資訊"},
    {"id": "leave_library", "text": "離開圖書館", "description": "避免風險但放棄當前機會"}
  ]
}"""
