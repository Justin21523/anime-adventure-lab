# frontend/gradio/app.py
"""
SagaForge Gradio Frontend
Web UI prototype for interacting with the API
"""

import os
import json
import httpx
import gradio as gr
from gradio.themes import Soft
from gradio.themes import Soft
from datetime import datetime
from typing import Dict, Any, Optional

# Import shared cache for consistent paths
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from core.shared_cache import bootstrap_cache
from core.config import get_config

# setup shared cache
cache = bootstrap_cache()

# frontend/gradio/app.py
import gradio as gr
import requests
import json
from typing import Dict, Any, List, Tuple

API_BASE = "http://localhost:8000/api/v1"


class StoryState:
    """Manage story state across Gradio interactions"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to initial state"""
        self.persona = None
        self.game_state = None
        self.conversation_history = []

    def load_sample_data(self):
        """Load sample persona and game state from API"""
        try:
            # Get sample persona
            persona_resp = requests.get(f"{API_BASE}/persona/sample")
            if persona_resp.status_code == 200:
                self.persona = persona_resp.json()

            # Get sample game state
            state_resp = requests.get(f"{API_BASE}/gamestate/sample")
            if state_resp.status_code == 200:
                self.game_state = state_resp.json()

            return True
        except Exception as e:
            print(f"Failed to load sample data: {e}")
            return False


# Global story state
story_state = StoryState()


def format_persona_display(persona: Dict[str, Any]) -> str:
    """Format persona for display"""
    if not persona:
        return "未載入角色"

    lines = [
        f"**{persona['name']}** ({persona.get('age', '?')}歲)",
        f"**性格**: {', '.join(persona.get('personality', []))}",
        f"**背景**: {persona.get('background', '未設定')}",
        f"**說話風格**: {persona.get('speaking_style', '未設定')}",
        f"**目標**: {'; '.join(persona.get('goals', []))}",
    ]
    return "\n".join(lines)


def format_game_state_display(state: Dict[str, Any]) -> str:
    """Format game state for display"""
    if not state:
        return "未載入遊戲狀態"

    lines = [
        f"**場景**: {state.get('scene_id', '未知')}",
        f"**回合數**: {state.get('turn_count', 0)}",
        f"**位置**: {state.get('current_location', '未知')}",
        f"**物品**: {', '.join(state.get('inventory', []))}",
    ]

    if state.get("flags"):
        lines.append("**狀態標記**:")
        for key, value in state["flags"].items():
            lines.append(f"  - {key}: {value}")

    return "\n".join(lines)


def process_turn(
    player_input: str, choice_id: str = None  # type: ignore
) -> Tuple[str, str, str, List[Tuple[str, str]]]:
    """Process a story turn and return updated displays"""
    if not story_state.persona or not story_state.game_state:
        return "錯誤: 請先載入角色和遊戲狀態", "", "", []

    if not player_input.strip():
        return "請輸入你的行動或回應", "", "", []

    try:
        # Prepare request
        request_data = {
            "player_input": player_input,
            "persona": story_state.persona,
            "game_state": story_state.game_state,
            "choice_id": choice_id,
        }

        # Send to API
        response = requests.post(f"{API_BASE}/turn", json=request_data)

        if response.status_code != 200:
            return f"API 錯誤: {response.status_code}", "", "", []

        result = response.json()

        # Update game state if provided
        if result.get("updated_state"):
            story_state.game_state = result["updated_state"]

        # Format response
        narration = result.get("narration", "")

        # Format dialogues
        dialogues = []
        for d in result.get("dialogues", []):
            emotion = f" ({d['emotion']})" if d.get("emotion") else ""
            dialogues.append(f"**{d['speaker']}**{emotion}: {d['text']}")

        dialogue_text = "\n".join(dialogues) if dialogues else ""

        # Format choices
        choices = []
        for c in result.get("choices", []):
            desc = f" - {c['description']}" if c.get("description") else ""
            choices.append((c["text"] + desc, c["id"]))

        # Add to conversation history
        story_state.conversation_history.append(
            {"player": player_input, "narration": narration, "dialogues": dialogue_text}
        )

        # Update displays
        persona_display = format_persona_display(story_state.persona)
        state_display = format_game_state_display(story_state.game_state)

        return narration, dialogue_text, state_display, choices

    except Exception as e:
        return f"處理錯誤: {str(e)}", "", "", []


def load_sample_data():
    """Load sample data and return displays"""
    success = story_state.load_sample_data()

    if success:
        persona_display = format_persona_display(story_state.persona)  # type: ignore
        state_display = format_game_state_display(story_state.game_state)  # type: ignore
        return (
            persona_display,
            state_display,
            "樣本數據載入成功！輸入你的第一個行動開始冒險。",
        )
    else:
        return (
            "載入失敗",
            "載入失敗",
            "無法從 API 載入樣本數據，請檢查 API 服務是否正常運行。",
        )


def reset_story():
    """Reset story state"""
    story_state.reset()
    return "", "", "故事已重置", "", []


def create_gradio_interface():
    """Create Gradio interface"""

    with gr.Blocks(title="SagaForge - 互動故事引擎", theme=Soft()) as demo:

        gr.Markdown("# 🏰 SagaForge - 互動故事引擎")
        gr.Markdown("體驗由 AI 驅動的互動式故事冒險")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 角色資訊")
                persona_display = gr.Markdown("未載入角色")

                gr.Markdown("## 遊戲狀態")
                state_display = gr.Markdown("未載入遊戲狀態")

                with gr.Row():
                    load_btn = gr.Button("載入樣本", variant="primary")
                    reset_btn = gr.Button("重置故事", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("## 故事進行")

                narration_display = gr.Markdown("", label="敘述")
                dialogue_display = gr.Markdown("", label="對話")

                player_input = gr.Textbox(
                    label="你的行動", placeholder="輸入你想要做的事情...", lines=2
                )

                choices = gr.Radio(
                    label="選擇項", choices=[], value=None, visible=False
                )

                submit_btn = gr.Button("提交行動", variant="primary")
                status_display = gr.Markdown("")

        # Event handlers
        def handle_turn(input_text, selected_choice):
            choice_id = selected_choice if selected_choice else None
            narration, dialogues, state_display_new, new_choices = process_turn(
                input_text, choice_id  # type: ignore
            )

            # Update choices
            choices_visible = len(new_choices) > 0
            choices_update = gr.update(
                choices=new_choices, visible=choices_visible, value=None
            )

            return (
                narration,
                dialogues,
                state_display_new,
                "",  # Clear input
                choices_update,
                "",  # Clear status
            )

        load_btn.click(
            load_sample_data, outputs=[persona_display, state_display, status_display]
        )

        reset_btn.click(
            reset_story,
            outputs=[
                persona_display,
                state_display,
                status_display,
                player_input,
                choices,
            ],
        )

        submit_btn.click(
            handle_turn,
            inputs=[player_input, choices],
            outputs=[
                narration_display,
                dialogue_display,
                state_display,
                player_input,
                choices,
                status_display,
            ],
        )

        # Allow Enter key submission
        player_input.submit(
            handle_turn,
            inputs=[player_input, choices],
            outputs=[
                narration_display,
                dialogue_display,
                state_display,
                player_input,
                choices,
                status_display,
            ],
        )

    return demo


class SagaForgeUI:
    """Main UI controller"""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.client = httpx.Client(timeout=30.0)

        # Bootstrap cache for consistent paths
        self.cache = bootstrap_cache()
        self.config = get_config()

    async def check_api_health(self) -> tuple[str, str]:
        """Check API server health status"""
        try:
            response = self.client.get(f"{self.api_base_url}/api/v1/healthz")
            if response.status_code == 200:
                data = response.json()

                # Format health info
                status = f"🟢 **API Status**: {data['status'].upper()}\n"
                status += f"⏱️ **Uptime**: {data['uptime_seconds']:.1f}s\n"
                status += f"🖥️ **CPU**: {data['system']['cpu_percent']:.1f}%\n"
                status += f"💾 **Memory**: {data['system']['memory_percent']:.1f}% ({data['system']['memory_available_gb']:.1f}GB free)\n"

                if data["cache"].get("gpu_available"):
                    gpu_mem = data["cache"].get("gpu_memory", {})
                    if gpu_mem:
                        status += f"🎮 **GPU**: {gpu_mem.get('allocated_gb', 0):.1f}GB / {gpu_mem.get('total_gb', 0):.1f}GB\n"
                    else:
                        status += f"🎮 **GPU**: Available ({data['cache']['gpu_count']} devices)\n"
                else:
                    status += "🎮 **GPU**: Not available\n"

                # Format raw JSON for details
                details = json.dumps(data, indent=2, ensure_ascii=False)

                return status, details
            else:
                return f"🔴 **API Error**: HTTP {response.status_code}", str(
                    response.text
                )

        except Exception as e:
            return f"🔴 **Connection Error**: {str(e)}", ""

    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface"""

        with gr.Blocks(
            title="SagaForge - Adventure Game Engine",
            theme=Soft(),
            css="""
                .main-header { text-align: center; margin-bottom: 20px; }
                .status-box { background: #f8f9fa; padding: 15px; border-radius: 8px; }
                .placeholder-box {
                    background: #e9ecef;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    border: 2px dashed #adb5bd;
                }
            """,
        ) as demo:

            # Header
            gr.Markdown(
                """
            # 🎮 SagaForge Adventure Engine

            **Stage 1 Prototype** - System Bootstrap & Health Monitoring

            ---
            """,
                elem_classes="main-header",
            )

            # Status Panel
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## 🏥 System Health")

                    health_status = gr.Markdown(
                        "Checking API status...", elem_classes="status-box"
                    )
                    check_btn = gr.Button("🔄 Refresh Status", variant="primary")

                    with gr.Accordion("📋 Detailed Health Info", open=False):
                        health_details = gr.Code(language="json", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("## ⚙️ Cache Info")

                    cache_summary = self.cache.get_summary()
                    cache_info = f"""
**Cache Root**: `{cache_summary['cache_root']}`

**GPU Info**:
- Available: {cache_summary['gpu_info']['cuda_available']}
- Devices: {cache_summary['gpu_info']['device_count']}

**Directories**: {len(cache_summary['directories'])} created

**Environment**:
- CUDA_VISIBLE_DEVICES: {cache_summary['env_vars']['CUDA_VISIBLE_DEVICES']}
"""
                    gr.Markdown(cache_info, elem_classes="status-box")

            gr.Markdown("---")

            # Feature Placeholders
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                    <div class="placeholder-box">
                        <h3>🤖 LLM Chat</h3>
                        <p><em>Coming in Stage 2</em></p>
                        <p>Interactive chat with persona and story engine</p>
                    </div>
                    """
                    )

                with gr.Column():
                    gr.Markdown(
                        """
                    <div class="placeholder-box">
                        <h3>📚 RAG Upload</h3>
                        <p><em>Coming in Stage 3</em></p>
                        <p>Upload documents for retrieval-augmented generation</p>
                    </div>
                    """
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                    <div class="placeholder-box">
                        <h3>🎨 T2I Generation</h3>
                        <p><em>Coming in Stage 5</em></p>
                        <p>Generate images with SDXL and ControlNet</p>
                    </div>
                    """
                    )

                with gr.Column():
                    gr.Markdown(
                        """
                    <div class="placeholder-box">
                        <h3>🔬 LoRA Training</h3>
                        <p><em>Coming in Stage 7</em></p>
                        <p>Fine-tune character models with LoRA</p>
                    </div>
                    """
                    )

            # Event handlers
            check_btn.click(
                fn=self.check_api_health, outputs=[health_status, health_details]
            )

            # Auto-check on load
            demo.load(fn=self.check_api_health, outputs=[health_status, health_details])

        return demo


def create_app(api_url: str = None) -> gr.Blocks:  # type: ignore
    """Create and return Gradio app"""
    if api_url is None:
        config = get_config()
        api_url = f"http://{config.api.host}:{config.api.port}"

    ui = SagaForgeUI(api_url)
    return ui.create_interface()


def main():
    """Run the Gradio app"""
    print("🚀 Starting SagaForge Web UI...")

    # Check if API server is specified
    api_url = os.getenv("API_URL", "http://localhost:8000")

    app = create_app(api_url)

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        # show_tips=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
