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
from datetime import datetime
from typing import Dict, Any, Optional

# Import shared cache for consistent paths
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from core.shared_cache import bootstrap_cache
from core.config import get_config


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
