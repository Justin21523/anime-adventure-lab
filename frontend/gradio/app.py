# frontend/gradio/app.py
"""
SagaForge Gradio Frontend
Web UI prototype for interacting with the API
"""

import os
import json
import httpx
import gradio as gr
import requests
import logging
from PIL import Image
from gradio.themes import Soft
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import io
import base64

HAS_TIMER = hasattr(gr, "Timer")
# Import shared cache for consistent paths
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from core.shared_cache import bootstrap_cache
from core.config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
DEFAULT_API = os.getenv("API_BASE") or os.getenv("API_URL") or "http://localhost:8000"
API_BASE = DEFAULT_API.rstrip("/")
if not API_BASE.endswith("/api/v1"):
    API_BASE = f"{API_BASE}/api/v1"


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

    def __init__(self, api_base_url: str = API_BASE):
        base = api_base_url.rstrip("/")
        if not base.endswith("/api/v1"):
            base = f"{base}/api/v1"
        self.api_base_url = base
        self.client = httpx.Client(timeout=30.0)

        # Bootstrap cache for consistent paths
        self.cache = bootstrap_cache()
        self.config = get_config()

    # -------------------- Helpers -------------------- #
    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Any]:
        """Minimal request wrapper with unified error handling."""
        url = f"{self.api_base_url}{path}"
        try:
            resp = self.client.request(
                method, url, json=json, files=files, data=data, params=params
            )
            resp.raise_for_status()
            try:
                return True, resp.json()
            except Exception:
                return True, resp.text
        except httpx.HTTPStatusError as exc:
            return False, f"HTTP {exc.response.status_code}: {exc.response.text}"
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def _image_to_file(image: Image.Image, filename: str = "upload.png") -> tuple:
        """Convert PIL Image to an uploadable tuple for requests."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return (filename, buf, "image/png")

    async def check_api_health(self) -> tuple[str, str]:
        """Check API server health status"""
        ok, payload = self._request("GET", "/health")
        if not ok:
            return f"🔴 **API Error**: {payload}", ""

        try:
            data = payload
            status = f"🟢 **API Status**: {data.get('status', 'unknown').upper()}\n"
            status += f"⏱️ **Uptime**: {data.get('uptime_seconds', 0):.1f}s\n"
            system = data.get("system", {})
            status += f"🖥️ **CPU**: {system.get('cpu_percent', 0):.1f}%\n"
            status += f"💾 **Memory**: {system.get('memory_percent', 0):.1f}% ({system.get('memory_available_gb', 0):.1f}GB free)\n"

            cache = data.get("cache", {})
            if cache.get("gpu_available"):
                gpu_mem = cache.get("gpu_memory", {})
                if gpu_mem:
                    status += f"🎮 **GPU**: {gpu_mem.get('allocated_gb', 0):.1f}GB / {gpu_mem.get('total_gb', 0):.1f}GB\n"
                else:
                    status += f"🎮 **GPU**: Available ({cache.get('gpu_count', 0)} devices)\n"
            else:
                status += "🎮 **GPU**: Not available\n"

            details = json.dumps(data, indent=2, ensure_ascii=False)
            return status, details
        except Exception as exc:
            return f"🔴 **Parse Error**: {exc}", ""

    # -------------------- API actions -------------------- #
    def run_agent_task(
        self, prompt: str, max_iter: int = 5, max_tools: int = 2, cot: bool = True
    ) -> tuple[str, str, Dict[str, Any]]:
        if not prompt.strip():
            return "請輸入任務描述。", "", {}

        payload = {
            "task_description": prompt.strip(),
            "parameters": {
                "max_iterations": max_iter,
                "max_tools_per_iteration": max_tools,
                "enable_chain_of_thought": cot,
            },
        }

        ok, data = self._request("POST", "/agent/task", json=payload)
        if not ok:
            return f"執行失敗：{data}", "", {}

        reasoning = data.get("reasoning_chain") or []
        summary_lines = [
            f"**成功**: {data.get('success')}",
            f"**使用工具**: {', '.join(data.get('tools_used', [])) or '無'}",
            f"**步驟**: {data.get('steps_taken', 0)}",
            f"**耗時**: {data.get('execution_time_ms', 0):.1f} ms",
            "",
            f"**結果**:\n{data.get('result', '')}",
        ]
        reasoning_text = "\n".join(reasoning) if reasoning else "（未啟用或無推理鏈）"
        return "\n".join(summary_lines), reasoning_text, data

    def list_agent_tools(self) -> Dict[str, Any]:
        ok, data = self._request("GET", "/agent/tools")
        return data if ok else {"error": data}

    def call_agent_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"tool_name": tool_name, "parameters": parameters}
        ok, data = self._request("POST", "/agent/tools/call", json=payload)
        return data if ok else {"error": data}

    def run_caption(self, image: Image.Image, max_length: int = 80) -> str:
        if image is None:
            return "請上傳圖片。"
        files = {"image": self._image_to_file(image)}
        ok, data = self._request(
            "POST",
            "/caption",
            files=files,
            data={"max_length": max_length, "language": "zh"},
        )
        if not ok:
            return f"Caption 失敗：{data}"
        return data.get("caption", "（無結果）")

    def run_vqa(self, image: Image.Image, question: str) -> str:
        if image is None:
            return "請上傳圖片。"
        if not question.strip():
            return "請輸入問題。"
        files = {"image": self._image_to_file(image)}
        form_data = {"question": question}
        ok, data = self._request("POST", "/vqa", files=files, data=form_data)
        if not ok:
            return f"VQA 失敗：{data}"
        answer = data.get("answer", "（無答案）")
        conf = data.get("confidence", 0)
        return f"{answer}  (信心 {conf:.2f})"

    def run_t2i(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: Optional[int] = None,
    ) -> tuple[Optional[Image.Image], str, Dict[str, Any]]:
        if not prompt.strip():
            return None, "請輸入提示詞。", {}

        payload = {
            "prompt": prompt.strip(),
            "parameters": {
                "negative_prompt": negative_prompt or None,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance,
                "seed": seed,
            },
        }
        ok, data = self._request("POST", "/t2i/txt2img", json=payload)
        if not ok:
            return None, f"生成失敗：{data}", {}

        image_path = data.get("image_path")
        img = None
        if image_path and Path(image_path).exists():
            try:
                img = Image.open(image_path)
            except Exception as exc:
                logger.warning(f"Failed to open generated image: {exc}")

        info = {
            "model_used": data.get("model_used"),
            "parameters": data.get("parameters"),
            "generation_info": data.get("generation_info"),
        }
        return img, "生成完成", info

    def rag_upload(self, file_obj, world_id: str, tags: str) -> str:
        if file_obj is None:
            return "請上傳檔案。"
        file_bytes = file_obj.read()
        files = {"file": (file_obj.name, file_bytes, file_obj.type or "application/octet-stream")}
        data = {
            "world_id": world_id or "default",
            "tags": tags or "",
        }
        ok, resp = self._request("POST", "/rag/upload", files=files, data=data)
        if not ok:
            return f"上傳失敗：{resp}"
        return f"已加入 {resp.get('doc_id', '')} / added={resp.get('added', False)}"

    def rag_search(self, query: str, top_k: int = 5) -> tuple[str, Dict[str, Any]]:
        if not query.strip():
            return "請輸入查詢。", {}
        ok, resp = self._request("POST", "/rag/search", json={"query": query, "top_k": top_k})
        if not ok:
            return f"搜尋失敗：{resp}", {}
        items = resp.get("results", []) or resp.get("chunks", [])
        lines = []
        for i, item in enumerate(items[:top_k], 1):
            lines.append(f"{i}. {item.get('content', '')[:200]}")
        return "\n".join(lines) or "無結果", resp

    def rag_ask(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if not query.strip():
            return {"error": "請輸入問題"}
        ok, resp = self._request("POST", "/rag/ask", json={"query": query, "top_k": top_k})
        return resp if ok else {"error": resp}

    def rag_preview(self, doc_id: str) -> Dict[str, Any]:
        if not doc_id:
            return {"error": "請輸入 doc_id"}
        ok, resp = self._request("GET", "/rag/documents")
        if not ok:
            return {"error": resp}
        if isinstance(resp, dict):
            docs = resp.get("documents", [])
            for d in docs:
                if d.get("doc_id") == doc_id:
                    return d
        return {"info": "not found"}

    def format_rag_answer(self, resp: Dict[str, Any]) -> str:
        if not isinstance(resp, dict):
            return "無法解析回應"
        if "error" in resp:
            return f"錯誤：{resp['error']}"
        lines = [f"**問題**: {resp.get('query','')}"]
        lines.append(f"**回答**: {resp.get('answer','(無)')}")
        sources = resp.get("sources", [])
        if sources:
            lines.append("**來源**:")
            for i, s in enumerate(sources, 1):
                lines.append(f"{i}. {s.get('doc_id','?')} (score {s.get('score',0):.3f})")
        return "\n".join(lines)

    def rag_list_docs_with_choices(self):
        resp = self.rag_list_docs()
        doc_ids = []
        if isinstance(resp, dict):
            docs = resp.get("documents", [])
            doc_ids = [d.get("doc_id") for d in docs if d.get("doc_id")]
        table = []
        for d in resp.get("documents", []):
            table.append(
                {
                    "doc_id": d.get("doc_id"),
                    "title": d.get("title"),
                    "chunks": d.get("chunks"),
                    "chars": d.get("total_chars"),
                }
            )
        return resp, gr.update(choices=doc_ids, value=doc_ids[0] if doc_ids else None), table

    def rag_stats(self) -> Dict[str, Any]:
        ok, resp = self._request("GET", "/rag/stats")
        return resp if ok else {"error": resp}

    def rag_rebuild(self) -> str:
        ok, resp = self._request("POST", "/rag/rebuild")
        return "已重建索引" if ok else f"重建失敗：{resp}"

    def rag_clear(self) -> str:
        ok, resp = self._request("DELETE", "/rag/clear")
        return "已清空索引" if ok else f"清空失敗：{resp}"

    def rag_list_docs(self) -> Dict[str, Any]:
        ok, resp = self._request("GET", "/rag/documents")
        return resp if ok else {"error": resp}

    def rag_delete_doc(self, doc_id: str) -> str:
        if not doc_id:
            return "請選擇/輸入有效的 doc_id"
        ok, resp = self._request("DELETE", f"/rag/documents/{doc_id}")
        return "已刪除" if ok else f"刪除失敗：{resp}"

    def list_jobs(self) -> Dict[str, Any]:
        ok, resp = self._request("GET", "/jobs")
        return resp if ok else {"error": resp}

    def submit_lora_job(self, base_model: str, output_name: str, dataset: str) -> str:
        payload = {"base_model": base_model, "output_name": output_name, "dataset": dataset}
        ok, resp = self._request("POST", "/finetune/lora", json=payload)
        if not ok:
            return f"提交失敗：{resp}"
        return f"Job {resp.get('job_id')} 已建立（狀態 {resp.get('status')})"

    def job_status(self, job_id: str) -> Dict[str, Any]:
        ok, resp = self._request("GET", f"/jobs/{job_id}")
        return resp if ok else {"error": resp}

    def submit_batch(
        self, job_type: str, inputs: list[str], simulate: bool = True, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        payload = {"job_type": job_type, "inputs": inputs, "simulate": simulate, "config": config or {}}
        ok, resp = self._request("POST", "/batch/submit", json=payload)
        return resp if ok else {"error": resp}

    def batch_status(self, job_id: str) -> Dict[str, Any]:
        ok, resp = self._request("GET", f"/batch/status/{job_id}")
        return resp if ok else {"error": resp}

    def batch_progress(self, task_id: str) -> Dict[str, Any]:
        ok, resp = self._request("GET", f"/batch/progress/{task_id}")
        return resp if ok else {"error": resp}

    def batch_queue_stats(self) -> Dict[str, Any]:
        ok, resp = self._request("GET", "/batch/queues")
        return resp if ok else {"error": resp}

    def batch_results(self, job_id: str) -> Dict[str, Any]:
        ok, resp = self._request("GET", f"/batch/results/{job_id}")
        return resp if ok else {"error": resp}

    def download_results_preview(self, job_id: str) -> tuple[str, Dict[str, Any]]:
        info = self.batch_results(job_id)
        if "error" in info:
            return f"取得失敗：{info['error']}", {}

        results_path = info.get("results_path")
        if not results_path or not Path(results_path).exists():
            return "結果檔不存在或尚未產出。", info

        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            return f"讀取結果檔失敗：{exc}", info

        preview = data
        if isinstance(data, dict) and "results" in data:
            preview = data.get("results", [])[:5]
        if isinstance(preview, list) and len(preview) > 5:
            preview = preview[:5]
        download_url = info.get("download_url")
        if download_url and download_url.startswith("/"):
            download_url = f"{self.api_base_url}{download_url}"
        return f"結果檔：{results_path}", {
            "preview": preview,
            "raw": data,
            "download_url": download_url,
        }

    def system_metrics(self) -> tuple[str, Dict[str, Any]]:
        ok, resp = self._request("GET", "/monitoring/metrics")
        if not ok:
            return f"取得失敗：{resp}", {}
        summary = f"CPU {resp.get('cpu_percent',0):.1f}% | MEM {resp.get('memory_percent',0):.1f}% | Disk {resp.get('disk_percent',0):.1f}%"
        return summary, resp

    def performance_report(self) -> Dict[str, Any]:
        ok, resp = self._request("GET", "/monitoring/performance")
        return resp if ok else {"error": resp}

    def admin_system(self) -> Dict[str, Any]:
        ok, resp = self._request("GET", "/admin/system")
        return resp if ok else {"error": resp}

    def unload_models(self) -> str:
        ok, resp = self._request("POST", "/admin/models/control", json={"action": "unload_all"})
        return "OK" if ok else f"失敗：{resp}"

    def health_components(self) -> Dict[str, Any]:
        ok, resp = self._request("GET", "/monitoring/health")
        return resp if ok else {"error": resp}

    def list_personas(self) -> List[Dict[str, Any]]:
        ok, data = self._request("GET", "/story/personas")
        if not ok:
            return []
        if isinstance(data, dict) and "personas" in data:
            return data.get("personas", [])
        if isinstance(data, list):
            return data
        return []

    def start_story_session(
        self,
        player_name: str,
        persona_id: str,
        setting: str,
        difficulty: str,
        enrich_with_rag: bool,
        use_agent: bool,
    ) -> tuple[str, str, List[tuple], Dict[str, Any]]:
        payload = {
            "player_name": player_name or "Player",
            "persona_id": persona_id or None,
            "setting": setting or "default",
            "difficulty": difficulty or "normal",
            "enrich_with_rag": enrich_with_rag,
            "use_agent": use_agent,
        }
        ok, data = self._request("POST", "/story/session", json=payload)
        if not ok:
            return "", f"建立故事失敗：{data}", [], {}
        choices = []
        for c in data.get("choices", []):
            choices.append((c.get("text", ""), c.get("id")))
        narrative = data.get("narrative", "")
        return data.get("session_id", ""), narrative, choices, data

    def continue_story(
        self, session_id: str, player_input: str, choice_id: Optional[str]
    ) -> tuple[str, List[tuple], Dict[str, Any]]:
        if not session_id:
            return "請先建立故事", [], {}
        payload = {
            "session_id": session_id,
            "player_input": player_input or "",
            "choice_id": choice_id or None,
        }
        ok, data = self._request("POST", "/story/turn", json=payload)
        if not ok:
            return f"行動失敗：{data}", [], {}
        choices = []
        for c in data.get("choices", []):
            choices.append((c.get("text", ""), c.get("id")))
        return data.get("narrative", ""), choices, data

    def story_session_detail(self, session_id: str) -> Dict[str, Any]:
        ok, resp = self._request("GET", f"/story/session/{session_id}")
        return resp if ok else {"error": resp}

    def story_session_context(self, session_id: str) -> Dict[str, Any]:
        ok, resp = self._request("GET", f"/story/session/{session_id}/context")
        return resp if ok else {"error": resp}

    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface"""

        accent = "#00b19d"
        with gr.Blocks(
            title="SagaForge - Multi-Modal Lab",
            theme=Soft(),
            css=f"""
                :root {{
                    --color-primary: {accent};
                    --shadow-spread: 0;
                }}
                body {{
                    background: radial-gradient(circle at 20% 20%, #0f172a 0, #0b1120 40%, #080c19 100%);
                    color: #e5e7eb;
                    font-family: "Inter", "Noto Sans", "Segoe UI", sans-serif;
                }}
                .hero {{
                    background: linear-gradient(135deg, rgba(0,177,157,0.12), rgba(59,130,246,0.08));
                    border: 1px solid rgba(255,255,255,0.08);
                    padding: 18px;
                    border-radius: 14px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
                }}
                .panel {{
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.05);
                    border-radius: 12px;
                    padding: 12px 14px;
                    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
                }}
            """,
        ) as demo:

            gr.Markdown(
                """
                <div class="hero">
                    <h1 style="margin-bottom: 8px;">SagaForge 前端控制台</h1>
                    <p style="margin: 0;">多模態 API 探索：健康檢查、Agent 任務、圖像理解與文字生圖。</p>
                </div>
                """
            )

            with gr.Tabs():
                # ---------------- Health ----------------
                with gr.Tab("🩺 系統狀態"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### 服務健康")
                            health_status = gr.Markdown("正在檢查 API ...", elem_classes="panel")
                            check_btn = gr.Button("重新整理", variant="primary")
                            with gr.Accordion("詳細資訊", open=False):
                                health_details = gr.Code(language="json", interactive=False)
                        with gr.Column(scale=1):
                            gr.Markdown("### 快取與環境", elem_classes="panel")
                            cache_summary = self.cache.get_summary()
                            cache_info = f"""
**Cache Root**: `{cache_summary['cache_root']}`

**GPU Info**
- Available: {cache_summary['gpu_info']['cuda_available']}
- Devices: {cache_summary['gpu_info']['device_count']}

**ENV**
- CUDA_VISIBLE_DEVICES: {cache_summary['env_vars']['CUDA_VISIBLE_DEVICES']}
"""
                            gr.Markdown(cache_info, elem_classes="panel")

                    check_btn.click(fn=self.check_api_health, outputs=[health_status, health_details])
                    demo.load(fn=self.check_api_health, outputs=[health_status, health_details])

                # ---------------- Agent ----------------
                with gr.Tab("🧠 Agent 任務"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            task_box = gr.Textbox(
                                label="任務描述",
                                placeholder="例：整理 2023 年 AI 關鍵事件並計算平均發佈間隔",
                                lines=4,
                            )
                            with gr.Row():
                                max_iter = gr.Slider(1, 10, value=5, step=1, label="最大迭代")
                                max_tools = gr.Slider(1, 5, value=2, step=1, label="每步工具數")
                            cot = gr.Checkbox(value=True, label="啟用推理鏈")
                            run_agent_btn = gr.Button("送出任務", variant="primary")
                        with gr.Column(scale=2):
                            agent_result = gr.Markdown(label="結果", elem_classes="panel")
                            reasoning_view = gr.Markdown(label="推理鏈", elem_classes="panel")
                            agent_json = gr.JSON(label="原始回應", elem_classes="panel")

                    run_agent_btn.click(
                        fn=self.run_agent_task,
                        inputs=[task_box, max_iter, max_tools, cot],
                        outputs=[agent_result, reasoning_view, agent_json],
                    )

                    with gr.Accordion("可用工具", open=False):
                        tools_btn = gr.Button("取得工具列表")
                        tools_json = gr.JSON(label="工具清單", elem_classes="panel")
                        tool_call_name = gr.Textbox(label="工具名稱", placeholder="calculator")
                        tool_call_params = gr.Textbox(
                            label="參數(JSON)", placeholder='{"expression": "2+2"}', lines=2
                        )
                        tool_call_btn = gr.Button("執行指定工具", variant="secondary")
                        tool_call_result = gr.JSON(label="工具結果", elem_classes="panel")

                    def _list_tools():
                        return self.list_agent_tools()

                    tools_btn.click(fn=_list_tools, outputs=tools_json)
                    tool_call_btn.click(
                        fn=lambda name, params: self.call_agent_tool(
                            name.strip(), json.loads(params or "{}")
                        ),
                        inputs=[tool_call_name, tool_call_params],
                        outputs=tool_call_result,
                    )

                # ---------------- VLM ----------------
                with gr.Tab("🖼️ 圖像理解"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            img_input = gr.Image(label="上傳圖片", type="pil")
                            cap_length = gr.Slider(20, 160, value=80, step=5, label="Caption 長度")
                            caption_btn = gr.Button("產生 Caption", variant="primary")
                            caption_out = gr.Markdown(elem_classes="panel")
                        with gr.Column(scale=2):
                            vqa_question = gr.Textbox(
                                label="VQA 問題",
                                placeholder="例：圖中有多少人？",
                                lines=2,
                            )
                            vqa_btn = gr.Button("送出 VQA", variant="secondary")
                            vqa_out = gr.Markdown(elem_classes="panel")

                    caption_btn.click(
                        fn=self.run_caption,
                        inputs=[img_input, cap_length],
                        outputs=caption_out,
                    )
                    vqa_btn.click(
                        fn=self.run_vqa,
                        inputs=[img_input, vqa_question],
                        outputs=vqa_out,
                    )

                # ---------------- T2I ----------------
                with gr.Tab("🎨 文字生圖"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            prompt = gr.Textbox(
                                label="提示詞",
                                placeholder="例：sunset over neon city, anime style, cinematic lighting",
                                lines=3,
                            )
                            negative = gr.Textbox(
                                label="負面提示詞",
                                placeholder="blurry, low quality, artifacts",
                                lines=2,
                            )
                            with gr.Row():
                                width = gr.Slider(256, 1024, value=640, step=64, label="寬度")
                                height = gr.Slider(256, 1024, value=640, step=64, label="高度")
                            with gr.Row():
                                steps = gr.Slider(10, 50, value=20, step=1, label="Steps")
                                guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance")
                            seed = gr.Number(label="Seed (可留空)", precision=0)
                            t2i_btn = gr.Button("生成圖片", variant="primary")
                        with gr.Column(scale=2):
                            gen_image = gr.Image(label="輸出", type="pil")
                            t2i_status = gr.Markdown(elem_classes="panel")
                            t2i_info = gr.JSON(label="生成資訊", elem_classes="panel")

                    def _t2i_wrapper(prompt, negative, width, height, steps, guidance, seed):
                        img, status, info = self.run_t2i(
                            prompt, negative, int(width), int(height), int(steps), float(guidance), seed if seed else None
                        )
                        return img, status, info

                    t2i_btn.click(
                        fn=_t2i_wrapper,
                        inputs=[prompt, negative, width, height, steps, guidance, seed],
                        outputs=[gen_image, t2i_status, t2i_info],
                    )

                # ---------------- Story ----------------
                with gr.Tab("📖 Story 冒險"):
                    personas = self.list_personas()
                    persona_choices = [(p.get("name", p.get("id", "")), p.get("id", "")) for p in personas] or []

                    session_state = gr.State("")
                    choices_state = gr.State([])

                    with gr.Row():
                        with gr.Column(scale=2):
                            player_name = gr.Textbox(label="玩家名稱", value="玩家", lines=1)
                            setting = gr.Textbox(label="世界觀/設定", value="fantasy forest", lines=1)
                            difficulty = gr.Dropdown(
                                ["easy", "normal", "hard"], value="normal", label="難度"
                            )
                            persona_dd = gr.Dropdown(
                                choices=[c[1] for c in persona_choices],
                                value=persona_choices[0][1] if persona_choices else None,
                                label="角色模板",
                            )
                            with gr.Row():
                                enrich_rag = gr.Checkbox(value=False, label="使用 RAG 增益")
                                use_agent = gr.Checkbox(value=True, label="Agent 輔助")
                            start_btn = gr.Button("建立故事", variant="primary")
                        with gr.Column(scale=3):
                            story_narrative = gr.Markdown(label="敘事", elem_classes="panel")
                            story_choices = gr.Radio(label="可選動作", choices=[], visible=False)
                            story_input = gr.Textbox(label="你的行動", lines=2)
                            act_btn = gr.Button("送出行動", variant="primary")
                            story_raw = gr.JSON(label="詳細回應", elem_classes="panel")
                            story_detail_btn = gr.Button("取得會話詳情/旗標", variant="secondary")
                            story_context_btn = gr.Button("取得上下文 Snapshot", variant="secondary")
                            story_detail_json = gr.JSON(label="Session Detail", elem_classes="panel")
                            story_context_json = gr.JSON(label="Context Snapshot", elem_classes="panel")
                            story_history = gr.Markdown(label="最近回合", elem_classes="panel")
                            story_inventory = gr.Markdown(label="物品/旗標", elem_classes="panel")
                            story_history_state = gr.State([])

                    def _start_story(player, persona, setting, difficulty, rag, agent):
                        sid, narrative, choices, raw = self.start_story_session(
                            player, persona, setting, difficulty, rag, agent
                        )
                        choice_update = gr.update(choices=choices, visible=bool(choices), value=None)
                        history = f"Turn 1: {narrative}" if narrative else ""
                        inv = raw.get("inventory", [])
                        flags = raw.get("flags", {})
                        inv_text = f"Inventory: {', '.join(inv) or '無'}\nFlags: {flags}"
                        return sid, narrative, choices, choice_update, raw, history, inv_text

                    start_btn.click(
                        fn=_start_story,
                        inputs=[player_name, persona_dd, setting, difficulty, enrich_rag, use_agent],
                        outputs=[session_state, story_narrative, choices_state, story_choices, story_raw, story_history, story_inventory],
                    )

                    def _act(session_id, user_input, choice_selection, cached_choices):
                        choice_id = choice_selection or None
                        narrative, choices, raw = self.continue_story(session_id, user_input, choice_id)
                        choice_update = gr.update(choices=choices, visible=bool(choices), value=None)
                        history_lines = []
                        if isinstance(raw, dict):
                            turn = raw.get("turn_count", "?")
                            history_lines.append(f"Turn {turn}: {raw.get('narrative','')}")
                        inv = raw.get("inventory", [])
                        flags = raw.get("flags", {})
                        inv_text = f"Inventory: {', '.join(inv) or '無'}\nFlags: {flags}"
                        return narrative, choices, choice_update, raw, "\n".join(history_lines), inv_text

                    act_btn.click(
                        fn=_act,
                        inputs=[session_state, story_input, story_choices, choices_state],
                        outputs=[story_narrative, choices_state, story_choices, story_raw, story_history, story_inventory],
                    )
                    story_detail_btn.click(
                        fn=self.story_session_detail, inputs=session_state, outputs=story_detail_json
                    )
                    story_context_btn.click(
                        fn=self.story_session_context, inputs=session_state, outputs=story_context_json
                    )

                # ---------------- RAG ----------------
                with gr.Tab("📚 RAG 管理"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            rag_file = gr.File(label="上傳文件")
                            rag_world = gr.Textbox(label="World ID", value="default", lines=1)
                            rag_tags = gr.Textbox(label="Tags (逗號分隔)", value="", lines=1)
                            rag_upload_btn = gr.Button("上傳並建檔", variant="primary")
                            rag_upload_status = gr.Markdown(elem_classes="panel")
                        with gr.Column(scale=2):
                            rag_query = gr.Textbox(label="搜尋查詢", placeholder="例：Explain the magic system", lines=2)
                            rag_topk = gr.Slider(1, 10, value=5, step=1, label="Top K")
                            rag_search_btn = gr.Button("搜尋", variant="secondary")
                            rag_result = gr.Markdown(elem_classes="panel")
                            rag_raw = gr.JSON(label="原始結果", elem_classes="panel")
                            rag_ask_btn = gr.Button("RAG 問答", variant="secondary")
                            rag_answer_fmt = gr.Markdown(elem_classes="panel")
                            rag_answer = gr.JSON(label="RAG 回答 (raw)", elem_classes="panel")

                    def _rag_upload(file, world, tags):
                        return self.rag_upload(file, world, tags)

                    rag_upload_btn.click(
                        fn=_rag_upload, inputs=[rag_file, rag_world, rag_tags], outputs=rag_upload_status
                    )

                    def _rag_search(query, topk):
                        text, raw = self.rag_search(query, int(topk))
                        return text, raw

                    rag_search_btn.click(
                        fn=_rag_search, inputs=[rag_query, rag_topk], outputs=[rag_result, rag_raw]
                    )
                    rag_ask_btn.click(
                        fn=lambda q, k: (self.format_rag_answer(self.rag_ask(q, int(k))), self.rag_ask(q, int(k))),
                        inputs=[rag_query, rag_topk],
                        outputs=[rag_answer_fmt, rag_answer],
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            rag_stats_btn = gr.Button("查看索引統計", variant="secondary")
                            rag_stats_json = gr.JSON(label="RAG Stats", elem_classes="panel")
                            rag_rebuild_btn = gr.Button("重建索引", variant="secondary")
                            rag_clear_btn = gr.Button("清空索引", variant="secondary")
                            rag_action_status = gr.Markdown(elem_classes="panel")
                        with gr.Column(scale=2):
                            rag_list_btn = gr.Button("列出文件", variant="secondary")
                            rag_docs_json = gr.JSON(label="文件列表", elem_classes="panel")
                            rag_doc_choice = gr.Dropdown(label="文件/Chunk ID", choices=[])
                            rag_table = gr.DataFrame(headers=["doc_id", "title", "chunks", "chars"], interactive=False)
                            rag_del_id = gr.Textbox(label="Doc/Chunk ID", placeholder="輸入要刪除的 ID")
                            rag_del_btn = gr.Button("刪除文件/Chunk", variant="secondary")
                            rag_del_status = gr.Markdown(elem_classes="panel")
                            rag_preview_btn = gr.Button("預覽內容", variant="secondary")
                            rag_preview_json = gr.JSON(label="預覽結果", elem_classes="panel")

                    rag_stats_btn.click(fn=self.rag_stats, outputs=rag_stats_json)
                    rag_rebuild_btn.click(fn=self.rag_rebuild, outputs=rag_action_status)
                    rag_clear_btn.click(fn=self.rag_clear, outputs=rag_action_status)
                    rag_list_btn.click(
                        fn=self.rag_list_docs_with_choices, outputs=[rag_docs_json, rag_doc_choice, rag_table]
                    )
                    rag_del_btn.click(fn=self.rag_delete_doc, inputs=rag_del_id, outputs=rag_del_status)
                    rag_preview_btn.click(fn=self.rag_preview, inputs=rag_doc_choice, outputs=rag_preview_json)

                # ---------------- Batch 任務 ----------------
                with gr.Tab("📦 批次任務"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            batch_type = gr.Dropdown(["caption", "vqa", "chat"], value="caption", label="Job Type")
                            batch_inputs = gr.Textbox(
                                label="輸入列表（每行一筆路徑或文字）",
                                placeholder="/path/img1.jpg\n/path/img2.jpg",
                                lines=4,
                            )
                            batch_sim = gr.Checkbox(value=True, label="模擬執行（無需 Celery）")
                            batch_submit_btn = gr.Button("提交批次 Job", variant="primary")
                            batch_submit_json = gr.JSON(label="提交回應", elem_classes="panel")
                        with gr.Column(scale=2):
                            batch_job_id = gr.Textbox(label="Job ID", placeholder="提交後取得")
                            batch_task_id = gr.Textbox(label="Task ID (若有 Celery)")
                            batch_status_btn = gr.Button("查詢狀態", variant="secondary")
                            batch_progress_btn = gr.Button("查詢進度", variant="secondary")
                            batch_status_json = gr.JSON(label="Job/Task 狀態", elem_classes="panel")
                            queue_btn = gr.Button("Queue 統計", variant="secondary")
                            queue_json = gr.JSON(label="Queue Stats", elem_classes="panel")
                            results_btn = gr.Button("下載/預覽結果", variant="secondary")
                            results_status = gr.Markdown(elem_classes="panel")
                            results_json = gr.JSON(label="結果預覽", elem_classes="panel")
                            batch_auto = gr.Checkbox(label="自動輪詢狀態/進度 (5s)", value=False)

                    def _submit_batch(job_type, inputs, simulate):
                        items = [line.strip() for line in (inputs or "").splitlines() if line.strip()]
                        if not items:
                            return {"error": "請輸入至少一筆資料"}
                        return self.submit_batch(job_type, items, simulate)

                    batch_submit_btn.click(
                        fn=_submit_batch,
                        inputs=[batch_type, batch_inputs, batch_sim],
                        outputs=batch_submit_json,
                    )
                    batch_status_btn.click(
                        fn=self.batch_status, inputs=batch_job_id, outputs=batch_status_json
                    )
                    batch_progress_btn.click(
                        fn=self.batch_progress, inputs=batch_task_id, outputs=batch_status_json
                    )
                    queue_btn.click(fn=self.batch_queue_stats, outputs=queue_json)
                    results_btn.click(
                        fn=lambda job_id: self.download_results_preview(job_id),
                        inputs=batch_job_id,
                        outputs=[results_status, results_json],
                    )

                    if HAS_TIMER:
                        batch_timer = gr.Timer(interval=5, active=False)

                        def _auto_poll(job_id, task_id):
                            status = self.batch_status(job_id) if job_id else {}
                            prog = self.batch_progress(task_id) if task_id else {}
                            return {"status": status, "progress": prog}

                        batch_timer.tick(fn=_auto_poll, inputs=[batch_job_id, batch_task_id], outputs=batch_status_json)

                        def _toggle_batch_poll(active: bool):
                            batch_timer.active = active
                            if active:
                                return gr.update(label="自動輪詢狀態/進度 (5s) ✅")
                            return gr.update(label="自動輪詢狀態/進度 (5s)")

                        batch_auto.change(fn=_toggle_batch_poll, inputs=batch_auto, outputs=batch_auto)
                    else:
                        batch_auto.visible = False
                        gr.Markdown("提示：Gradio 版本 <4 不支援 Timer，自動輪詢已停用。", elem_classes="panel")

                # ---------------- LoRA / 訓練 ----------------
                with gr.Tab("🔬 LoRA / 訓練"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            base_model = gr.Textbox(label="Base Model", value="stable-diffusion-xl")
                            output_name = gr.Textbox(label="輸出名稱", value="my-lora")
                            dataset_path = gr.Textbox(label="資料集路徑", value="data/my_dataset")
                            lora_btn = gr.Button("提交 LoRA Job", variant="primary")
                            lora_status = gr.Markdown(elem_classes="panel")
                        with gr.Column(scale=2):
                            jobs_btn = gr.Button("刷新 Jobs", variant="secondary")
                            jobs_json = gr.JSON(label="Jobs", elem_classes="panel")
                            job_id_input = gr.Textbox(label="查詢 Job ID", placeholder="輸入 job_id")
                            job_query_btn = gr.Button("查詢單筆 Job", variant="secondary")
                            job_query_json = gr.JSON(label="Job Detail", elem_classes="panel")

                    lora_btn.click(
                        fn=self.submit_lora_job,
                        inputs=[base_model, output_name, dataset_path],
                        outputs=lora_status,
                    )

                    def _list_jobs():
                        return self.list_jobs()

                    jobs_btn.click(fn=_list_jobs, outputs=jobs_json)
                    job_query_btn.click(fn=self.job_status, inputs=job_id_input, outputs=job_query_json)

                # ---------------- 監控 / 管理 ----------------
                with gr.Tab("📈 監控與管理"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            metrics_btn = gr.Button("即時系統指標", variant="primary")
                            metrics_text = gr.Markdown(elem_classes="panel")
                            metrics_json = gr.JSON(label="Metrics", elem_classes="panel")
                        with gr.Column(scale=2):
                            perf_btn = gr.Button("性能報告", variant="secondary")
                            perf_json = gr.JSON(label="Performance", elem_classes="panel")
                            admin_btn = gr.Button("模型卸載 (LLM/VLM)", variant="secondary")
                            admin_status = gr.Markdown(elem_classes="panel")
                            admin_info_btn = gr.Button("系統資訊", variant="secondary")
                            admin_info_json = gr.JSON(label="Admin Info", elem_classes="panel")
                            celery_btn = gr.Button("Celery/Redis 健康檢查", variant="secondary")
                            celery_json = gr.JSON(label="Health Components", elem_classes="panel")
                            auto_refresh = gr.Checkbox(label="自動刷新監控 (5s)", value=False)

                    def _metrics():
                        text, raw = self.system_metrics()
                        return text, raw

                    metrics_btn.click(fn=_metrics, outputs=[metrics_text, metrics_json])
                    perf_btn.click(fn=self.performance_report, outputs=perf_json)
                    admin_btn.click(fn=self.unload_models, outputs=admin_status)
                    admin_info_btn.click(fn=self.admin_system, outputs=admin_info_json)
                    # 利用 /monitoring/health 來檢查 Celery/Redis/RAG 等狀態
                    celery_btn.click(fn=self.health_components, outputs=celery_json)

                    if HAS_TIMER:
                        monitor_timer = gr.Timer(interval=5, active=False)

                        def _auto_monitor():
                            text, raw = self.system_metrics()
                            return text, raw, self.health_components()

                        monitor_timer.tick(
                            fn=_auto_monitor,
                            outputs=[metrics_text, metrics_json, celery_json],
                        )

                        def _toggle_monitor(active: bool):
                            monitor_timer.active = active
                            if active:
                                return gr.update(label="自動刷新監控 (5s) ✅")
                            return gr.update(label="自動刷新監控 (5s)")

                        auto_refresh.change(fn=_toggle_monitor, inputs=auto_refresh, outputs=auto_refresh)
                    else:
                        auto_refresh.visible = False
                        gr.Markdown("提示：Gradio 版本 <4 不支援 Timer，自動刷新已停用。", elem_classes="panel")

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
