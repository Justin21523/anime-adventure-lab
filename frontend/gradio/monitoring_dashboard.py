# frontend/gradio_app/monitoring_dashboard.py
import gradio as gr
import requests
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"


class MonitoringUI:
    """Gradio-based monitoring dashboard"""

    def __init__(self):
        self.api_url = f"{API_BASE_URL}{API_PREFIX}"

    def get_system_metrics(self) -> Tuple[str, str, str]:
        """Get system metrics for display"""
        try:
            response = requests.get(f"{self.api_url}/monitoring/metrics", timeout=5)
            if response.status_code == 200:
                data = response.json()

                # Format system info
                system_info = f"""
                **CPU 使用率**: {data.get('cpu_percent', 0):.1f}%
                **記憶體使用率**: {data.get('memory_percent', 0):.1f}%
                **記憶體使用量**: {data.get('memory_used_gb', 0):.1f} / {data.get('memory_total_gb', 0):.1f} GB
                **磁碟使用率**: {data.get('disk_percent', 0):.1f}%
                """

                # GPU info
                gpu_info = "**GPU 狀態**:\n"
                gpu_metrics = data.get("gpu_metrics", {})
                if gpu_metrics:
                    for gpu_id, metrics in gpu_metrics.items():
                        usage = (metrics["memory_used"] / metrics["memory_total"]) * 100
                        gpu_info += f"- {gpu_id.upper()}: {usage:.1f}% ({metrics['memory_used']:.1f}GB / {metrics['memory_total']:.1f}GB)\n"
                else:
                    gpu_info += "- No GPU available\n"

                # Status summary
                status = (
                    "🟢 系統正常運行"
                    if data.get("cpu_percent", 0) < 80
                    and data.get("memory_percent", 0) < 85
                    else "🟡 系統負載較高"
                )

                return system_info, gpu_info, status
            else:
                return "無法取得系統資訊", "無法取得GPU資訊", "🔴 API 連線失敗"

        except Exception as e:
            return f"錯誤: {str(e)}", "無法取得GPU資訊", "🔴 連線錯誤"

    def get_task_metrics(self) -> Tuple[str, str]:
        """Get task execution metrics"""
        try:
            response = requests.get(f"{self.api_url}/monitoring/tasks", timeout=5)
            if response.status_code == 200:
                data = response.json()

                task_info = f"""
                **總任務數**: {data.get('total_tasks', 0)}
                **已完成**: {data.get('completed_tasks', 0)}
                **失敗數**: {data.get('failed_tasks', 0)}
                **等待中**: {data.get('pending_tasks', 0)}
                **平均執行時間**: {data.get('avg_execution_time', 0):.0f}ms
                **每分鐘任務數**: {data.get('tasks_per_minute', 0):.1f}
                **錯誤率**: {data.get('error_rate', 0):.1%}
                """

                # Task status visualization
                task_chart = self.create_task_chart(data)

                return task_info, task_chart
            else:
                return "無法取得任務資訊", None

        except Exception as e:
            return f"錯誤: {str(e)}", None

    def create_task_chart(self, task_data: Dict[str, Any]) -> go.Figure:
        """Create task status pie chart"""
        labels = ["已完成", "失敗", "等待中"]
        values = [
            task_data.get("completed_tasks", 0),
            task_data.get("failed_tasks", 0),
            task_data.get("pending_tasks", 0),
        ]
        colors = ["#10b981", "#ef4444", "#f59e0b"]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels, values=values, marker=dict(colors=colors), hole=0.4
                )
            ]
        )

        fig.update_layout(
            title="任務執行狀態 Task Status",
            showlegend=True,
            height=400,
            font=dict(size=12),
        )

        return fig

    def get_batch_jobs(self, status_filter: str = None) -> str:
        """Get recent batch jobs"""
        try:
            params = {"limit": 20}
            if status_filter and status_filter != "全部":
                status_map = {
                    "等待中": "PENDING",
                    "處理中": "PROCESSING",
                    "已完成": "COMPLETED",
                    "失敗": "FAILED",
                    "已取消": "CANCELLED",
                }
                params["status"] = status_map.get(status_filter)

            response = requests.get(
                f"{self.api_url}/batch/list", params=params, timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                jobs = data.get("jobs", [])

                if not jobs:
                    return "暫無批次任務"

                # Format job list
                job_list = []
                for job in jobs:
                    progress = 0
                    if job.get("total_items", 0) > 0:
                        progress = (
                            job.get("processed_items", 0) / job["total_items"]
                        ) * 100

                    status_emoji = {
                        "PENDING": "⏳",
                        "PROCESSING": "⚡",
                        "COMPLETED": "✅",
                        "FAILED": "❌",
                        "CANCELLED": "⏹️",
                    }.get(job.get("status"), "❓")

                    job_info = f"""
                    {status_emoji} **{job.get('job_id', '')[:8]}** - {job.get('job_type', '')}
                    - 狀態: {job.get('status', '')}
                    - 進度: {progress:.1f}% ({job.get('processed_items', 0)}/{job.get('total_items', 0)})
                    - 建立時間: {job.get('created_at', '')[:19] if job.get('created_at') else ''}
                    """

                    if job.get("error_message"):
                        job_info += f"    - 錯誤: {job['error_message'][:100]}..."

                    job_list.append(job_info)

                return "\n".join(job_list)
            else:
                return "無法取得批次任務資訊"

        except Exception as e:
            return f"錯誤: {str(e)}"

    def get_worker_status(self) -> str:
        """Get Celery worker status"""
        try:
            response = requests.get(f"{self.api_url}/monitoring/workers", timeout=5)
            if response.status_code == 200:
                workers = response.json()

                if not workers:
                    return "暫無活動的工作節點"

                worker_info = []
                for worker in workers:
                    status_emoji = "🟢" if worker.get("active_tasks", 0) > 0 else "🟡"
                    info = f"""
                    {status_emoji} **{worker.get('name', '')}**
                    - 活動任務: {worker.get('active_tasks', 0)}
                    - 已處理任務: {worker.get('processed_tasks', 0)}
                    - 最後心跳: {worker.get('last_heartbeat', '')[:19] if worker.get('last_heartbeat') else ''}
                    """
                    worker_info.append(info)

                return "\n".join(worker_info)
            else:
                return "無法取得工作節點資訊"

        except Exception as e:
            return f"錯誤: {str(e)}"

    def submit_test_batch(self, job_type: str, item_count: int) -> str:
        """Submit a test batch job"""
        try:
            # Create test inputs based on job type
            if job_type == "caption":
                inputs = [f"/tmp/test_image_{i}.jpg" for i in range(item_count)]
                config = {"max_length": 50, "num_beams": 3}
            elif job_type == "vqa":
                inputs = [
                    {
                        "image_path": f"/tmp/test_image_{i}.jpg",
                        "question": f"What is in this image {i}?",
                    }
                    for i in range(item_count)
                ]
                config = {"max_length": 100}
            elif job_type == "chat":
                inputs = [
                    [{"role": "user", "content": f"Hello, this is test message {i}"}]
                    for i in range(item_count)
                ]
                config = {"max_length": 200, "temperature": 0.7}
            else:
                return "不支援的任務類型"

            payload = {"job_type": job_type, "inputs": inputs, "config": config}

            response = requests.post(
                f"{self.api_url}/batch/submit", json=payload, timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return f"✅ 測試任務已提交\nJob ID: {data.get('job_id', '')}\nTask ID: {data.get('task_id', '')}"
            else:
                return f"❌ 提交失敗: {response.text}"

        except Exception as e:
            return f"❌ 錯誤: {str(e)}"

    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(
            title="Multi-Modal Lab 監控儀表板", theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown("# 🖥️ Multi-Modal Lab 監控儀表板")
            gr.Markdown("即時監控系統資源、任務執行狀態和批次處理進度")

            with gr.Tabs():
                # System Monitoring Tab
                with gr.Tab("🔧 系統監控"):
                    with gr.Row():
                        with gr.Column():
                            system_info = gr.Markdown("載入中...")
                            gpu_info = gr.Markdown("載入中...")
                        with gr.Column():
                            system_status = gr.Markdown("載入中...")
                            refresh_btn = gr.Button("🔄 重新整理", variant="primary")

                    refresh_btn.click(
                        fn=self.get_system_metrics,
                        outputs=[system_info, gpu_info, system_status],
                    )

                # Task Monitoring Tab
                with gr.Tab("📊 任務監控"):
                    with gr.Row():
                        with gr.Column():
                            task_info = gr.Markdown("載入中...")
                            task_refresh_btn = gr.Button(
                                "🔄 重新整理任務", variant="primary"
                            )
                        with gr.Column():
                            task_chart = gr.Plot(label="任務狀態圖表")

                    task_refresh_btn.click(
                        fn=self.get_task_metrics, outputs=[task_info, task_chart]
                    )

                # Batch Jobs Tab
                with gr.Tab("📋 批次任務"):
                    with gr.Row():
                        status_filter = gr.Dropdown(
                            choices=[
                                "全部",
                                "等待中",
                                "處理中",
                                "已完成",
                                "失敗",
                                "已取消",
                            ],
                            value="全部",
                            label="狀態篩選",
                        )
                        jobs_refresh_btn = gr.Button("🔄 重新整理", variant="primary")

                    batch_jobs_display = gr.Markdown("載入中...")

                    jobs_refresh_btn.click(
                        fn=self.get_batch_jobs,
                        inputs=[status_filter],
                        outputs=[batch_jobs_display],
                    )

                    status_filter.change(
                        fn=self.get_batch_jobs,
                        inputs=[status_filter],
                        outputs=[batch_jobs_display],
                    )

                # Workers Tab
                with gr.Tab("👷 工作節點"):
                    worker_info = gr.Markdown("載入中...")
                    worker_refresh_btn = gr.Button(
                        "🔄 重新整理工作節點", variant="primary"
                    )

                    worker_refresh_btn.click(
                        fn=self.get_worker_status, outputs=[worker_info]
                    )

                # Test Tab
                with gr.Tab("🧪 測試工具"):
                    gr.Markdown("### 提交測試批次任務")
                    with gr.Row():
                        test_job_type = gr.Dropdown(
                            choices=["caption", "vqa", "chat"],
                            value="caption",
                            label="任務類型",
                        )
                        test_item_count = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1, label="測試項目數量"
                        )
                        submit_test_btn = gr.Button(
                            "🚀 提交測試任務", variant="primary"
                        )

                    test_result = gr.Markdown("")

                    submit_test_btn.click(
                        fn=self.submit_test_batch,
                        inputs=[test_job_type, test_item_count],
                        outputs=[test_result],
                    )

            # Auto-refresh every 30 seconds
            interface.load(
                fn=self.get_system_metrics,
                outputs=[system_info, gpu_info, system_status],
            )

            interface.load(fn=self.get_task_metrics, outputs=[task_info, task_chart])

            interface.load(
                fn=self.get_batch_jobs,
                inputs=[status_filter],
                outputs=[batch_jobs_display],
            )

            interface.load(fn=self.get_worker_status, outputs=[worker_info])

        return interface


def create_monitoring_app():
    """Create and launch monitoring dashboard"""
    ui = MonitoringUI()
    interface = ui.create_interface()

    return interface


if __name__ == "__main__":
    app = create_monitoring_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from main Gradio app
        share=False,
        debug=True,
    )
