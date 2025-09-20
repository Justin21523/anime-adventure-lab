# core/monitoring/dashboard.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, Any
import json
from .metrics import MetricsCollector
from ..batch.manager import BatchManager
from .logger import structured_logger


class MonitoringDashboard:
    """Simple monitoring dashboard"""

    def __init__(
        self, metrics_collector: MetricsCollector, batch_manager: BatchManager
    ):
        self.metrics_collector = metrics_collector
        self.batch_manager = batch_manager
        self.templates = Jinja2Templates(directory="backend/templates")

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data"""
        try:
            # System metrics
            import psutil
            import torch

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            gpu_info = []
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_info.append(
                        {
                            "id": i,
                            "name": torch.cuda.get_device_name(i),
                            "memory_used": torch.cuda.memory_allocated(i) / 1024**3,
                            "memory_total": torch.cuda.get_device_properties(
                                i
                            ).total_memory
                            / 1024**3,
                        }
                    )

            # Task metrics
            task_metrics = await self.metrics_collector.get_task_metrics()

            # Recent jobs
            recent_jobs = await self.batch_manager.list_jobs(limit=10)

            # Worker status
            from workers.celery_app import celery_app

            try:
                inspect = celery_app.control.inspect()
                stats = inspect.stats()
                active_tasks = inspect.active()

                workers = []
                for worker_name, worker_stats in (stats or {}).items():
                    workers.append(
                        {
                            "name": worker_name,
                            "active_tasks": len(active_tasks.get(worker_name, [])),
                            "processed": worker_stats.get("total", {}).get(
                                "worker.processed", 0
                            ),
                        }
                    )
            except:
                workers = []

            return {
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / 1024**3,
                    "memory_total_gb": memory.total / 1024**3,
                    "gpu_info": gpu_info,
                },
                "tasks": task_metrics,
                "recent_jobs": recent_jobs,
                "workers": workers,
            }

        except Exception as e:
            structured_logger.error(f"Failed to get dashboard data: {e}")
            return {}

    def render_dashboard_html(self, data: Dict[str, Any]) -> str:
        """Render dashboard HTML"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Modal Lab Monitoring</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .metric {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee; }}
        .metric:last-child {{ border-bottom: none; }}
        .value {{ font-weight: bold; color: #2563eb; }}
        .progress {{ width: 100%; height: 20px; background: #e5e7eb; border-radius: 10px; overflow: hidden; }}
        .progress-bar {{ height: 100%; background: linear-gradient(90deg, #10b981, #059669); transition: width 0.3s; }}
        .status-good {{ color: #10b981; }}
        .status-warning {{ color: #f59e0b; }}
        .status-error {{ color: #ef4444; }}
        h1 {{ color: #1f2937; }}
        h2 {{ color: #374151; margin-top: 0; }}
        .refresh-btn {{ background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }}
        .refresh-btn:hover {{ background: #1d4ed8; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f9fafb; font-weight: 600; }}
    </style>
    <script>
        function refreshDashboard() {{
            location.reload();
        }}
        setInterval(refreshDashboard, 30000); // Auto-refresh every 30 seconds
    </script>
</head>
<body>
    <div class="container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;">
            <h1>Multi-Modal Lab Monitoring Dashboard</h1>
            <button class="refresh-btn" onclick="refreshDashboard()">Refresh</button>
        </div>

        <div class="grid">
            <!-- System Resources -->
            <div class="card">
                <h2>系統資源 System Resources</h2>
                <div class="metric">
                    <span>CPU 使用率</span>
                    <span class="value">{data.get('system', {}).get('cpu_percent', 0):.1f}%</span>
                </div>
                <div class="progress">
                    <div class="progress-bar" style="width: {data.get('system', {}).get('cpu_percent', 0)}%"></div>
                </div>

                <div class="metric">
                    <span>記憶體使用率</span>
                    <span class="value">{data.get('system', {}).get('memory_percent', 0):.1f}%</span>
                </div>
                <div class="progress">
                    <div class="progress-bar" style="width: {data.get('system', {}).get('memory_percent', 0)}%"></div>
                </div>

                <div class="metric">
                    <span>記憶體使用量</span>
                    <span class="value">{data.get('system', {}).get('memory_used_gb', 0):.1f} / {data.get('system', {}).get('memory_total_gb', 0):.1f} GB</span>
                </div>
            </div>

            <!-- GPU Info -->
            <div class="card">
                <h2>GPU 狀態 GPU Status</h2>
                {self._render_gpu_info(data.get('system', {}).get('gpu_info', []))}
            </div>

            <!-- Task Metrics -->
            <div class="card">
                <h2>任務統計 Task Metrics</h2>
                <div class="metric">
                    <span>總任務數</span>
                    <span class="value">{data.get('tasks', {}).get('total_tasks', 0)}</span>
                </div>
                <div class="metric">
                    <span>已完成</span>
                    <span class="value status-good">{data.get('tasks', {}).get('completed_tasks', 0)}</span>
                </div>
                <div class="metric">
                    <span>失敗數</span>
                    <span class="value status-error">{data.get('tasks', {}).get('failed_tasks', 0)}</span>
                </div>
                <div class="metric">
                    <span>錯誤率</span>
                    <span class="value">{data.get('tasks', {}).get('error_rate', 0):.1%}</span>
                </div>
                <div class="metric">
                    <span>平均執行時間</span>
                    <span class="value">{data.get('tasks', {}).get('avg_execution_time', 0):.0f}ms</span>
                </div>
            </div>

            <!-- Workers -->
            <div class="card">
                <h2>工作節點 Workers</h2>
                {self._render_workers(data.get('workers', []))}
            </div>
        </div>

        <!-- Recent Jobs -->
        <div class="card">
            <h2>最近任務 Recent Jobs</h2>
            {self._render_recent_jobs(data.get('recent_jobs', []))}
        </div>
    </div>
</body>
</html>
        """

    def _render_gpu_info(self, gpu_info):
        if not gpu_info:
            return "<p>No GPU devices available</p>"

        html = ""
        for gpu in gpu_info:
            usage_percent = (
                (gpu["memory_used"] / gpu["memory_total"]) * 100
                if gpu["memory_total"] > 0
                else 0
            )
            html += f"""
                <div class="metric">
                    <span>GPU {gpu['id']}: {gpu['name']}</span>
                    <span class="value">{usage_percent:.1f}%</span>
                </div>
                <div class="progress">
                    <div class="progress-bar" style="width: {usage_percent}%"></div>
                </div>
                <div class="metric">
                    <span>VRAM</span>
                    <span class="value">{gpu['memory_used']:.1f} / {gpu['memory_total']:.1f} GB</span>
                </div>
            """
        return html

    def _render_workers(self, workers):
        if not workers:
            return "<p>No active workers</p>"

        html = ""
        for worker in workers:
            status_class = (
                "status-good" if worker["active_tasks"] > 0 else "status-warning"
            )
            html += f"""
                <div class="metric">
                    <span>{worker['name']}</span>
                    <span class="value {status_class}">{worker['active_tasks']} active</span>
                </div>
                <div class="metric">
                    <span>已處理任務</span>
                    <span class="value">{worker['processed']}</span>
                </div>
            """
        return html

    def _render_recent_jobs(self, jobs):
        if not jobs:
            return "<p>No recent jobs</p>"

        html = """
            <table>
                <thead>
                    <tr>
                        <th>Job ID</th>
                        <th>類型 Type</th>
                        <th>狀態 Status</th>
                        <th>進度 Progress</th>
                        <th>建立時間 Created</th>
                    </tr>
                </thead>
                <tbody>
        """

        for job in jobs:
            progress = 0
            if job.get("total_items", 0) > 0:
                progress = (job.get("processed_items", 0) / job["total_items"]) * 100

            status_class = {
                "COMPLETED": "status-good",
                "FAILED": "status-error",
                "PROCESSING": "status-warning",
                "PENDING": "status-warning",
            }.get(job.get("status", ""), "")

            html += f"""
                <tr>
                    <td>{job.get('job_id', '')[:8]}...</td>
                    <td>{job.get('job_type', '')}</td>
                    <td><span class="{status_class}">{job.get('status', '')}</span></td>
                    <td>{progress:.1f}%</td>
                    <td>{job.get('created_at', '')[:19] if job.get('created_at') else ''}</td>
                </tr>
            """

        html += "</tbody></table>"
        return html
