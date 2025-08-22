# frontend/gradio/performance_monitor.py
import gradio as gr
import requests
import json
import time
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any


class PerformanceMonitor:
    """Real-time performance monitoring UI"""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.metrics_history = []
        self.max_history_points = 60  # Keep last 60 data points

    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            response = requests.get(f"{self.api_base_url}/monitoring/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics"""
        try:
            response = requests.get(
                f"{self.api_base_url}/monitoring/metrics", timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                # Add timestamp
                data["timestamp"] = datetime.now()
                return data
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            response = requests.get(
                f"{self.api_base_url}/monitoring/cache/stats", timeout=5
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def cleanup_memory(self) -> str:
        """Trigger memory cleanup"""
        try:
            response = requests.post(
                f"{self.api_base_url}/monitoring/cleanup", timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                freed_gb = result.get("gpu_memory_freed_gb", 0)
                return f"✅ Cleanup completed. Freed {freed_gb:.2f}GB GPU memory."
            else:
                return f"❌ Cleanup failed: HTTP {response.status_code}"
        except Exception as e:
            return f"❌ Cleanup failed: {str(e)}"

    def cleanup_cache(self) -> str:
        """Trigger cache cleanup"""
        try:
            response = requests.post(
                f"{self.api_base_url}/monitoring/cache/cleanup", timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                files_removed = result.get("files_before", 0) - result.get(
                    "files_after", 0
                )
                size_freed = result.get("size_freed_mb", 0)
                return f"✅ Cache cleanup completed. Removed {files_removed} files, freed {size_freed:.1f}MB."
            else:
                return f"❌ Cache cleanup failed: HTTP {response.status_code}"
        except Exception as e:
            return f"❌ Cache cleanup failed: {str(e)}"

    def refresh_health_status(self):
        """Refresh health status display"""
        health = self.get_system_health()

        if health.get("status") == "healthy":
            status_color = "🟢"
            status_text = "System Healthy"
        elif health.get("status") == "warning":
            status_color = "🟡"
            status_text = f"Warning: {', '.join(health.get('issues', []))}"
        else:
            status_color = "🔴"
            status_text = f"Error: {health.get('error', 'Unknown error')}"

        # Format health info
        memory_info = health.get("memory", {})
        cache_info = health.get("cache", {})

        health_html = f"""
        <div style="padding: 15px; border-radius: 8px; background: #f8f9fa;">
            <h3>{status_color} {status_text}</h3>
            <p><strong>Timestamp:</strong> {health.get('timestamp', 'N/A')}</p>

            <h4>Memory Usage</h4>
            <ul>
                <li>CPU: {memory_info.get('cpu_percent', 0):.1f}%</li>
                <li>RAM: {memory_info.get('memory_percent', 0):.1f}% ({memory_info.get('memory_available_gb', 0):.1f}GB available)</li>
                <li>GPU: {memory_info.get('gpu_allocated_gb', 0):.2f}GB / {memory_info.get('gpu_total_gb', 0):.1f}GB</li>
            </ul>

            <h4>Cache Status</h4>
            <ul>
                <li>Disk Cache: {cache_info.get('disk_cache_files', 0)} files, {cache_info.get('disk_cache_size_mb', 0):.1f}MB</li>
                <li>Redis: {'✅ Available' if cache_info.get('redis_available') else '❌ Unavailable'}</li>
            </ul>
        </div>
        """

        return health_html

    def update_metrics_chart(self):
        """Update real-time metrics chart"""
        metrics = self.get_detailed_metrics()

        if "error" not in metrics:
            # Add to history
            self.metrics_history.append(metrics)

            # Keep only recent data
            if len(self.metrics_history) > self.max_history_points:
                self.metrics_history = self.metrics_history[-self.max_history_points :]

        if not self.metrics_history:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        # Extract data for plotting
        timestamps = [m["timestamp"] for m in self.metrics_history]
        cpu_usage = [
            m.get("system", {}).get("cpu_percent", 0) for m in self.metrics_history
        ]
        memory_usage = [
            m.get("system", {}).get("memory_percent", 0) for m in self.metrics_history
        ]
        gpu_usage = [
            m.get("gpu", {}).get("memory_allocated", 0) for m in self.metrics_history
        ]

        # Create subplot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_usage, name="CPU %", line=dict(color="blue"))
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps, y=memory_usage, name="Memory %", line=dict(color="green")
            )
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=gpu_usage,
                name="GPU Memory GB",
                line=dict(color="red"),
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="Real-time System Metrics",
            xaxis_title="Time",
            yaxis=dict(title="Percentage", side="left"),
            yaxis2=dict(title="GPU Memory (GB)", side="right", overlaying="y"),
            height=400,
        )

        return fig

    def create_interface(self):
        """Create Gradio interface for performance monitoring"""

        with gr.Blocks(title="SagaForge Performance Monitor") as interface:
            gr.Markdown("# 🔧 SagaForge Performance Monitor")
            gr.Markdown("Real-time system monitoring and optimization tools")

            with gr.Tabs():
                # Health Status Tab
                with gr.Tab("🏥 System Health"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            health_display = gr.HTML(value=self.refresh_health_status())
                            refresh_btn = gr.Button(
                                "🔄 Refresh Status", variant="primary"
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("### Quick Actions")
                            cleanup_memory_btn = gr.Button(
                                "🧹 Cleanup Memory", variant="secondary"
                            )
                            cleanup_cache_btn = gr.Button(
                                "🗑️ Cleanup Cache", variant="secondary"
                            )

                            memory_result = gr.Textbox(
                                label="Memory Cleanup Result", interactive=False
                            )
                            cache_result = gr.Textbox(
                                label="Cache Cleanup Result", interactive=False
                            )

                # Real-time Metrics Tab
                with gr.Tab("📊 Real-time Metrics"):
                    metrics_chart = gr.Plot(value=self.update_metrics_chart())
                    auto_refresh = gr.Checkbox(label="Auto-refresh (5s)", value=False)

                    with gr.Row():
                        manual_refresh_btn = gr.Button("🔄 Manual Refresh")
                        clear_history_btn = gr.Button("🗑️ Clear History")

                # Cache Analytics Tab
                with gr.Tab("💾 Cache Analytics"):
                    cache_stats_display = gr.JSON(label="Cache Statistics")

                    def update_cache_stats():
                        return self.get_cache_stats()

                    cache_refresh_btn = gr.Button("🔄 Refresh Cache Stats")
                    cache_refresh_btn.click(
                        update_cache_stats, outputs=cache_stats_display
                    )

                # Export Tools Tab
                with gr.Tab("📁 Export Tools"):
                    gr.Markdown("### Story Export")

                    with gr.Row():
                        session_id_input = gr.Textbox(
                            label="Session ID",
                            placeholder="Enter session ID to export",
                            value="test_session",
                        )
                        format_dropdown = gr.Dropdown(
                            choices=["html", "json", "pdf", "archive"],
                            label="Export Format",
                            value="html",
                        )

                    with gr.Row():
                        include_images = gr.Checkbox(label="Include Images", value=True)
                        include_metadata = gr.Checkbox(
                            label="Include Metadata", value=True
                        )

                    export_btn = gr.Button("📤 Export Story", variant="primary")
                    export_result = gr.Textbox(label="Export Result", interactive=False)

                    def export_story(session_id, format_type, inc_images, inc_metadata):
                        try:
                            export_data = {
                                "session_id": session_id,
                                "format": format_type,
                                "include_images": inc_images,
                                "include_metadata": inc_metadata,
                            }

                            response = requests.post(
                                f"{self.api_base_url}/export/story",
                                json=export_data,
                                timeout=60,
                            )

                            if response.status_code == 200:
                                result = response.json()
                                return f"✅ Export completed: {result.get('export_path')} ({result.get('file_size_mb', 0):.1f}MB)"
                            else:
                                return f"❌ Export failed: HTTP {response.status_code}"
                        except Exception as e:
                            return f"❌ Export failed: {str(e)}"

            # Event handlers
            refresh_btn.click(self.refresh_health_status, outputs=health_display)
            cleanup_memory_btn.click(self.cleanup_memory, outputs=memory_result)
            cleanup_cache_btn.click(self.cleanup_cache, outputs=cache_result)

            manual_refresh_btn.click(self.update_metrics_chart, outputs=metrics_chart)
            clear_history_btn.click(
                lambda: self.metrics_history.clear() or self.update_metrics_chart(),
                outputs=metrics_chart,
            )

            export_btn.click(
                export_story,
                inputs=[
                    session_id_input,
                    format_dropdown,
                    include_images,
                    include_metadata,
                ],
                outputs=export_result,
            )

            # Auto-refresh functionality
            def auto_refresh_handler():
                if auto_refresh.value:
                    return self.update_metrics_chart()
                return gr.update()

            # Update every 5 seconds when auto-refresh is enabled
            interface.load(auto_refresh_handler, outputs=metrics_chart, every=5)

        return interface


def create_performance_monitor_app():
    """Create and launch performance monitor app"""
    monitor = PerformanceMonitor()
    interface = monitor.create_interface()
    return interface


if __name__ == "__main__":
    app = create_performance_monitor_app()
    app.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)
