# frontend/gradio/safety_admin.py
import gradio as gr
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import io

# API Base URL (should be configurable)
API_BASE = "http://localhost:8000"


class SafetyAdminDashboard:
    """Safety Administration Dashboard for monitoring and managing safety systems"""

    def __init__(self):
        self.api_base = API_BASE

    def check_api_health(self) -> Tuple[str, str]:
        """Check safety API health status"""
        try:
            response = requests.get(f"{self.api_base}/safety/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                components = data.get("components", {})
                models = data.get("models_loaded", {})

                health_info = f"Status: {status}\n\n"
                health_info += "Components:\n"
                for comp, state in components.items():
                    health_info += f"  • {comp}: {state}\n"

                health_info += "\nModels Loaded:\n"
                for model, loaded in models.items():
                    health_info += f"  • {model}: {'✓' if loaded else '✗'}\n"

                return health_info, "✅ Healthy" if status == "healthy" else "⚠️ Issues"
            else:
                return f"API Error: {response.status_code}", "❌ Unhealthy"

        except Exception as e:
            return f"Connection Error: {str(e)}", "❌ Offline"

    def get_audit_summary(self, days: int = 7) -> Tuple[str, str]:
        """Get audit summary for the last N days"""
        try:
            response = requests.get(f"{self.api_base}/safety/audit/summary?days={days}")
            if response.status_code == 200:
                data = response.json()

                summary = f"Audit Summary (Last {days} days)\n\n"
                summary += f"Total Events: {data.get('total_events', 0)}\n"
                summary += f"File Uploads: {data.get('uploads', 0)}\n"
                summary += f"Generations: {data.get('generations', 0)}\n"
                summary += f"Safety Violations: {data.get('safety_violations', 0)}\n\n"

                # License types breakdown
                license_types = data.get("license_types", {})
                if license_types:
                    summary += "License Types:\n"
                    for license_type, count in license_types.items():
                        summary += f"  • {license_type}: {count}\n"

                # Safety actions
                safety_actions = data.get("safety_actions", {})
                if safety_actions:
                    summary += "\nSafety Actions:\n"
                    for action, count in safety_actions.items():
                        summary += f"  • {action}: {count}\n"

                # Generate chart data
                chart_data = self._create_audit_chart(data)

                return summary, chart_data
            else:
                return f"Failed to fetch audit data: {response.status_code}", ""

        except Exception as e:
            return f"Error fetching audit data: {str(e)}", ""

    def _create_audit_chart(self, data: Dict[str, Any]) -> str:
        """Create audit summary chart"""
        try:
            plt.figure(figsize=(12, 8))

            # Create subplot layout
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            # 1. Events overview (pie chart)
            events = [
                data.get("uploads", 0),
                data.get("generations", 0),
                data.get("safety_violations", 0),
            ]
            labels = ["Uploads", "Generations", "Safety Violations"]
            colors = ["#2E8B57", "#4682B4", "#DC143C"]

            ax1.pie(
                events, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
            )
            ax1.set_title("Event Distribution")

            # 2. License types (bar chart)
            license_types = data.get("license_types", {})
            if license_types:
                licenses = list(license_types.keys())
                counts = list(license_types.values())
                ax2.bar(licenses, counts, color="#4682B4")
                ax2.set_title("License Types")
                ax2.set_xlabel("License Type")
                ax2.set_ylabel("Count")
                plt.setp(ax2.get_xticklabels(), rotation=45)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No license data",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
                ax2.set_title("License Types")

            # 3. Safety actions (bar chart)
            safety_actions = data.get("safety_actions", {})
            if safety_actions:
                actions = list(safety_actions.keys())
                action_counts = list(safety_actions.values())
                ax3.bar(actions, action_counts, color="#DC143C")
                ax3.set_title("Safety Actions")
                ax3.set_xlabel("Action Type")
                ax3.set_ylabel("Count")
                plt.setp(ax3.get_xticklabels(), rotation=45)
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No safety actions",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                )
                ax3.set_title("Safety Actions")

            # 4. Summary metrics (text)
            ax4.axis("off")
            summary_text = f"""
            Summary Metrics

            Total Events: {data.get('total_events', 0)}
            Period: {data.get('period_days', 7)} days

            Safety Rate: {((data.get('total_events', 1) - data.get('safety_violations', 0)) / max(data.get('total_events', 1), 1) * 100):.1f}%

            Most Common License: {max(license_types.keys(), key=license_types.get) if license_types else 'N/A'}
            """
            ax4.text(
                0.1,
                0.9,
                summary_text,
                transform=ax4.transAxes,
                fontsize=10,
                verticalalignment="top",
            )

            plt.tight_layout()

            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            plt.close()

            return img_buffer.getvalue()

        except Exception as e:
            print(f"Chart creation error: {e}")
            return ""

    def check_prompt_safety(
        self, prompt: str, check_nsfw: bool, check_injection: bool
    ) -> Tuple[str, str]:
        """Check prompt safety"""
        try:
            response = requests.post(
                f"{self.api_base}/safety/check/prompt",
                json={
                    "prompt": prompt,
                    "check_nsfw": check_nsfw,
                    "check_injection": check_injection,
                },
            )

            if response.status_code == 200:
                data = response.json()

                result = f"Safety Check Results\n\n"
                result += f"Overall Safe: {'✅ Yes' if data.get('is_safe', False) else '❌ No'}\n\n"

                prompt_check = data.get("prompt_check", {})
                result += f"Clean Prompt: {prompt_check.get('clean_prompt', 'N/A')}\n"
                result += f"Original Length: {prompt_check.get('original_length', 0)}\n"
                result += f"Cleaned Length: {prompt_check.get('cleaned_length', 0)}\n\n"

                warnings = data.get("warnings", [])
                if warnings:
                    result += "Warnings:\n"
                    for warning in warnings:
                        result += f"  • {warning}\n"

                actions = data.get("actions_taken", [])
                if actions:
                    result += "\nActions Taken:\n"
                    for action in actions:
                        result += f"  • {action}\n"

                status = (
                    "✅ Safe" if data.get("is_safe", False) else "⚠️ Potentially Unsafe"
                )
                return result, status
            else:
                return f"API Error: {response.status_code}", "❌ Error"

        except Exception as e:
            return f"Connection Error: {str(e)}", "❌ Error"

    def get_license_info(self, file_id: str) -> str:
        """Get license information for a file"""
        try:
            response = requests.get(f"{self.api_base}/safety/license/{file_id}")

            if response.status_code == 200:
                data = response.json()
                license_info = data.get("license_info", {})

                info = f"License Information for File: {file_id}\n\n"
                info += f"License Type: {license_info.get('license_type', 'Unknown')}\n"
                info += f"Attribution Required: {'Yes' if license_info.get('attribution_required', False) else 'No'}\n"
                info += f"Commercial Use: {'Allowed' if license_info.get('commercial_use', False) else 'Prohibited'}\n"
                info += f"Derivative Works: {'Allowed' if license_info.get('derivative_works', False) else 'Prohibited'}\n"
                info += f"Share Alike: {'Required' if license_info.get('share_alike', False) else 'Not Required'}\n\n"

                if license_info.get("author"):
                    info += f"Author: {license_info.get('author')}\n"
                if license_info.get("source_url"):
                    info += f"Source: {license_info.get('source_url')}\n"

                attribution = data.get("attribution_text", "")
                if attribution:
                    info += f"\nAttribution Text:\n{attribution}\n"

                return info
            elif response.status_code == 404:
                return "File not found in license database"
            else:
                return f"API Error: {response.status_code}"

        except Exception as e:
            return f"Connection Error: {str(e)}"

    def list_supported_licenses(self) -> str:
        """List all supported license types"""
        try:
            response = requests.get(f"{self.api_base}/safety/licenses/list")

            if response.status_code == 200:
                data = response.json()
                licenses = data.get("supported_licenses", {})

                info = "Supported License Types\n\n"
                for license_type, props in licenses.items():
                    info += f"{license_type}: {props.get('full_name', 'Unknown')}\n"
                    info += f"  • Attribution Required: {'Yes' if props.get('attribution_required', False) else 'No'}\n"
                    info += f"  • Commercial Use: {'Allowed' if props.get('commercial_use', False) else 'Prohibited'}\n"
                    info += f"  • Derivative Works: {'Allowed' if props.get('derivative_works', False) else 'Prohibited'}\n"
                    info += f"  • Share Alike: {'Required' if props.get('share_alike', False) else 'Not Required'}\n"

                    if "url" in props:
                        info += f"  • More Info: {props['url']}\n"

                    info += "\n"

                return info
            else:
                return f"API Error: {response.status_code}"

        except Exception as e:
            return f"Connection Error: {str(e)}"


def create_safety_admin_interface():
    """Create Gradio interface for safety administration"""

    dashboard = SafetyAdminDashboard()

    with gr.Blocks(
        title="SagaForge Safety Administration", theme=gr.themes.Soft()
    ) as interface:

        gr.Markdown("# 🛡️ SagaForge Safety Administration Dashboard")
        gr.Markdown("Monitor and manage safety systems, licensing, and compliance.")

        with gr.Tabs():
            # System Health Tab
            with gr.Tab("🔍 System Health"):
                gr.Markdown("## System Health Monitoring")

                with gr.Row():
                    health_check_btn = gr.Button(
                        "Check System Health", variant="primary"
                    )
                    health_status = gr.Textbox(label="Status", interactive=False)

                health_details = gr.Textbox(
                    label="Health Details", lines=15, interactive=False
                )

                health_check_btn.click(
                    dashboard.check_api_health, outputs=[health_details, health_status]
                )

            # Audit Summary Tab
            with gr.Tab("📊 Audit Summary"):
                gr.Markdown("## Compliance Audit Summary")

                with gr.Row():
                    audit_days = gr.Slider(
                        minimum=1, maximum=30, value=7, step=1, label="Days to Review"
                    )
                    audit_btn = gr.Button("Generate Audit Report", variant="primary")

                with gr.Row():
                    audit_summary = gr.Textbox(
                        label="Audit Summary", lines=20, interactive=False
                    )
                    audit_chart = gr.Image(label="Audit Visualization", type="numpy")

                audit_btn.click(
                    dashboard.get_audit_summary,
                    inputs=[audit_days],
                    outputs=[audit_summary, audit_chart],
                )

            # Prompt Safety Tab
            with gr.Tab("✅ Prompt Safety"):
                gr.Markdown("## Prompt Safety Testing")

                with gr.Row():
                    with gr.Column():
                        test_prompt = gr.Textbox(
                            label="Prompt to Test",
                            lines=3,
                            placeholder="Enter a prompt to test for safety...",
                        )

                        with gr.Row():
                            check_nsfw = gr.Checkbox(label="Check NSFW", value=True)
                            check_injection = gr.Checkbox(
                                label="Check Injection", value=True
                            )

                        safety_check_btn = gr.Button("Check Safety", variant="primary")

                    with gr.Column():
                        safety_status = gr.Textbox(
                            label="Safety Status", interactive=False
                        )
                        safety_details = gr.Textbox(
                            label="Safety Check Results", lines=15, interactive=False
                        )

                safety_check_btn.click(
                    dashboard.check_prompt_safety,
                    inputs=[test_prompt, check_nsfw, check_injection],
                    outputs=[safety_details, safety_status],
                )

            # License Management Tab
            with gr.Tab("📄 License Management"):
                gr.Markdown("## License Information & Management")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### File License Lookup")
                        file_id_input = gr.Textbox(
                            label="File ID",
                            placeholder="Enter file ID to lookup license...",
                        )
                        lookup_btn = gr.Button("Lookup License", variant="primary")

                        license_details = gr.Textbox(
                            label="License Details", lines=15, interactive=False
                        )

                    with gr.Column():
                        gr.Markdown("### Supported Licenses")
                        list_licenses_btn = gr.Button(
                            "List Supported Licenses", variant="secondary"
                        )

                        supported_licenses = gr.Textbox(
                            label="Supported License Types", lines=15, interactive=False
                        )

                lookup_btn.click(
                    dashboard.get_license_info,
                    inputs=[file_id_input],
                    outputs=[license_details],
                )

                list_licenses_btn.click(
                    dashboard.list_supported_licenses, outputs=[supported_licenses]
                )

            # Configuration Tab
            with gr.Tab("⚙️ Configuration"):
                gr.Markdown("## Safety System Configuration")

                gr.Markdown("### API Configuration")
                api_base_input = gr.Textbox(
                    label="API Base URL",
                    value=API_BASE,
                    placeholder="http://localhost:8000",
                )

                gr.Markdown("### Safety Thresholds")
                nsfw_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.85,
                    step=0.05,
                    label="NSFW Detection Threshold",
                )

                face_blur_enable = gr.Checkbox(label="Auto-blur Faces", value=False)

                strict_prompt_filter = gr.Checkbox(
                    label="Strict Prompt Filtering", value=True
                )

                save_config_btn = gr.Button("Save Configuration", variant="primary")
                config_status = gr.Textbox(
                    label="Configuration Status", interactive=False
                )

                def save_configuration(api_base, nsfw_thresh, face_blur, strict_filter):
                    # In a real implementation, this would save to config file
                    return "Configuration saved successfully (demo mode)"

                save_config_btn.click(
                    save_configuration,
                    inputs=[
                        api_base_input,
                        nsfw_threshold,
                        face_blur_enable,
                        strict_prompt_filter,
                    ],
                    outputs=[config_status],
                )

        # Footer
        gr.Markdown("---")
        gr.Markdown(
            "💡 **Tips:** Use this dashboard to monitor safety compliance, check content safety, and manage licensing requirements."
        )

    return interface


if __name__ == "__main__":
    # Launch the safety admin dashboard
    interface = create_safety_admin_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from main app
        share=False,
        debug=True,
    )
