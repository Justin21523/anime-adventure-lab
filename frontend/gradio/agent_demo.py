# frontend/gradio_app/agent_demo.py
"""
Gradio demo for Agent system
"""

import gradio as gr
import requests
import json
from typing import Dict, Any

# API configuration
API_BASE = "http://localhost:8000/api/v1"


def call_agent_api(endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Call agent API endpoint"""
    try:
        url = f"{API_BASE}/{endpoint}"
        if data:
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)

        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def list_available_tools():
    """List all available agent tools"""
    result = call_agent_api("agent/tools")

    if "error" in result:
        return f"Error: {result['error']}"

    tools_info = []
    for tool in result.get("tools", []):
        params = []
        for param_name, param_info in tool.get("parameters", {}).items():
            required = " (required)" if param_info.get("required") else " (optional)"
            default = (
                f" [default: {param_info.get('default')}]"
                if param_info.get("default") is not None
                else ""
            )
            params.append(
                f"  • {param_name}: {param_info.get('type', 'unknown')}{required}{default}"
            )

        tool_text = (
            f"**{tool['name']}**\n{tool['description']}\nParameters:\n"
            + "\n".join(params)
        )
        tools_info.append(tool_text)

    return "\n\n" + "\n\n".join(tools_info) if tools_info else "No tools available"


def call_tool_direct(tool_name: str, parameters_json: str):
    """Call a specific tool directly"""
    try:
        if not tool_name.strip():
            return "Please enter a tool name"

        # Parse parameters
        if parameters_json.strip():
            try:
                parameters = json.loads(parameters_json)
            except json.JSONDecodeError as e:
                return f"Invalid JSON in parameters: {e}"
        else:
            parameters = {}

        # Call API
        result = call_agent_api(
            "agent/call", {"tool_name": tool_name.strip(), "parameters": parameters}
        )

        if "error" in result:
            return f"Error: {result['error']}"

        # Format response
        output = []
        output.append(f"**Tool:** {result.get('tool_name', 'unknown')}")
        output.append(f"**Success:** {result.get('success', False)}")
        output.append(f"**Execution Time:** {result.get('execution_time_ms', 0):.1f}ms")

        if result.get("success"):
            output.append(
                f"**Result:**\n```json\n{json.dumps(result.get('result', {}), indent=2, ensure_ascii=False)}\n```"
            )
        else:
            output.append(f"**Error:** {result.get('error_message', 'Unknown error')}")

        return "\n\n".join(output)

    except Exception as e:
        return f"Error: {str(e)}"


def execute_agent_task(
    task_description: str, max_iterations: int, enable_reasoning: bool
):
    """Execute an agent task with automatic tool selection"""
    try:
        if not task_description.strip():
            return "Please enter a task description"

        # Call API
        result = call_agent_api(
            "agent/task",
            {
                "task_description": task_description.strip(),
                "parameters": {
                    "max_iterations": max_iterations,
                    "max_tools_per_iteration": 2,
                    "enable_chain_of_thought": enable_reasoning,
                },
            },
        )

        if "error" in result:
            return f"Error: {result['error']}"

        # Format response
        output = []
        output.append(f"**Task:** {result.get('task_description', 'unknown')}")
        output.append(f"**Success:** {result.get('success', False)}")
        output.append(f"**Execution Time:** {result.get('execution_time_ms', 0):.1f}ms")
        output.append(f"**Tools Used:** {', '.join(result.get('tools_used', []))}")
        output.append(f"**Steps Taken:** {result.get('steps_taken', 0)}")

        if result.get("success"):
            output.append(f"**Result:**\n{result.get('result', 'No result')}")
        else:
            output.append(f"**Error:** {result.get('error_message', 'Unknown error')}")

        # Show reasoning chain if available
        reasoning = result.get("reasoning_chain", [])
        if reasoning and enable_reasoning:
            output.append("**Reasoning Chain:**")
            for i, step in enumerate(reasoning, 1):
                output.append(f"{i}. {step}")

        return "\n\n".join(output)

    except Exception as e:
        return f"Error: {str(e)}"


def get_agent_status():
    """Get current agent status"""
    result = call_agent_api("agent/status")

    if "error" in result:
        return f"Error: {result['error']}"

    output = []
    output.append(f"**Status:** {result.get('status', 'unknown')}")
    output.append(f"**Tools Loaded:** {result.get('tools_loaded', 0)}")
    output.append(
        f"**Available Tools:** {', '.join(result.get('available_tools', []))}"
    )
    output.append(f"**Max Iterations:** {result.get('max_iterations', 'unknown')}")
    output.append(f"**Config File:** {result.get('config_file', 'unknown')}")

    return "\n\n".join(output)


# Create Gradio interface
with gr.Blocks(title="Agent System Demo", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🤖 Agent System Demo")
    gr.Markdown("Test the agent system's tool calling and task execution capabilities")

    with gr.Tabs():
        # Tools tab
        with gr.TabItem("🛠️ Available Tools"):
            gr.Markdown("### List all available agent tools and their parameters")

            tools_btn = gr.Button("Refresh Tool List", variant="primary")
            tools_output = gr.Markdown("")

            tools_btn.click(fn=list_available_tools, outputs=tools_output)

        # Direct tool call tab
        with gr.TabItem("🔧 Direct Tool Call"):
            gr.Markdown("### Call a specific tool directly")

            with gr.Row():
                with gr.Column(scale=1):
                    tool_name_input = gr.Textbox(
                        label="Tool Name", placeholder="calculator", value="calculator"
                    )
                    parameters_input = gr.Textbox(
                        label="Parameters (JSON)",
                        placeholder='{"expression": "2+2*3"}',
                        lines=3,
                        value='{"expression": "2+2*3"}',
                    )
                    call_btn = gr.Button("Call Tool", variant="primary")

                with gr.Column(scale=2):
                    call_output = gr.Markdown("")

            call_btn.click(
                fn=call_tool_direct,
                inputs=[tool_name_input, parameters_input],
                outputs=call_output,
            )

        # Agent task tab
        with gr.TabItem("🎯 Agent Task"):
            gr.Markdown("### Execute complex tasks with automatic tool selection")

            with gr.Row():
                with gr.Column(scale=1):
                    task_input = gr.Textbox(
                        label="Task Description",
                        placeholder="Calculate the square root of 256 and tell me the result",
                        lines=3,
                        value="Calculate 15 * 23 + 47",
                    )
                    max_iter_input = gr.Slider(
                        label="Max Iterations", minimum=1, maximum=10, value=5, step=1
                    )
                    reasoning_input = gr.Checkbox(
                        label="Enable Chain of Thought", value=True
                    )
                    task_btn = gr.Button("Execute Task", variant="primary")

                with gr.Column(scale=2):
                    task_output = gr.Markdown("")

            task_btn.click(
                fn=execute_agent_task,
                inputs=[task_input, max_iter_input, reasoning_input],
                outputs=task_output,
            )

        # Status tab
        with gr.TabItem("📊 Agent Status"):
            gr.Markdown("### Current agent system status")

            status_btn = gr.Button("Get Status", variant="primary")
            status_output = gr.Markdown("")

            status_btn.click(fn=get_agent_status, outputs=status_output)

    # Example usage
    gr.Markdown(
        """
    ### 💡 Example Usage:

    **Calculator Tool:**
    - Tool: `calculator`
    - Parameters: `{"expression": "sqrt(144) + 5*3"}`

    **Web Search Tool:**
    - Tool: `web_search`
    - Parameters: `{"query": "Python machine learning", "max_results": 3}`

    **File Operations Tool:**
    - Tool: `file_ops`
    - Parameters: `{"operation": "list", "path": "."}`

    **Agent Tasks:**
    - "Calculate the area of a circle with radius 5"
    - "Search for information about artificial intelligence"
    - "List all files in the current directory"
    """
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7861, share=False)
