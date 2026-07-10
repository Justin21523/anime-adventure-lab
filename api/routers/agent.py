# api/routers/agent.py
"""
Agent Tools Router.

Provides endpoints for:
- listing and executing tools
- running simple and advanced reasoning agents
- managing multi-step tasks
- story-related agents
- health and system info
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from core.config import get_config
from core.exceptions import ValidationError
from core.agents import (
    ToolRegistry,
    AgentExecutor,
    MultiStepProcessor,
    SimpleReasoningAgent,
    AdvancedReasoningAgent,
)
from core.agents.story_integration import StoryAgentManager, StoryContext
from schemas.agent import (
    AgentParameters,
    AgentToolCallRequest,
    AgentToolCallResponse,
    AgentTaskRequest,
    AgentTaskResponse,
    AgentToolListResponse,
    AgentCatalogResponse,
    AgentStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances (lazy-initialized singletons)
_tool_registry: Optional[ToolRegistry] = None
_agent_executor: Optional[AgentExecutor] = None
_agent_instance: Optional[SimpleReasoningAgent] = None
_advanced_agent_instance: Optional[AdvancedReasoningAgent] = None
_multi_step_processor: Optional[MultiStepProcessor] = None
_story_manager: Optional[StoryAgentManager] = None


def _require_generic_tool_api() -> None:
    enabled = os.getenv("AGENT_ENABLE_GENERIC_TOOL_API", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        raise HTTPException(
            status_code=403,
            detail="Generic agent tool execution is disabled; use domain-specific agent endpoints",
        )


# --------------------------------------------------------------------------- #
# Helpers for configuration / singletons
# --------------------------------------------------------------------------- #


def _get_agent_settings() -> Dict[str, Any]:
    """Load agent defaults from config with safe fallbacks."""
    try:
        agent_config = get_config().get_agent_config() or {}
        settings = agent_config.get("agent_settings", {})
        return {
            "max_iterations": settings.get("default_max_iterations", 5),
            "max_tools_per_iteration": settings.get(
                "default_max_tools_per_iteration", 2
            ),
            "enable_reasoning": settings.get("enable_chain_of_thought", True),
            "config_path": str(Path(get_config().config_dir) / "agent.yaml"),
        }
    except Exception:  # noqa: BLE001
        # Ultimate fallback if config is unavailable for any reason.
        return {
            "max_iterations": 5,
            "max_tools_per_iteration": 2,
            "enable_reasoning": True,
            "config_path": "configs/agent.yaml",
        }


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def get_agent_executor() -> AgentExecutor:
    """Get global agent executor instance."""
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = AgentExecutor()
    return _agent_executor


def get_agent() -> SimpleReasoningAgent:
    """Get or create the default reasoning agent instance."""
    global _agent_instance
    if _agent_instance is None:
        settings = _get_agent_settings()
        _agent_instance = SimpleReasoningAgent(
            name="default_agent",
            description="General purpose reasoning agent",
            max_iterations=settings["max_iterations"],
            max_tools_per_iteration=settings["max_tools_per_iteration"],
            enable_reasoning=settings["enable_reasoning"],
        )
    return _agent_instance


def get_advanced_agent() -> AdvancedReasoningAgent:
    """Get or create the advanced reasoning agent instance."""
    global _advanced_agent_instance
    if _advanced_agent_instance is None:
        settings = _get_agent_settings()
        _advanced_agent_instance = AdvancedReasoningAgent(
            max_iterations=settings["max_iterations"],
            max_tools_per_iteration=settings["max_tools_per_iteration"],
            enable_reasoning=settings["enable_reasoning"],
        )
    return _advanced_agent_instance


def get_multi_step_processor() -> MultiStepProcessor:
    """Get global multi-step processor instance."""
    global _multi_step_processor
    if _multi_step_processor is None:
        _multi_step_processor = MultiStepProcessor()
    return _multi_step_processor


def get_story_manager() -> StoryAgentManager:
    """Get global story manager instance."""
    global _story_manager
    if _story_manager is None:
        _story_manager = StoryAgentManager()
    return _story_manager


def _parameters_to_dict(params: AgentParameters) -> Dict[str, Any]:
    """Convert AgentParameters to a plain dict (Pydantic v1/v2 compatible)."""
    if hasattr(params, "model_dump"):
        return params.model_dump()
    if hasattr(params, "dict"):
        return params.dict()
    # Very defensive fallback; not expected in normal operation.
    return dict(params)  # type: ignore[arg-type]


def _apply_agent_parameters(
    agent: SimpleReasoningAgent, params: Optional[AgentParameters]
) -> Dict[str, Any]:
    """
    Apply AgentParameters to an agent instance (max_iterations etc.).
    Returns a plain dict representation of the parameters for logging/response.
    """
    if not params:
        return {}

    try:
        if getattr(params, "max_iterations", None) is not None:
            agent.max_iterations = params.max_iterations  # type: ignore[assignment]
        if getattr(params, "max_tools_per_iteration", None) is not None:
            agent.max_tools_per_iteration = (  # type: ignore[assignment]
                params.max_tools_per_iteration
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to apply AgentParameters to agent: %s", exc)

    return _parameters_to_dict(params)


async def _run_agent_task(
    agent: SimpleReasoningAgent,
    request: AgentTaskRequest,
    agent_label: str,
) -> AgentTaskResponse:
    """
    Shared implementation for executing a task with either the default or
    advanced agent.
    """
    try:
        params_dict = _apply_agent_parameters(agent, request.parameters)

        enable_cot = True
        if request.parameters is not None and hasattr(
            request.parameters, "enable_chain_of_thought"
        ):
            enable_cot = bool(request.parameters.enable_chain_of_thought)

        response = await agent.execute_task(
            task_description=request.task_description,
            parameters=params_dict or None,
            enable_chain_of_thought=enable_cot,
        )

        return AgentTaskResponse(
            task_description=request.task_description,
            result=str(response.result) if response.success else "Task failed",
            success=response.success,
            tools_used=response.tools_used,
            steps_taken=len(response.tools_used),
            execution_time_ms=response.execution_time_ms,
            reasoning_chain=response.reasoning_chain,
            error_message=response.error_message,
            parameters=params_dict,
        )
    except HTTPException:
        # Just bubble it up if it was intentionally raised.
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("%s task failed: %s", agent_label, exc)
        raise HTTPException(500, f"{agent_label} task failed: {str(exc)}") from exc


# --------------------------------------------------------------------------- #
# Tool endpoints
# --------------------------------------------------------------------------- #


@router.get("/agent/catalog", response_model=AgentCatalogResponse)
async def get_agent_catalog():
    """Return Story orchestrator catalog (sub-agents/tools/default profile) for UI."""
    try:
        from core.agents.catalog import get_story_agent_catalog

        story = get_story_agent_catalog()
        return AgentCatalogResponse(  # type: ignore[call-arg]
            success=True,
            story=story,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to build agent catalog: %s", exc)
        raise HTTPException(500, f"Failed to build agent catalog: {str(exc)}") from exc


@router.get("/agent/tools", response_model=AgentToolListResponse)
async def list_tools():
    """List all available agent tools from the default agent's registry."""
    try:
        agent = get_agent()
        tools = []

        for tool_name in agent.tool_registry.list_tools():
            tool = agent.tool_registry.get_tool(tool_name)
            if not tool:
                continue

            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    name: {
                        "type": getattr(
                            param,
                            "type",
                            param.get("type") if isinstance(param, dict) else None,
                        ),
                        "description": getattr(
                            param,
                            "description",
                            param.get("description")
                            if isinstance(param, dict)
                            else None,
                        ),
                        "required": getattr(
                            param,
                            "required",
                            param.get("required")
                            if isinstance(param, dict)
                            else True,
                        ),
                        "default": getattr(
                            param,
                            "default",
                            param.get("default") if isinstance(param, dict) else None,
                        ),
                    }
                    for name, param in tool.parameters.items()
                },
                "timeout_seconds": tool.timeout_seconds,
            }
            tools.append(tool_info)

        return AgentToolListResponse(tools=tools, total_count=len(tools))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to list tools: %s", exc)
        raise HTTPException(500, f"Failed to list tools: {str(exc)}") from exc


@router.post("/agent/tools/list")
async def list_available_tools():
    """List all available agent tools directly from the global tool registry."""
    try:
        registry = get_tool_registry()
        tools_info = registry.get_all_tools_info()

        return {"success": True, "tools_count": len(tools_info), "tools": tools_info}
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to list tools: %s", exc)
        raise HTTPException(500, f"Failed to list tools: {str(exc)}") from exc


@router.post("/agent/tools/call", response_model=AgentToolCallResponse)
async def call_tool(request: AgentToolCallRequest):
    """Execute a single tool call."""
    _require_generic_tool_api()
    try:
        registry = get_tool_registry()
        executor = get_agent_executor()

        if not registry.is_tool_available(request.tool_name):
            raise HTTPException(404, f"Tool '{request.tool_name}' not found")

        result = await executor.execute_tool(
            request.tool_name, request.parameters or {}
        )

        return AgentToolCallResponse(
            success=result.success,
            tool_name=request.tool_name,
            result=result.result if result.success else {"error": result.error},
            execution_time_ms=result.execution_time_ms,
            parameters=request.parameters,
            error_message=None if result.success else result.error,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Tool call failed: %s", exc)
        raise HTTPException(500, f"Tool call failed: {str(exc)}") from exc


@router.post("/agent/tools/batch")
async def call_multiple_tools(tool_calls: List[Dict[str, Any]]):
    """Execute multiple tools in parallel."""
    _require_generic_tool_api()
    try:
        executor = get_agent_executor()
        registry = get_tool_registry()

        # Validate all tools exist
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            if not tool_name:
                raise ValidationError("tool_name", None, "Tool name required")

            if not registry.is_tool_available(tool_name):
                raise ValidationError("tool_name", tool_name, "Tool not found")

        results = await executor.execute_multiple_tools(
            tool_calls=tool_calls, max_concurrent=3
        )

        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "tool_name": result.tool_name,
                    "success": result.success,
                    "result": (
                        result.result if result.success else {"error": result.error}
                    ),
                    "execution_time_ms": result.execution_time_ms,
                    "parameters": result.parameters or {},
                }
            )

        return {
            "success": True,
            "results_count": len(formatted_results),
            "results": formatted_results,
        }
    except ValidationError as exc:
        logger.error("Batch tool call validation failed: %s", exc)
        raise HTTPException(400, f"Validation error: {str(exc)}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Batch tool call failed: %s", exc)
        raise HTTPException(500, f"Batch tool call failed: {str(exc)}") from exc


# --------------------------------------------------------------------------- #
# Agent task endpoints
# --------------------------------------------------------------------------- #


@router.post("/agent/task", response_model=AgentTaskResponse)
async def execute_task(request: AgentTaskRequest):
    """Execute a complex task with the default reasoning agent."""
    agent = get_agent()
    return await _run_agent_task(agent, request, agent_label="Agent")


@router.post("/agent/advanced/task", response_model=AgentTaskResponse)
async def execute_advanced_task(request: AgentTaskRequest):
    """Execute a task using the advanced reasoning agent (with reflection & QA)."""
    agent = get_advanced_agent()
    return await _run_agent_task(agent, request, agent_label="Advanced agent")


@router.get("/agent/status", response_model=AgentStatusResponse)
async def get_agent_status():
    """Get current status and configuration of the default agent."""
    try:
        agent = get_agent()
        settings = _get_agent_settings()

        return AgentStatusResponse(  # type: ignore[call-arg]
            status="active",
            tools_loaded=len(agent.tool_registry.list_tools()),
            available_tools=agent.tool_registry.list_tools(),
            max_iterations=agent.max_iterations,
            max_tools_per_iteration=agent.max_tools_per_iteration,
            config_file=settings.get("config_path", "configs/agent.yaml"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to get agent status: %s", exc)
        raise HTTPException(500, f"Failed to get agent status: {str(exc)}") from exc


@router.get("/agent/advanced/status", response_model=AgentStatusResponse)
async def get_advanced_agent_status():
    """Get current status and configuration of the advanced agent."""
    try:
        agent = get_advanced_agent()
        settings = _get_agent_settings()

        return AgentStatusResponse(  # type: ignore[call-arg]
            status="active",
            tools_loaded=len(agent.tool_registry.list_tools()),
            available_tools=agent.tool_registry.list_tools(),
            max_iterations=agent.max_iterations,
            max_tools_per_iteration=agent.max_tools_per_iteration,
            config_file=settings.get("config_path", "configs/agent.yaml"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to get advanced agent status: %s", exc)
        raise HTTPException(
            500, f"Failed to get advanced agent status: {str(exc)}"
        ) from exc


@router.post("/agent/reload")
async def reload_agent_tools():
    """Reload agent tools from configuration (recreate default & advanced agents)."""
    try:
        global _agent_instance, _advanced_agent_instance
        _agent_instance = None
        _advanced_agent_instance = None

        agent = get_agent()

        return {
            "message": "Agent tools reloaded successfully",
            "tools_loaded": len(agent.tool_registry.list_tools()),
            "available_tools": agent.tool_registry.list_tools(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to reload agent: %s", exc)
        raise HTTPException(500, f"Failed to reload agent: {str(exc)}") from exc


# --------------------------------------------------------------------------- #
# Multi-step task endpoints
# --------------------------------------------------------------------------- #


@router.post("/agent/multistep/create")
async def create_multistep_task(
    task_id: str, description: str, context: Optional[Dict[str, Any]] = None
):
    """Create a new multi-step task."""
    try:
        processor = get_multi_step_processor()

        task = await processor.create_task(
            task_id=task_id, description=description, context=context
        )

        return {
            "success": True,
            "task_id": task.task_id,
            "description": task.description,
            "status": task.status.value,
            "message": f"Multi-step task '{task_id}' created successfully",
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Multi-step task creation failed: %s", exc)
        raise HTTPException(500, f"Task creation failed: {str(exc)}") from exc


@router.post("/agent/multistep/plan")
async def auto_plan_task(task_id: str, planning_agent: str = "reasoning"):
    """Automatically plan steps for a multi-step task."""
    try:
        processor = get_multi_step_processor()

        if task_id not in processor.active_tasks:
            raise HTTPException(404, f"Task '{task_id}' not found")

        task = processor.active_tasks[task_id]
        await processor.auto_plan_task(task, planning_agent)

        return {
            "success": True,
            "task_id": task_id,
            "steps_planned": len(task.steps),
            "steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "agent": step.agent_name,
                    "dependencies": step.dependencies,
                }
                for step in task.steps
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Task planning failed: %s", exc)
        raise HTTPException(500, f"Task planning failed: {str(exc)}") from exc


@router.post("/agent/multistep/execute")
async def execute_multistep_task(
    task_id: str, background_tasks: BackgroundTasks, max_parallel_steps: int = 2
):
    """Execute a multi-step task."""
    try:
        processor = get_multi_step_processor()

        if task_id not in processor.active_tasks:
            raise HTTPException(404, f"Task '{task_id}' not found")

        # Execute task
        result = await processor.execute_task(
            task_id=task_id, max_parallel_steps=max_parallel_steps
        )

        return {"success": True, "execution_result": result}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Multi-step task execution failed: %s", exc)
        raise HTTPException(500, f"Task execution failed: {str(exc)}") from exc


@router.get("/agent/multistep/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a multi-step task."""
    try:
        processor = get_multi_step_processor()
        status = processor.get_task_status(task_id)

        if status is None:
            raise HTTPException(404, f"Task '{task_id}' not found")

        return {"success": True, "task_status": status}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to get task status: %s", exc)
        raise HTTPException(500, f"Failed to get task status: {str(exc)}") from exc


@router.get("/agent/multistep/list")
async def list_active_tasks():
    """List all active multi-step tasks."""
    try:
        processor = get_multi_step_processor()
        active_tasks = processor.list_active_tasks()

        return {
            "success": True,
            "active_tasks": active_tasks,
            "count": len(active_tasks),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to list active tasks: %s", exc)
        raise HTTPException(500, f"Failed to list active tasks: {str(exc)}") from exc


@router.delete("/agent/multistep/cancel/{task_id}")
async def cancel_multistep_task(task_id: str):
    """Cancel an active multi-step task."""
    try:
        processor = get_multi_step_processor()
        success = processor.cancel_task(task_id)

        if success:
            return {
                "success": True,
                "message": f"Task '{task_id}' cancelled successfully",
            }

        raise HTTPException(404, f"Task '{task_id}' not found")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Task cancellation failed: %s", exc)
        raise HTTPException(500, f"Task cancellation failed: {str(exc)}") from exc


# --------------------------------------------------------------------------- #
# Story-related endpoints
# --------------------------------------------------------------------------- #


@router.post("/agent/story/action")
async def process_story_action(
    story_id: str,
    character_name: str,
    current_scene: str,
    player_action: str,
    character_state: Optional[Dict[str, Any]] = None,
    story_history: Optional[List[Dict[str, Any]]] = None,
    available_actions: Optional[List[str]] = None,
    narrative_style: str = "adventure",
):
    """Process a player action within a story context."""
    try:
        story_manager = get_story_manager()

        story_context = StoryContext(
            story_id=story_id,
            character_name=character_name,
            current_scene=current_scene,
            character_state=character_state or {},
            story_history=story_history or [],
            available_actions=available_actions or [],
            narrative_style=narrative_style,
        )

        story_agent = story_manager.get_story_agent("narrative")
        if not story_agent:
            raise HTTPException(500, "Story agent not available")

        result = await story_agent.process_story_action(
            story_context=story_context, player_action=player_action
        )

        return {
            "success": result["success"],
            "story_response": result.get("story_response"),
            "error": result.get("error"),
            "fallback_response": result.get("fallback_response"),
            "agent_info": {
                "steps_taken": result.get("agent_steps", 0),
                "tools_used": result.get("tools_used", []),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Story action processing failed: %s", exc)
        raise HTTPException(500, f"Story action processing failed: {str(exc)}") from exc


@router.post("/agent/story/scene")
async def generate_scene_description(
    story_id: str,
    character_name: str,
    current_scene: str,
    character_state: Optional[Dict[str, Any]] = None,
    narrative_style: str = "adventure",
    scene_type: str = "descriptive",
):
    """Generate a detailed scene description."""
    try:
        story_manager = get_story_manager()

        story_context = StoryContext(
            story_id=story_id,
            character_name=character_name,
            current_scene=current_scene,
            character_state=character_state or {},
            story_history=[],
            available_actions=[],
            narrative_style=narrative_style,
        )

        story_agent = story_manager.get_story_agent("world")
        if not story_agent:
            raise HTTPException(500, "World building agent not available")

        result = await story_agent.generate_scene_description(
            story_context=story_context, scene_type=scene_type
        )

        return {
            "success": result["success"],
            "scene_description": result.get("scene_description"),
            "scene_type": result.get("scene_type"),
            "generated_for": result.get("generated_for"),
            "error": result.get("error"),
            "fallback_description": result.get("fallback_description"),
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Scene generation failed: %s", exc)
        raise HTTPException(500, f"Scene generation failed: {str(exc)}") from exc


@router.post("/agent/story/complex")
async def process_complex_story_scenario(
    story_id: str,
    character_name: str,
    current_scene: str,
    scenario_type: str,
    scenario_data: Dict[str, Any],
    character_state: Optional[Dict[str, Any]] = None,
    story_history: Optional[List[Dict[str, Any]]] = None,
    narrative_style: str = "adventure",
):
    """Process complex story scenarios using multiple agents."""
    try:
        story_manager = get_story_manager()

        story_context = StoryContext(
            story_id=story_id,
            character_name=character_name,
            current_scene=current_scene,
            character_state=character_state or {},
            story_history=story_history or [],
            available_actions=[],
            narrative_style=narrative_style,
        )

        result = await story_manager.process_complex_story_scenario(
            story_context=story_context,
            scenario_type=scenario_type,
            scenario_data=scenario_data,
        )

        return {
            "success": result["success"],
            "scenario_type": result["scenario_type"],
            "scenario_result": result.get("scenario_result"),
            "error": result.get("error"),
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Complex story scenario failed: %s", exc)
        raise HTTPException(500, f"Complex story scenario failed: {str(exc)}") from exc


@router.get("/agent/story/active")
async def list_active_stories():
    """List all active story contexts."""
    try:
        story_manager = get_story_manager()
        active_stories = story_manager.get_active_stories()

        return {
            "success": True,
            "active_stories": active_stories,
            "count": len(active_stories),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to list active stories: %s", exc)
        raise HTTPException(500, f"Failed to list active stories: {str(exc)}") from exc


# --------------------------------------------------------------------------- #
# System info / health endpoints
# --------------------------------------------------------------------------- #


@router.get("/agent/info")
async def get_agent_system_info():
    """Get agent system information and statistics."""
    try:
        registry = get_tool_registry()
        processor = get_multi_step_processor()
        executor = get_agent_executor()

        return {
            "success": True,
            "system_info": {
                "available_tools": len(registry.list_tools()),
                "active_tasks": len(processor.list_active_tasks()),
                "active_executions": len(executor.get_active_executions()),
                "tool_categories": {},  # Reserved for future category breakdown
                "agent_types": [
                    "reasoning",
                    "advanced_reasoning",
                    "narrative",
                    "dialogue",
                    "world",
                    "action",
                ],
            },
            "tool_list": registry.list_tools(),
            "active_tasks": processor.list_active_tasks(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to get agent info: %s", exc)
        raise HTTPException(500, f"Failed to get agent info: {str(exc)}") from exc


@router.post("/agent/tools/configure")
async def configure_tool_settings(tool_name: str, settings: Dict[str, Any]):
    """Configure tool-specific settings (currently only web_search)."""
    try:
        if tool_name == "web_search":
            from core.agents.tools.web_search import configure_search_engine

            result = configure_search_engine(**settings)
            return result

        return {
            "success": False,
            "error": f"Configuration not supported for tool: {tool_name}",
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Tool configuration failed: %s", exc)
        raise HTTPException(500, f"Tool configuration failed: {str(exc)}") from exc


@router.get("/agent/health")
async def agent_health_check():
    """Health check for agent system components."""
    try:
        registry = get_tool_registry()
        executor = get_agent_executor()
        tools = registry.list_tools()

        health_status: Dict[str, Any] = {
            "tool_registry": len(tools) > 0,
            "agent_executor": executor is not None,
            "available_tools": tools,
            "system_ready": True,
        }

        # Test a simple tool execution if possible.
        tool_to_test: Optional[str] = None
        test_params: Dict[str, Any] = {}

        if registry.is_tool_available("calculator"):
            tool_to_test = "calculator"
            test_params = {"expression": "2+2"}
        elif registry.is_tool_available("basic_math"):
            tool_to_test = "basic_math"
            test_params = {"a": 2, "b": 2, "op": "+"}
        elif tools:
            tool_to_test = tools[0]

        if tool_to_test:
            try:
                # Prefer direct function call to avoid executor timeouts in health.
                tool_fn = registry.get_function(tool_to_test)
                validated = registry.validate_parameters(tool_to_test, test_params) or {}
                if tool_fn:
                    if asyncio.iscoroutinefunction(tool_fn):
                        tool_result = await tool_fn(**validated)
                    else:
                        tool_result = tool_fn(**validated)
                    health_status["tool_execution_test"] = True
                    health_status["tool_test_tool"] = tool_to_test
                    health_status["tool_test_result"] = tool_result
                else:
                    health_status["tool_execution_test"] = False
                    health_status["tool_test_tool"] = tool_to_test
                    health_status["tool_test_error"] = "Function not found"
            except Exception as exc:  # noqa: BLE001
                health_status["tool_execution_test"] = False
                health_status["tool_test_tool"] = tool_to_test
                health_status["tool_test_error"] = str(exc)
        else:
            health_status["tool_execution_test"] = False
            health_status["tool_test_error"] = "No tools registered"

        return {"success": True, "health": health_status, "timestamp": time.time()}
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent health check failed: %s", exc)
        return {"success": False, "error": str(exc), "timestamp": time.time()}
