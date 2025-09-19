# api/routers/agent.py
"""
Agent Tools Router
"""

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
from schemas.agent import AgentParameters
import time
from core.exceptions import ValidationError, MultiModalLabError
from schemas.agent import (
    AgentToolCallRequest,
    AgentToolCallResponse,
    AgentTaskRequest,
    AgentTaskResponse,
    AgentToolListResponse,
    AgentStatusResponse,
)

# Import agent system components
from core.agents import (
    ToolRegistry,
    AgentExecutor,
    MultiStepProcessor,
    SimpleReasoningAgent,
)
from core.agents import BaseAgent
from core.agents.story_integration import StoryAgent, StoryAgentManager, StoryContext

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances
_tool_registry = None
_agent_executor = None
_agent_instance = None
_multi_step_processor = None
_story_manager = None


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry instance"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def get_agent_executor() -> AgentExecutor:
    """Get global agent executor instance"""
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = AgentExecutor()
    return _agent_executor


def get_agent() -> BaseAgent:
    """Get or create agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = BaseAgent()  # type: ignore
    return _agent_instance


def get_multi_step_processor() -> MultiStepProcessor:
    """Get global multi-step processor instance"""
    global _multi_step_processor
    if _multi_step_processor is None:
        _multi_step_processor = MultiStepProcessor()
    return _multi_step_processor


def get_story_manager() -> StoryAgentManager:
    """Get global story manager instance"""
    global _story_manager
    if _story_manager is None:
        _story_manager = StoryAgentManager()
    return _story_manager


@router.get("/agent/tools", response_model=AgentToolListResponse)
async def list_tools():
    """List all available agent tools"""
    try:
        agent = get_agent()
        tools = []

        for tool_name in agent.tool_registry.list_tools():
            tool = agent.tool_registry.get_tool(tool_name)
            if tool:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        name: {
                            "type": param.type,
                            "description": param.description,
                            "required": param.required,
                            "default": param.default,
                        }
                        for name, param in tool.parameters.items()
                    },
                    "timeout_seconds": tool.timeout_seconds,
                }
                tools.append(tool_info)

        return AgentToolListResponse(tools=tools, total_count=len(tools))

    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(500, f"Failed to list tools: {str(e)}")


@router.post("/agent/tools/list")
async def list_available_tools():
    """List all available agent tools"""
    try:
        registry = get_tool_registry()
        tools_info = registry.get_all_tools_info()

        return {"success": True, "tools_count": len(tools_info), "tools": tools_info}

    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(500, f"Failed to list tools: {str(e)}")


@router.post("/agent/tools/batch")
async def call_multiple_tools(tool_calls: List[Dict[str, Any]]):
    """Execute multiple tools in parallel"""
    try:
        executor = get_agent_executor()

        # Validate all tools exist
        registry = get_tool_registry()
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            if not tool_name:
                raise ValidationError("tool_name", None, "Tool name required")

            if not registry.is_tool_available(tool_name):
                raise ValidationError("tool_name", tool_name, "Tool not found")

        # Execute tools
        results = await executor.execute_multiple_tools(
            tool_calls=tool_calls, max_concurrent=3
        )

        # Format response
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

    except ValidationError as e:
        logger.error(f"Batch tool call validation failed: {e}")
        raise HTTPException(400, f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Batch tool call failed: {e}")
        raise HTTPException(500, f"Batch tool call failed: {str(e)}")


@router.post("/agent/task", response_model=AgentTaskResponse)
async def execute_task(request: AgentTaskRequest):
    """Execute a complex agent task with automatic tool selection"""
    try:
        agent = get_agent()

        # Set agent parameters if provided
        if request.parameters:
            agent.max_iterations = request.parameters.max_iterations
            agent.max_tools_per_iteration = request.parameters.max_tools_per_iteration

        # Execute task
        response = await agent.execute_task(
            task_description=request.task_description,
            parameters=request.parameters.dict() if request.parameters else None,
            enable_chain_of_thought=(
                request.parameters.enable_chain_of_thought
                if request.parameters
                else True
            ),
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
            parameters=request.parameters or {},
        )

    except Exception as e:
        logger.error(f"Agent task failed: {e}")
        raise HTTPException(500, f"Agent task failed: {str(e)}")


@router.get("/agent/status", response_model=AgentStatusResponse)
async def get_agent_status():
    """Get current agent status and configuration"""
    try:
        agent = get_agent()

        return AgentStatusResponse(  # type: ignore
            status="active",
            tools_loaded=len(agent.tool_registry.list_tools()),
            available_tools=agent.tool_registry.list_tools(),
            max_iterations=agent.max_iterations,
            max_tools_per_iteration=agent.max_tools_per_iteration,
            config_file=agent.tool_registry.config_path,
        )

    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(500, f"Failed to get agent status: {str(e)}")


@router.post("/agent/reload")
async def reload_agent_tools():
    """Reload agent tools from configuration"""
    try:
        global _agent_instance
        _agent_instance = None  # Force recreation

        # Get new agent instance (will reload tools)
        agent = get_agent()

        return {
            "message": "Agent tools reloaded successfully",
            "tools_loaded": len(agent.tool_registry.list_tools()),
            "available_tools": agent.tool_registry.list_tools(),
        }

    except Exception as e:
        logger.error(f"Failed to reload agent: {e}")
        raise HTTPException(500, f"Failed to reload agent: {str(e)}")


@router.post("/agent/multistep/create")
async def create_multistep_task(
    task_id: str, description: str, context: Optional[Dict[str, Any]] = None
):
    """Create a new multi-step task"""
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

    except Exception as e:
        logger.error(f"Multi-step task creation failed: {e}")
        raise HTTPException(500, f"Task creation failed: {str(e)}")


@router.post("/agent/multistep/plan")
async def auto_plan_task(task_id: str, planning_agent: str = "reasoning"):
    """Automatically plan steps for a multi-step task"""
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

    except Exception as e:
        logger.error(f"Task planning failed: {e}")
        raise HTTPException(500, f"Task planning failed: {str(e)}")


@router.post("/agent/multistep/execute")
async def execute_multistep_task(
    task_id: str, background_tasks: BackgroundTasks, max_parallel_steps: int = 2
):
    """Execute a multi-step task"""
    try:
        processor = get_multi_step_processor()

        if task_id not in processor.active_tasks:
            raise HTTPException(404, f"Task '{task_id}' not found")

        # Execute task
        result = await processor.execute_task(
            task_id=task_id, max_parallel_steps=max_parallel_steps
        )

        return {"success": True, "execution_result": result}

    except Exception as e:
        logger.error(f"Multi-step task execution failed: {e}")
        raise HTTPException(500, f"Task execution failed: {str(e)}")


@router.get("/agent/multistep/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a multi-step task"""
    try:
        processor = get_multi_step_processor()
        status = processor.get_task_status(task_id)

        if status is None:
            raise HTTPException(404, f"Task '{task_id}' not found")

        return {"success": True, "task_status": status}

    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(500, f"Failed to get task status: {str(e)}")


@router.get("/agent/multistep/list")
async def list_active_tasks():
    """List all active multi-step tasks"""
    try:
        processor = get_multi_step_processor()
        active_tasks = processor.list_active_tasks()

        return {
            "success": True,
            "active_tasks": active_tasks,
            "count": len(active_tasks),
        }

    except Exception as e:
        logger.error(f"Failed to list active tasks: {e}")
        raise HTTPException(500, f"Failed to list active tasks: {str(e)}")


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
    """Process a player action within story context"""
    try:
        story_manager = get_story_manager()

        # Create story context
        story_context = StoryContext(
            story_id=story_id,
            character_name=character_name,
            current_scene=current_scene,
            character_state=character_state or {},
            story_history=story_history or [],
            available_actions=available_actions or [],
            narrative_style=narrative_style,
        )

        # Get story agent and process action
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

    except Exception as e:
        logger.error(f"Story action processing failed: {e}")
        raise HTTPException(500, f"Story action processing failed: {str(e)}")


@router.post("/agent/story/scene")
async def generate_scene_description(
    story_id: str,
    character_name: str,
    current_scene: str,
    character_state: Optional[Dict[str, Any]] = None,
    narrative_style: str = "adventure",
    scene_type: str = "descriptive",
):
    """Generate detailed scene description"""
    try:
        story_manager = get_story_manager()

        # Create story context
        story_context = StoryContext(
            story_id=story_id,
            character_name=character_name,
            current_scene=current_scene,
            character_state=character_state or {},
            story_history=[],
            available_actions=[],
            narrative_style=narrative_style,
        )

        # Get story agent and generate scene
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

    except Exception as e:
        logger.error(f"Scene generation failed: {e}")
        raise HTTPException(500, f"Scene generation failed: {str(e)}")


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
    """Process complex story scenarios using multiple agents"""
    try:
        story_manager = get_story_manager()

        # Create story context
        story_context = StoryContext(
            story_id=story_id,
            character_name=character_name,
            current_scene=current_scene,
            character_state=character_state or {},
            story_history=story_history or [],
            available_actions=[],
            narrative_style=narrative_style,
        )

        # Process complex scenario
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

    except Exception as e:
        logger.error(f"Complex story scenario failed: {e}")
        raise HTTPException(500, f"Complex story scenario failed: {str(e)}")


@router.get("/agent/story/active")
async def list_active_stories():
    """List all active story contexts"""
    try:
        story_manager = get_story_manager()
        active_stories = story_manager.get_active_stories()

        return {
            "success": True,
            "active_stories": active_stories,
            "count": len(active_stories),
        }

    except Exception as e:
        logger.error(f"Failed to list active stories: {e}")
        raise HTTPException(500, f"Failed to list active stories: {str(e)}")


@router.delete("/agent/multistep/cancel/{task_id}")
async def cancel_multistep_task(task_id: str):
    """Cancel an active multi-step task"""
    try:
        processor = get_multi_step_processor()
        success = processor.cancel_task(task_id)

        if success:
            return {
                "success": True,
                "message": f"Task '{task_id}' cancelled successfully",
            }
        else:
            raise HTTPException(404, f"Task '{task_id}' not found")

    except Exception as e:
        logger.error(f"Task cancellation failed: {e}")
        raise HTTPException(500, f"Task cancellation failed: {str(e)}")


# Additional utility endpoints


@router.get("/agent/info")
async def get_agent_system_info():
    """Get agent system information and statistics"""
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
                "tool_categories": {},  # Could add category breakdown
                "agent_types": [
                    "reasoning",
                    "narrative",
                    "dialogue",
                    "world",
                    "action",
                ],
            },
            "tool_list": registry.list_tools(),
            "active_tasks": processor.list_active_tasks(),
        }

    except Exception as e:
        logger.error(f"Failed to get agent info: {e}")
        raise HTTPException(500, f"Failed to get agent info: {str(e)}")


@router.post("/agent/tools/configure")
async def configure_tool_settings(tool_name: str, settings: Dict[str, Any]):
    """Configure tool-specific settings"""
    try:
        # This is a placeholder for tool configuration
        # Each tool could have its own configuration method

        if tool_name == "web_search":
            from core.agent.tools.web_search import configure_search_engine

            result = configure_search_engine(**settings)
            return result
        else:
            return {
                "success": False,
                "error": f"Configuration not supported for tool: {tool_name}",
            }

    except Exception as e:
        logger.error(f"Tool configuration failed: {e}")
        raise HTTPException(500, f"Tool configuration failed: {str(e)}")


# Health check for agent system
@router.get("/agent/health")
async def agent_health_check():
    """Health check for agent system components"""
    try:
        registry = get_tool_registry()
        executor = get_agent_executor()

        # Basic health checks
        health_status = {
            "tool_registry": len(registry.list_tools()) > 0,
            "agent_executor": executor is not None,
            "available_tools": registry.list_tools(),
            "system_ready": True,
        }

        # Test a simple tool execution
        try:
            test_result = await executor.execute_tool(
                "calculator", {"expression": "2+2"}
            )
            health_status["tool_execution_test"] = test_result.success
        except Exception as e:
            health_status["tool_execution_test"] = False
            health_status["tool_test_error"] = str(e)

        return {"success": True, "health": health_status, "timestamp": time.time()}

    except Exception as e:
        logger.error(f"Agent health check failed: {e}")
        return {"success": False, "error": str(e), "timestamp": time.time()}
