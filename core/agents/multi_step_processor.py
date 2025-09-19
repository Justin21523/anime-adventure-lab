# core/agent/multi_step_processor.py
"""
Multi-Step Task Processor
Handles complex tasks requiring multiple agent interactions and tool usage
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .base_agent import BaseAgent, SimpleReasoningAgent, AgentMemory
from .executor import AgentExecutor, ExecutionResult

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task processing status"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskStep:
    """Individual step in multi-step task"""

    step_id: int
    description: str
    agent_name: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[int] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None

    def is_ready(self, completed_steps: List[int]) -> bool:
        """Check if step is ready to execute based on dependencies"""
        return all(dep_id in completed_steps for dep_id in self.dependencies)


@dataclass
class MultiStepTask:
    """Multi-step task definition"""

    task_id: str
    description: str
    steps: List[TaskStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=lambda: time.time())

    def add_step(
        self,
        description: str,
        agent_name: Optional[str] = None,
        dependencies: Optional[List[int]] = None,
    ) -> int:
        """Add a step to the task"""
        step_id = len(self.steps)
        step = TaskStep(
            step_id=step_id,
            description=description,
            agent_name=agent_name,
            dependencies=dependencies or [],
        )
        self.steps.append(step)
        return step_id

    def get_ready_steps(self) -> List[TaskStep]:
        """Get steps ready for execution"""
        completed_step_ids = [
            step.step_id for step in self.steps if step.status == TaskStatus.COMPLETED
        ]

        return [
            step
            for step in self.steps
            if step.status == TaskStatus.PENDING and step.is_ready(completed_step_ids)
        ]


class MultiStepProcessor:
    """
    Processes complex multi-step tasks using multiple agents and tools
    Handles dependencies, parallel execution, and error recovery
    """

    def __init__(self):
        self.executor = AgentExecutor()
        self.agents: Dict[str, BaseAgent] = {}
        self.active_tasks: Dict[str, MultiStepTask] = {}

        # Register default agents
        self._register_default_agents()

    def _register_default_agents(self):
        """Register default agent instances"""
        # General reasoning agent
        self.agents["reasoning"] = SimpleReasoningAgent(
            name="reasoning", description="General purpose reasoning and planning agent"
        )

        # Specialized agents can be added here
        self.agents["analysis"] = SimpleReasoningAgent(
            name="analysis", description="Data analysis and computation agent"
        )

        self.agents["research"] = SimpleReasoningAgent(
            name="research", description="Information gathering and research agent"
        )

    def register_agent(self, name: str, agent: BaseAgent):
        """Register a custom agent"""
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")

    async def create_task(
        self, task_id: str, description: str, context: Optional[Dict[str, Any]] = None
    ) -> MultiStepTask:
        """Create a new multi-step task"""
        task = MultiStepTask(
            task_id=task_id, description=description, context=context or {}
        )

        self.active_tasks[task_id] = task
        logger.info(f"Created multi-step task: {task_id}")

        return task

    async def auto_plan_task(
        self, task: MultiStepTask, planning_agent: str = "reasoning"
    ) -> MultiStepTask:
        """
        Automatically plan task steps using an agent
        """
        if planning_agent not in self.agents:
            raise ValueError(f"Planning agent '{planning_agent}' not found")

        agent = self.agents[planning_agent]

        # Use agent to break down the task
        planning_context = f"""
        Task: {task.description}
        Context: {task.context}

        Break this into 2-5 specific steps that can be executed by agents and tools.
        Consider what information is needed and what tools might be useful.
        """

        try:
            # Plan steps using the agent
            planned_steps = await agent.plan_task(planning_context)

            # Convert planned steps to TaskStep objects
            for i, step_desc in enumerate(planned_steps):
                # Assign appropriate agent based on step content
                agent_name = self._select_agent_for_step(step_desc)

                # Determine dependencies (simple: each step depends on previous)
                dependencies = [i - 1] if i > 0 else []

                task.add_step(
                    description=step_desc,
                    agent_name=agent_name,
                    dependencies=dependencies,
                )

            logger.info(f"Auto-planned {len(task.steps)} steps for task {task.task_id}")

        except Exception as e:
            logger.error(f"Auto-planning failed for task {task.task_id}: {e}")
            # Fallback to manual step creation
            task.add_step(
                description=f"Analyze and execute: {task.description}",
                agent_name="reasoning",
            )

        return task

    def _select_agent_for_step(self, step_description: str) -> str:
        """Select appropriate agent for a step based on content"""
        step_lower = step_description.lower()

        if any(
            word in step_lower for word in ["analyze", "calculate", "compute", "data"]
        ):
            return "analysis"
        elif any(
            word in step_lower for word in ["search", "research", "find", "information"]
        ):
            return "research"
        else:
            return "reasoning"

    async def execute_task(
        self, task_id: str, max_parallel_steps: int = 2
    ) -> Dict[str, Any]:
        """
        Execute a multi-step task with dependency handling

        Args:
            task_id: ID of the task to execute
            max_parallel_steps: Maximum parallel step execution

        Returns:
            Task execution summary
        """
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.active_tasks[task_id]
        task.status = TaskStatus.IN_PROGRESS

        logger.info(
            f"Starting execution of task {task_id} with {len(task.steps)} steps"
        )

        try:
            semaphore = asyncio.Semaphore(max_parallel_steps)

            while True:
                # Get steps ready for execution
                ready_steps = task.get_ready_steps()

                if not ready_steps:
                    # Check if all steps are completed
                    pending_steps = [
                        step for step in task.steps if step.status == TaskStatus.PENDING
                    ]

                    if not pending_steps:
                        # All steps completed
                        break
                    else:
                        # Deadlock or failed dependencies
                        failed_steps = [
                            step
                            for step in task.steps
                            if step.status == TaskStatus.FAILED
                        ]

                        if failed_steps:
                            task.status = TaskStatus.FAILED
                            logger.error(
                                f"Task {task_id} failed due to failed dependencies"
                            )
                            break

                        # Wait a bit and retry
                        await asyncio.sleep(0.1)
                        continue

                # Execute ready steps in parallel
                step_tasks = []
                for step in ready_steps:
                    step_task = asyncio.create_task(
                        self._execute_step_with_semaphore(semaphore, task, step)
                    )
                    step_tasks.append(step_task)

                # Wait for current batch to complete
                if step_tasks:
                    await asyncio.gather(*step_tasks, return_exceptions=True)

            # Determine final task status
            failed_steps = [
                step for step in task.steps if step.status == TaskStatus.FAILED
            ]

            if failed_steps:
                task.status = TaskStatus.FAILED
            else:
                task.status = TaskStatus.COMPLETED

            # Compile results
            results = self._compile_task_results(task)

            logger.info(f"Task {task_id} completed with status: {task.status}")

            return results

        except Exception as e:
            logger.error(f"Task {task_id} execution failed: {e}")
            task.status = TaskStatus.FAILED
            return {
                "task_id": task_id,
                "status": task.status.value,
                "error": str(e),
                "completed_steps": 0,
                "total_steps": len(task.steps),
            }

    async def _execute_step_with_semaphore(
        self, semaphore: asyncio.Semaphore, task: MultiStepTask, step: TaskStep
    ):
        """Execute a single step with concurrency control"""
        async with semaphore:
            await self._execute_step(task, step)

    async def _execute_step(self, task: MultiStepTask, step: TaskStep):
        """Execute a single task step"""
        try:
            step.status = TaskStatus.IN_PROGRESS
            logger.info(f"Executing step {step.step_id}: {step.description}")

            # Get agent for this step
            agent_name = step.agent_name or "reasoning"
            if agent_name not in self.agents:
                raise ValueError(f"Agent '{agent_name}' not found")

            agent = self.agents[agent_name]

            # Prepare context for agent execution
            step_context = {
                "task_description": task.description,
                "step_description": step.description,
                "task_context": task.context,
                "previous_results": self._get_previous_results(task, step),
            }

            # Execute step using agent
            result = await agent.execute_task(
                task_description=step.description, context=step_context
            )

            if result["success"]:
                step.status = TaskStatus.COMPLETED
                step.result = result
                logger.info(f"Step {step.step_id} completed successfully")
            else:
                step.status = TaskStatus.FAILED
                step.error = result.get("error", "Unknown error")
                logger.error(f"Step {step.step_id} failed: {step.error}")

        except Exception as e:
            step.status = TaskStatus.FAILED
            step.error = str(e)
            logger.error(f"Step {step.step_id} execution failed: {e}")

    def _get_previous_results(
        self, task: MultiStepTask, current_step: TaskStep
    ) -> Dict[str, Any]:
        """Get results from completed dependency steps"""
        results = {}

        for dep_id in current_step.dependencies:
            if dep_id < len(task.steps):
                dep_step = task.steps[dep_id]
                if dep_step.status == TaskStatus.COMPLETED and dep_step.result:
                    results[f"step_{dep_id}"] = dep_step.result

        return results

    def _compile_task_results(self, task: MultiStepTask) -> Dict[str, Any]:
        """Compile final task results"""
        completed_steps = [
            step for step in task.steps if step.status == TaskStatus.COMPLETED
        ]

        failed_steps = [step for step in task.steps if step.status == TaskStatus.FAILED]

        # Collect all results
        step_results = []
        for step in task.steps:
            step_results.append(
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "status": step.status.value,
                    "result": step.result,
                    "error": step.error,
                    "agent": step.agent_name,
                }
            )

        # Final summary
        final_result = ""
        if completed_steps:
            # Combine results from completed steps
            result_parts = []
            for step in completed_steps:
                if step.result and isinstance(step.result, dict):
                    result_text = step.result.get("result", "")
                    if result_text:
                        result_parts.append(f"Step {step.step_id}: {result_text}")

            final_result = "\n".join(result_parts)

        return {
            "task_id": task.task_id,
            "description": task.description,
            "status": task.status.value,
            "final_result": final_result,
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "total_steps": len(task.steps),
            "step_details": step_results,
            "context": task.context,
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a task"""
        if task_id not in self.active_tasks:
            return None

        task = self.active_tasks[task_id]
        return self._compile_task_results(task)

    def list_active_tasks(self) -> List[str]:
        """List all active task IDs"""
        return list(self.active_tasks.keys())

    def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED

            # Cancel any pending steps
            for step in task.steps:
                if step.status == TaskStatus.PENDING:
                    step.status = TaskStatus.CANCELLED

            return True
        return False


import time
