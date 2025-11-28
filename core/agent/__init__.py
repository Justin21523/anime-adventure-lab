"""兼容舊版匯入的 Agent 套件。

將 core.agent.* 轉向 core.agents.* 實作。
"""

from core.agents.base_agent import BaseAgent  # noqa: F401
from core.agents.tool_registry import ToolRegistry  # noqa: F401
from core.agents.executor import SafeExecutor, AgentExecutor, ExecutionResult  # noqa: F401
from core.agents.multi_step_processor import (  # noqa: F401
    MultiStepProcessor,
    MultiStepTask,
    TaskStatus,
)
from core.agents.base_agent import SimpleReasoningAgent  # noqa: F401
from core.agents.story_integration import (  # noqa: F401
    StoryAgent,
    StoryContext,
    StoryAgentManager,
)

# 轉出工具子模組
from core.agents.tools import *  # noqa: F401,F403

__all__ = [
    "BaseAgent",
    "ToolRegistry",
    "SafeExecutor",
    "AgentExecutor",
    "ExecutionResult",
    "MultiStepProcessor",
    "MultiStepTask",
    "TaskStatus",
    "SimpleReasoningAgent",
    "StoryAgent",
    "StoryContext",
    "StoryAgentManager",
]
__all__ += [n for n in globals() if not n.startswith("_") and n not in __all__]
