"""代理到 core.agents.story_integration 的兼容模組。"""

from core.agents.story_integration import StoryAgent, StoryContext, StoryAgentManager  # noqa: F401

__all__ = ["StoryAgent", "StoryContext", "StoryAgentManager"]
