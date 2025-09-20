# tests/mocks/__init__.py
"""
Mock工具包
"""

from .model_mocks import (
    MockLLMAdapter,
    MockVLMEngine,
    MockRAGEngine,
    MockAgentExecutor,
    MockGameEngine,
)

from .api_mocks import (
    mock_all_models,
    mock_file_operations,
    create_mock_image,
    create_mock_upload_file,
)

from .sample_data import (
    get_sample_data,
    SAMPLE_DOCUMENTS,
    SAMPLE_CONVERSATIONS,
    SAMPLE_GAME_DATA,
    SAMPLE_AGENT_TOOLS,
)

__all__ = [
    # Model mocks
    "MockLLMAdapter",
    "MockVLMEngine",
    "MockRAGEngine",
    "MockAgentExecutor",
    "MockGameEngine",
    # API mocks
    "mock_all_models",
    "mock_file_operations",
    "create_mock_image",
    "create_mock_upload_file",
    # Sample data
    "get_sample_data",
    "SAMPLE_DOCUMENTS",
    "SAMPLE_CONVERSATIONS",
    "SAMPLE_GAME_DATA",
    "SAMPLE_AGENT_TOOLS",
]
