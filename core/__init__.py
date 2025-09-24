# core/__init__.py - 修正缺失的導入

"""
Core Multi-Modal Lab Components
"""

from .t2i import get_t2i_pipeline, save_image_to_cache, LoRAManager
from .rag import DocumentMemory


# 基礎配置
try:
    from .config import get_config, ModelConfig, APIConfig, SafetyConfig
except ImportError as e:
    print(f"Warning: Config import failed: {e}")

# 例外處理
try:
    from .exceptions import (
        ValidationError,
        SafetyError,
        RateLimitError,
        ImageProcessingError,
        ModelLoadError,
        ConfigurationError,
    )
except ImportError as e:
    print(f"Warning: Exceptions import failed: {e}")

# 效能與快取
try:
    # from .performance import PerformanceMonitor
    from .shared_cache import get_shared_cache
except ImportError as e:
    print(f"Warning: Performance/Cache import failed: {e}")

# 安全模組
try:
    from .safety import get_content_filter, get_input_validator, get_rate_limiter
except ImportError as e:
    print(f"Warning: Safety import failed: {e}")

# 工具模組
try:
    from .utils.image import ImageProcessor
    from .utils.cache import CacheManager
except ImportError as e:
    print(f"Warning: Utils import failed: {e}")


# Registry - 創建一個簡單的 registry 物件
class SimpleRegistry:
    """Simple registry for tools and models"""

    def __init__(self):
        self.items = {}

    def register(self, name: str, item: any):  # type: ignore
        """Register an item"""
        self.items[name] = item

    def get(self, name: str):
        """Get registered item"""
        return self.items.get(name)

    def list_items(self):
        """List all registered items"""
        return list(self.items.keys())


# 創建全域 registry 實例
registry = SimpleRegistry()

# 版本資訊
__version__ = "0.1.0"
__author__ = "Multi-Modal Lab Team"

# 所有公開的 API
__all__ = [
    # 配置
    "get_config",
    "ModelConfig",
    "APIConfig",
    "SafetyConfig",
    # 例外
    "ValidationError",
    "SafetyError",
    "RateLimitError",
    "ImageProcessingError",
    "ModelLoadError",
    "ConfigurationError",
    # 效能
    "PerformanceMonitor",
    "get_shared_cache",
    # 安全
    "get_content_filter",
    "get_input_validator",
    "get_rate_limiter",
    # 工具
    "ImageProcessor",
    "CacheManager",
    # Registry
    "registry",
    "get_t2i_pipeline",
    "save_image_to_cache",
    "LoRAManager",
    "DocumentMemory",
    # 中繼資料
    "__version__",
    "__author__",
]
