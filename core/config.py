# core/config.py
"""
Configuration Management
Loads YAML configs with environment variable overrides
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import SettingsConfigDict, BaseSettings


class APIConfig(BaseSettings):
    """API Server Configuration"""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    cors_origins: str = Field(
        default="http://localhost:7860", description="CORS origins"
    )
    debug: bool = Field(default=False, description="Debug mode")

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]


class ModelConfig(BaseSettings):
    """Model Configuration"""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    # GPU/Device settings
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")  # type: ignore
    device_map: str = Field(default="auto", description="Model device mapping")
    torch_dtype: str = Field(default="float16", description="Default torch dtype")

    # Memory optimization
    use_4bit_loading: bool = Field(default=True, description="Use 4-bit quantization")
    use_fp16: bool = True
    use_gradient_checkpointing: bool = Field(
        default=True, description="Use gradient checkpointing"
    )
    use_xformers: bool = Field(default=False, description="Use xformers attention")
    max_memory_gb: float = Field(default=8.0, description="Max VRAM usage in GB")

    # Default models
    default_llm: str = Field(
        default="microsoft/DialoGPT-medium", description="Default LLM model"
    )
    default_embedding: str = Field(
        default="BAAI/bge-m3", description="Default embedding model"
    )
    default_sd_model: str = Field(
        default="runwayml/stable-diffusion-v1-5", description="Default SD model"
    )
    default_vlm_model: str = Field(
        default="runwayml/stable-diffusion-v1-5", description="Default VLM model"
    )


class RAGConfig(BaseSettings):
    """RAG Configuration"""

    model_config = SettingsConfigDict(env_prefix="RAG_")

    # Chunking
    chunk_size: int = Field(
        default=700, description="Chunk size in characters (Chinese)"
    )
    chunk_overlap: int = Field(default=120, description="Chunk overlap in characters")

    # Retrieval
    top_k: int = Field(default=8, description="Top-K retrieval results")
    rerank_top_k: int = Field(
        default=50, description="Rerank top-K before final selection"
    )
    hybrid_alpha: float = Field(
        default=0.7, description="Semantic vs BM25 weight (0.0-1.0)"
    )

    # Models
    embedding_model: str = Field(default="BAAI/bge-m3", description="Embedding model")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-large", description="Reranker model"
    )


class DatabaseConfig(BaseSettings):
    """Database Configuration"""

    model_config = SettingsConfigDict(env_prefix="DB_")

    url: str = Field(default="sqlite:///./saga_forge.db", env="DATABASE_URL")  # type: ignore
    echo: bool = Field(default=False, description="Echo SQL queries")


class CacheConfig(BaseSettings):
    """Cache Configuration"""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    root: str = Field(default="../ai_warehouse/cache", env="AI_CACHE_ROOT")  # type: ignore
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")  # type: ignore


class AppConfig:
    """Main Application Configuration"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/app.yaml"
        self.yaml_config = self._load_yaml_config()

        # Initialize component configs
        self.api = APIConfig()
        self.model = ModelConfig()
        self.rag = RAGConfig()
        self.database = DatabaseConfig()
        self.cache = CacheConfig()

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            # Create default config if not exists
            self._create_default_config(config_file)

        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _create_default_config(self, config_file: Path) -> None:
        """Create default configuration file"""
        config_file.parent.mkdir(parents=True, exist_ok=True)

        default_config = {
            "app": {
                "name": "SagaForge",
                "version": "0.1.0",
                "description": "LLM + RAG + T2I + VLM Adventure Game Engine",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "features": {
                "enable_rag": True,
                "enable_t2i": True,
                "enable_vlm": True,
                "enable_training": True,
                "enable_safety_filter": True,
            },
            "limits": {
                "max_upload_size_mb": 100,
                "max_batch_size": 50,
                "request_timeout_seconds": 300,
            },
        }

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split(".")
        value = self.yaml_config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "app": self.get("app", {}),
            "features": self.get("features", {}),
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug,
            },
            "model": {
                "device_map": self.model.device_map,
                "torch_dtype": self.model.torch_dtype,
                "use_4bit": self.model.use_4bit_loading,
                "max_memory_gb": self.model.max_memory_gb,
            },
            "cache_root": self.cache.root,
        }


# Global config instance
_app_config = None


def get_config(config_path: Optional[str] = None) -> AppConfig:
    """Get or create global configuration instance"""
    global _app_config
    if _app_config is None:
        _app_config = AppConfig(config_path)
    return _app_config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    import json

    print(json.dumps(config.get_summary(), indent=2))
