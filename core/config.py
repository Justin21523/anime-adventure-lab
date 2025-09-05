# core/config.py
"""
Configuration Management
Loads YAML configs with environment variable overrides
"""

import os
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, NamedTuple
from dataclasses import dataclass
from pydantic import Field
from pydantic_settings import SettingsConfigDict, BaseSettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIConfig(BaseSettings):
    """API Server Configuration"""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    prefix: str = Field(default="/api/v1", description="API PREFIX")
    cors_origins: str = Field(
        default="http://localhost:7860", description="CORS origins"
    )
    debug: bool = Field(default=False, description="Debug mode")
    max_workers: int = Field(default=2, description="Max concurrent workers")
    request_timeout: int = Field(default=300, description="Request timeout seconds")

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]


class ModelConfig(BaseSettings):
    """Model Configuration"""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    # GPU/Device settings
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")  # type: ignore
    device: str = Field(default="auto", description="Device mapping strategy")
    device_map: str = Field(default="auto", description="Model device mapping")
    torch_dtype: str = Field(default="float16", description="Default torch dtype")
    max_batch_size: int = Field(default=4, description="Max inference batch size")

    # Extended multi-modal settings (from core_config_module.py)
    log_level: str = Field(default="INFO", description="LOG_LEVEL")

    # Memory optimization
    use_4bit_loading: bool = Field(default=True, description="Use 4-bit quantization")
    use_fp16: bool = True
    use_gradient_checkpointing: bool = Field(
        default=True, description="Use gradient checkpointing"
    )
    use_xformers: bool = Field(default=False, description="Use xformers attention")
    max_memory_gb: float = Field(default=8.0, description="Max VRAM usage in GB")

    # Performance settings
    enable_attention_slicing: bool = Field(
        default=True, description="Enable attention slicing"
    )
    enable_vae_slicing: bool = Field(default=True, description="Enable VAE slicing")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")

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

    # Model specific settings
    caption_model: str = Field(
        default="Salesforce/blip2-opt-2.7b", description="Caption model"
    )
    vqa_model: str = Field(default="llava-hf/llava-1.5-7b-hf", description="VQA model")
    chat_model: str = Field(default="Qwen/Qwen-7B-Chat", description="Chat model")
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5", description="Embedding model"
    )

    # Game settings
    game_save_path: str = Field(default="outputs/games", description="GAME_SAVE_PATH")
    max_game_sessions: int = Field(default=10, description="MAX_GAME_SESSIONS")

    # Training settings
    training_output_path: str = Field(
        default="outputs/training", description="TRAINING_OUTPUT_PATH"
    )
    max_training_jobs: int = Field(default=2, description="MAX_TRAINING_JOBS")
    max_image_size: int = Field(default=2048, description="MAX_IMAGE_SIZE")


class SafetyConfig(BaseSettings):
    """Safety and Content Filtering Configuration"""

    model_config = SettingsConfigDict(env_prefix="SAFETY_")

    enable_nsfw_filter: bool = Field(default=True, description="ENABLE_NSFW_FILTER")
    enable_nsfw_filter: bool = Field(default=True, description="Enable NSFW detection")
    enable_face_blur: bool = Field(default=False, description="Enable face blurring")
    enable_watermark: bool = Field(default=False, description="Enable watermarking")
    blocked_terms: str = Field(default="", description="Comma-separated blocked terms")

    @property
    def blocked_terms_list(self) -> List[str]:
        if not self.blocked_terms:
            return []
        return [term.strip().lower() for term in self.blocked_terms.split(",")]


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
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0", description="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0", description="CELERY_RESULT_BACKEND"
    )
    memory_ttl_minutes: int = Field(default=60, description="Memory cache TTL")
    model_cache_size_gb: int = Field(default=10, description="Model cache size limit")
    auto_cleanup: bool = Field(default=True, description="Auto cleanup old cache")


class AppConfig:
    """Main Application Configuration"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/app.yaml"
        self.config_dir = "configs"
        self.yaml_config = self._load_yaml_config()

        # Initialize component configs
        self.api = APIConfig()
        self.model = ModelConfig()
        self.safety = SafetyConfig()
        self.rag = RAGConfig()
        self.database = DatabaseConfig()
        self.cache = CacheConfig()
        self._cache = {}

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            # Create default config if not exists
            self._create_default_config(config_file)

        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML config file with caching"""
        if filename in self._cache:
            return self._cache[filename]

        file_path = Path(self.config_dir) / filename
        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            return {}

        self._cache[filename] = config or {}
        return self._cache[filename]

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model configuration"""
        models_config = self.load_yaml("models.yaml")
        if model_name not in models_config:
            logger.warning(f"Model config not found: {model_name}")
            return {}
        return models_config[model_name]

    def get_train_config(self, config_name: str) -> Dict[str, Any]:
        """Get training configuration"""
        config_path = f"train/{config_name}"
        return self.load_yaml(config_path)

    def get_preset_config(self, preset_name: str) -> Dict[str, Any]:
        """Get style preset configuration"""
        config_path = f"presets/{preset_name}"
        return self.load_yaml(config_path)

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.load_yaml("agent.yaml")

    def get_game_persona_config(self) -> Dict[str, Any]:
        """Get game persona configuration"""
        return self.load_yaml("game_persona.json")

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
                "enable_caption": True,
                "enable_vqa": True,
                "enable_chat": True,
                "enable_vlm": True,
                "enable_rag": True,
                "enable_agent": True,
                "enable_game": True,
                "enable_t2i": True,
                "enable_safety": True,
                "enable_export": True,
                "enable_train": True,
                "preload_models": False,
            },
            "limits": {
                "max_upload_size_mb": 100,
                "max_batch_size": 50,
                "request_timeout_seconds": 300,
            },
            "performance": {
                "low_vram_mode": True,
                "mixed_precision": True,
                "gradient_checkpointing": True,
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

    def get_feature_flag(self, feature: str) -> bool:
        """Get feature flag value"""
        return self.get(f"features.enable_{feature}", True)

    def to_dict(self) -> Dict[str, Any]:
        """Export full configuration as dict"""
        return {
            "api": self.api.dict(),
            "model": self.model.dict(),
            "safety": self.safety.dict(),
            "cache": self.cache.dict(),
            "yaml_config": self.yaml_config,
        }

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


def setup_logging(config: Optional[AppConfig] = None) -> None:
    """Setup logging based on configuration"""
    if config is None:
        config = get_config()

    log_config = config.get("logging", {})

    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format=log_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_config.get("file", "/tmp/multi-modal-lab.log")),
        ],
    )


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    import json

    print(json.dumps(config.get_summary(), indent=2))
