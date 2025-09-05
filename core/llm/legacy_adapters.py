# core/llm/legacy_adapters.py
"""
Legacy Model Adapters Integration
Provides compatibility with existing ollama_llm.py and transformers_llm.py
"""

import logging
from typing import Optional, Dict, Any, List, Union
from .base import BaseLLM, ChatMessage, LLMResponse
from .adapter import EnhancedLLMAdapter
from ..exceptions import ModelNotFoundError, ModelLoadError

logger = logging.getLogger(__name__)


class OllamaLLMAdapter(BaseLLM):
    """
    Ollama LLM Adapter - for local model serving via Ollama
    Integrates with existing ollama_llm.py functionality
    """

    def __init__(self, model_name: str, api_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.api_url = api_url
        self._client = None

    def load_model(self) -> None:
        """Load/verify Ollama model availability"""
        try:
            import requests

            # Check if Ollama is running
            response = requests.get(f"{self.api_url}/api/version", timeout=5)
            if response.status_code != 200:
                raise ModelLoadError(self.model_name, "Ollama server not available")

            # Check if model exists
            models_response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            if models_response.status_code == 200:
                models = models_response.json().get("models", [])
                model_names = [m["name"] for m in models]

                if self.model_name not in model_names:
                    logger.warning(
                        f"Model {self.model_name} not found in Ollama. Available: {model_names}"
                    )
                    # Try to pull the model
                    self._pull_model()

            self._loaded = True
            logger.info(f"Ollama model {self.model_name} ready")

        except ImportError:
            raise ModelLoadError(self.model_name, "requests library not available")
        except Exception as e:
            raise ModelLoadError(self.model_name, f"Ollama connection failed: {str(e)}")

    def _pull_model(self) -> None:
        """Pull model from Ollama registry"""
        try:
            import requests

            logger.info(f"Pulling Ollama model: {self.model_name}")
            response = requests.post(
                f"{self.api_url}/api/pull",
                json={"name": self.model_name},
                timeout=300,  # 5 minute timeout for model download
            )

            if response.status_code == 200:
                logger.info(f"Successfully pulled model: {self.model_name}")
            else:
                raise ModelLoadError(
                    self.model_name, f"Failed to pull model: {response.text}"
                )

        except Exception as e:
            raise ModelLoadError(self.model_name, f"Model pull failed: {str(e)}")

    def unload_model(self) -> None:
        """Unload model (for Ollama, just disconnect)"""
        self._client = None
        self._loaded = False
        logger.info(f"Disconnected from Ollama model: {self.model_name}")

    def generate(
        self, prompt: str, max_length: int = 512, temperature: float = 0.7, **kwargs
    ) -> str:
        """Generate text using Ollama API"""
        if not self.is_available():
            raise ModelNotFoundError(self.model_name)

        try:
            import requests

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_length,
                    **kwargs,
                },
                "stream": False,
            }

            response = requests.post(
                f"{self.api_url}/api/generate", json=payload, timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                raise Exception(f"Ollama API error: {response.text}")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion using Ollama API"""
        if not self.is_available():
            raise ModelNotFoundError(self.model_name)

        try:
            import requests

            # Convert messages to Ollama format
            formatted_messages = []
            for msg in self.format_messages(messages):
                formatted_messages.append({"role": msg.role, "content": msg.content})

            payload = {
                "model": self.model_name,
                "messages": formatted_messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_length,
                    **kwargs,
                },
                "stream": False,
            }

            response = requests.post(
                f"{self.api_url}/api/chat", json=payload, timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                message = result.get("message", {})
                content = message.get("content", "")

                # Extract usage info if available
                usage = {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0)
                    + result.get("eval_count", 0),
                }

                return LLMResponse(
                    content=content,
                    model_name=self.model_name,
                    usage=usage,
                    metadata={
                        "eval_duration": result.get("eval_duration"),
                        "load_duration": result.get("load_duration"),
                    },
                )
            else:
                raise Exception(f"Ollama API error: {response.text}")

        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Ollama model is available"""
        return self._loaded


class LegacyTransformersLLM(BaseLLM):
    """
    Legacy Transformers LLM - wraps existing transformers_llm.py functionality
    Provides backward compatibility
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        self.kwargs = kwargs
        self._legacy_model = None

    def load_model(self) -> None:
        """Load using legacy transformers implementation"""
        try:
            # This would import from existing transformers_llm.py
            # For now, we'll use the enhanced implementation
            from .adapter import EnhancedTransformersLLM
            from .model_loader import ModelLoadConfig

            config = ModelLoadConfig(model_name=self.model_name, **self.kwargs)

            self._legacy_model = EnhancedTransformersLLM(self.model_name, config)
            self._legacy_model.load_model()
            self._loaded = True

            logger.info(f"Legacy Transformers model loaded: {self.model_name}")

        except Exception as e:
            raise ModelLoadError(self.model_name, f"Legacy loading failed: {str(e)}")

    def unload_model(self) -> None:
        """Unload legacy model"""
        if self._legacy_model:
            self._legacy_model.unload_model()
            self._legacy_model = None
        self._loaded = False

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using legacy model"""
        if not self.is_available():
            raise ModelNotFoundError(self.model_name)

        return self._legacy_model.generate(prompt, **kwargs)  # type: ignore

    def chat(
        self, messages: List[Union[ChatMessage, Dict[str, str]]], **kwargs
    ) -> LLMResponse:
        """Chat using legacy model"""
        if not self.is_available():
            raise ModelNotFoundError(self.model_name)

        return self._legacy_model.chat(messages, **kwargs)  # type: ignore

    def is_available(self) -> bool:
        """Check if legacy model is available"""
        return self._loaded and self._legacy_model is not None


class UnifiedLLMAdapter(EnhancedLLMAdapter):
    """
    Unified LLM Adapter - integrates all legacy and new implementations
    Provides a single interface for all LLM types
    """

    def __init__(self):
        super().__init__()
        self._legacy_models: Dict[str, BaseLLM] = {}

    def get_llm(
        self,
        model_name: Optional[str] = None,
        model_type: str = "auto",
        provider: str = "auto",
        **kwargs,
    ) -> BaseLLM:
        """
        Get LLM instance with provider detection

        Args:
            model_name: Model name
            model_type: Model type (auto, qwen, llama, etc.)
            provider: Provider (auto, transformers, ollama, legacy)
            **kwargs: Additional configuration

        Returns:
            LLM instance
        """
        if model_name is None:
            model_name = self.config.get("model.chat_model", "Qwen/Qwen-7B-Chat")

        # Auto-detect provider
        if provider == "auto":
            provider = self._detect_provider(model_name)  # type: ignore

        cache_key = f"{provider}_{model_name}_{hash(str(kwargs))}"

        # Return cached instance if available
        if cache_key in self._legacy_models:
            return self._legacy_models[cache_key]

        # Create appropriate LLM instance
        if provider == "ollama":
            llm = OllamaLLMAdapter(model_name, **kwargs)  # type: ignore
        elif provider == "legacy":
            llm = LegacyTransformersLLM(model_name, **kwargs)  # type: ignore
        else:
            # Use enhanced transformers implementation
            return super().get_llm(model_name, model_type, **kwargs)

        self._legacy_models[cache_key] = llm
        return llm

    def _detect_provider(self, model_name: str) -> str:
        """Auto-detect the best provider for a model"""
        # Check if it's an Ollama model (local models)
        if "/" not in model_name or model_name.startswith("llama"):
            # Try Ollama first for local models
            try:
                import requests

                response = requests.get("http://localhost:11434/api/version", timeout=2)
                if response.status_code == 200:
                    return "ollama"
            except:
                pass

        # Default to enhanced transformers
        return "transformers"

    def list_all_models(self) -> Dict[str, List[str]]:
        """List models from all providers"""
        models = {
            "transformers": self.list_loaded_models(),
            "ollama": self._list_ollama_models(),
            "legacy": list(self._legacy_models.keys()),
        }
        return models

    def _list_ollama_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            import requests

            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m["name"] for m in models]
        except:
            pass
        return []

    def unload_all_providers(self) -> Dict[str, int]:
        """Unload models from all providers"""
        counts = {
            "transformers": self.unload_all(),
            "legacy": 0,
        }

        # Unload legacy models
        for model in self._legacy_models.values():
            try:
                model.unload_model()
                counts["legacy"] += 1
            except:
                pass

        self._legacy_models.clear()
        return counts


# Global unified adapter instance
_unified_llm_adapter: Optional[UnifiedLLMAdapter] = None


def get_unified_llm_adapter() -> UnifiedLLMAdapter:
    """Get global unified LLM adapter instance"""
    global _unified_llm_adapter
    if _unified_llm_adapter is None:
        _unified_llm_adapter = UnifiedLLMAdapter()
    return _unified_llm_adapter


# Backward compatibility functions
def get_ollama_adapter(model_name: str, **kwargs) -> OllamaLLMAdapter:
    """Get Ollama adapter for backward compatibility"""
    return OllamaLLMAdapter(model_name, **kwargs)


def get_legacy_transformers_adapter(model_name: str, **kwargs) -> LegacyTransformersLLM:
    """Get legacy transformers adapter for backward compatibility"""
    return LegacyTransformersLLM(model_name, **kwargs)
