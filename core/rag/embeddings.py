# core/rag/embeddings.py
import os
import logging
import numpy as np
from typing import List, Union, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod

from ..config import get_config
from ..shared_cache import get_shared_cache
from ..exceptions import EmbeddingError, handle_model_error


logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    """Base class for embedding models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dimension = None
        self._loaded = False

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts to embeddings"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """SentenceTransformer-based embedding model"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.config = get_config()
        self.cache = get_shared_cache()
        self.model = None

    @handle_model_error
    def _load_model(self):
        """Load SentenceTransformer model"""
        if self._loaded:
            return

        try:
            cache_dir = self.cache.get_path("MODELS_EMBEDDING")

            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(cache_dir),
                device=self.config.model.device,
            )

            self.dimension = self.model.get_sentence_embedding_dimension()
            self._loaded = True

            logger.info(
                f"Loaded SentenceTransformer: {self.model_name}, dim={self.dimension}"
            )

        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer {self.model_name}: {e}")
            raise EmbeddingError(f"Model loading failed: {e}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts to embeddings"""
        if not self._loaded:
            self._load_model()

        try:
            if isinstance(texts, str):
                texts = [texts]

            embeddings = self.model.encode(  # type: ignore
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise EmbeddingError(f"Text encoding error: {e}")

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if not self._loaded:
            self._load_model()
        return self.dimension  # type: ignore


class TransformerEmbedding(BaseEmbeddingModel):
    """Transformers-based embedding model"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.config = get_config()
        self.cache = get_shared_cache()
        self.model = None
        self.tokenizer = None

    @handle_model_error
    def _load_model(self):
        """Load Transformer model"""
        if self._loaded:
            return

        try:
            cache_dir = self.cache.get_path("MODELS_EMBEDDING")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=str(cache_dir)
            )

            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=str(cache_dir),
                torch_dtype=(
                    torch.float16 if self.config.model.use_fp16 else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else None,
            )

            # Test to get dimension
            test_embedding = self.encode("test")
            self.dimension = test_embedding.shape[-1]
            self._loaded = True

            logger.info(f"Loaded Transformer: {self.model_name}, dim={self.dimension}")

        except Exception as e:
            logger.error(f"Failed to load Transformer {self.model_name}: {e}")
            raise EmbeddingError(f"Model loading failed: {e}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts to embeddings"""
        if not self._loaded:
            self._load_model()

        try:
            if isinstance(texts, str):
                texts = [texts]

            embeddings = []

            for text in texts:
                inputs = self.tokenizer(  # type: ignore
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)  # type: ignore

                    # Use mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

                    # Normalize
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

                    embeddings.append(embedding.cpu().numpy())

            return np.array(embeddings).astype(np.float32)

        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise EmbeddingError(f"Text encoding error: {e}")

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if not self._loaded:
            self._load_model()
        return self.dimension


class EmbeddingModelFactory:
    """Factory for creating embedding models"""

    SUPPORTED_MODELS = {
        # SentenceTransformer models
        "bge-base-zh-v1.5": SentenceTransformerEmbedding,
        "bge-large-zh-v1.5": SentenceTransformerEmbedding,
        "bge-m3": SentenceTransformerEmbedding,
        "all-MiniLM-L6-v2": SentenceTransformerEmbedding,
        "paraphrase-multilingual-MiniLM-L12-v2": SentenceTransformerEmbedding,
        # Transformer models
        "BAAI/bge-base-zh-v1.5": TransformerEmbedding,
        "BAAI/bge-large-zh-v1.5": TransformerEmbedding,
        "BAAI/bge-m3": TransformerEmbedding,
    }

    @classmethod
    def create_model(cls, model_name: str) -> BaseEmbeddingModel:
        """Create embedding model by name"""

        # Try exact match first
        if model_name in cls.SUPPORTED_MODELS:
            model_class = cls.SUPPORTED_MODELS[model_name]
            return model_class(model_name)

        # Try SentenceTransformer by default
        try:
            return SentenceTransformerEmbedding(model_name)
        except:
            # Fallback to Transformer
            try:
                return TransformerEmbedding(model_name)
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to create embedding model {model_name}: {e}"
                )

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported model names"""
        return list(cls.SUPPORTED_MODELS.keys())


class EmbeddingManager:
    """Manages embedding models with caching"""

    def __init__(self):
        self._models = {}
        self._default_model = None

    def get_model(self, model_name: Optional[str] = None) -> BaseEmbeddingModel:
        """Get embedding model, with caching"""
        if model_name is None:
            if self._default_model is None:
                config = get_config()
                model_name = config.model.embedding_model
            else:
                return self._default_model

        if model_name not in self._models:
            self._models[model_name] = EmbeddingModelFactory.create_model(model_name)

            # Set as default if first model
            if self._default_model is None:
                self._default_model = self._models[model_name]

        return self._models[model_name]

    def encode(
        self, texts: Union[str, List[str]], model_name: Optional[str] = None
    ) -> np.ndarray:
        """Encode texts using specified or default model"""
        model = self.get_model(model_name)
        return model.encode(texts)

    def get_dimension(self, model_name: Optional[str] = None) -> int:
        """Get embedding dimension for model"""
        model = self.get_model(model_name)
        return model.get_dimension()

    def clear_cache(self):
        """Clear model cache"""
        self._models.clear()
        self._default_model = None


# Global embedding manager
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager() -> EmbeddingManager:
    """Get global embedding manager"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager


def encode_text(
    texts: Union[str, List[str]], model_name: Optional[str] = None
) -> np.ndarray:
    """Convenience function to encode text"""
    manager = get_embedding_manager()
    return manager.encode(texts, model_name)


def get_embedding_dimension(model_name: Optional[str] = None) -> int:
    """Convenience function to get embedding dimension"""
    manager = get_embedding_manager()
    return manager.get_dimension(model_name)
