# core/performance.py
import asyncio
import hashlib
import json
import time
from typing import Any, Optional, Dict, List
from functools import wraps, lru_cache
from dataclasses import dataclass
import torch
import redis
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from core.config import get_config
from core.shared_cache import get_shared_cache


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PerformanceOptimizer:
    """Centralized performance optimization and caching"""

    def __init__(self):
        self.settings = get_config()
        self.redis_client = redis.from_url(self.settings.cache.redis_url)
        self.embedding_cache = {}
        self.model_cache = {}
        self.stats = {
            "embedding": CacheStats(),
            "model": CacheStats(),
            "generation": CacheStats(),
        }

    def cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache keys"""
        content = json.dumps(
            {"args": args, "kwargs": sorted(kwargs.items())},
            sort_keys=True,
            default=str,
        )
        hash_obj = hashlib.md5(content.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"

    async def get_cached_embedding(
        self, text: str, model_name: str
    ) -> Optional[torch.Tensor]:
        """Get cached embedding or return None"""
        cache_key = self.cache_key("embedding", text, model_name)

        try:
            # Try memory cache first
            if cache_key in self.embedding_cache:
                self.stats["embedding"].hits += 1
                return self.embedding_cache[cache_key]

            # Try Redis cache
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                embedding = torch.tensor(json.loads(cached_data))  # type: ignore
                # Store in memory cache (LRU will handle size)
                if len(self.embedding_cache) < 1000:
                    self.embedding_cache[cache_key] = embedding
                self.stats["embedding"].hits += 1
                return embedding

            self.stats["embedding"].misses += 1
            return None

        except Exception as e:
            print(f"Cache retrieval error: {e}")
            self.stats["embedding"].misses += 1
            return None

    async def cache_embedding(
        self, text: str, model_name: str, embedding: torch.Tensor, ttl: int = 3600
    ):
        """Cache embedding with TTL"""
        cache_key = self.cache_key("embedding", text, model_name)

        try:
            # Cache in memory (limited size)
            if len(self.embedding_cache) < 1000:
                self.embedding_cache[cache_key] = embedding

            # Cache in Redis with TTL
            embedding_json = json.dumps(embedding.tolist())
            self.redis_client.setex(cache_key, ttl, embedding_json)

        except Exception as e:
            print(f"Cache storage error: {e}")

    @lru_cache(maxsize=5)
    def get_model(self, model_name: str, device: str = "auto"):
        """Cached model loading with LRU eviction"""
        try:
            if torch.cuda.is_available() and device == "auto":
                device = "cuda"

            print(f"Loading model {model_name} on {device}")

            # Model-specific optimizations
            if "bge" in model_name.lower():
                model = SentenceTransformer(model_name, device=device)
                if hasattr(model, "half") and device == "cuda":
                    model = model.half()  # Use fp16 for better performance
            else:
                # Generic model loading
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                )

            self.stats["model"].hits += 1
            return model

        except Exception as e:
            print(f"Model loading error: {e}")
            self.stats["model"].misses += 1
            raise

    def batch_embeddings(
        self, texts: List[str], model_name: str, batch_size: int = 32
    ) -> torch.Tensor:
        """Optimized batch embedding with caching"""
        model = self.get_model(model_name)

        # Check cache for all texts
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = asyncio.run(self.get_cached_embedding(text, model_name))
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Compute embeddings for uncached texts
        if uncached_texts:
            with torch.no_grad():
                if hasattr(model, "encode"):
                    # SentenceTransformer
                    new_embeddings = model.encode(
                        uncached_texts,
                        batch_size=batch_size,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                    )
                else:
                    # Generic transformer model
                    tokenizer = AutoTokenizer.from_pretrained(model_name)

                    all_embeddings = []
                    for i in range(0, len(uncached_texts), batch_size):
                        batch_texts = uncached_texts[i : i + batch_size]
                        inputs = tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=512,
                        )

                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}

                        outputs = model(**inputs)
                        # Use mean pooling
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                        all_embeddings.append(embeddings)

                    new_embeddings = torch.cat(all_embeddings, dim=0)

            # Cache new embeddings
            for i, (text_idx, embedding) in enumerate(
                zip(uncached_indices, new_embeddings)
            ):
                asyncio.run(
                    self.cache_embedding(texts[text_idx], model_name, embedding)
                )
                cached_embeddings[text_idx] = embedding

        # Reconstruct full embedding tensor
        result_embeddings = []
        for i in range(len(texts)):
            result_embeddings.append(cached_embeddings[i])

        return torch.stack(result_embeddings)

    def clear_cache(self, cache_type: str = "all"):
        """Clear specified cache type"""
        if cache_type in ["all", "embedding"]:
            self.embedding_cache.clear()
            # Clear Redis embedding cache
            for key in self.redis_client.scan_iter(match="embedding:*"):
                self.redis_client.delete(key)

        if cache_type in ["all", "model"]:
            self.get_model.cache_clear()

        print(f"Cleared {cache_type} cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            "embedding": {
                "hits": self.stats["embedding"].hits,
                "misses": self.stats["embedding"].misses,
                "hit_rate": self.stats["embedding"].hit_rate,
                "memory_size": len(self.embedding_cache),
            },
            "model": {
                "hits": self.stats["model"].hits,
                "misses": self.stats["model"].misses,
                "hit_rate": self.stats["model"].hit_rate,
                "cache_size": self.get_model.cache_info().currsize,
            },
            "redis_info": self.redis_client.info("memory"),
        }


# Global optimizer instance
optimizer = PerformanceOptimizer()


def cached_embedding(model_name: str, ttl: int = 3600):
    """Decorator for caching embedding functions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(text: str, *args, **kwargs):
            # Try cache first
            cached = await optimizer.get_cached_embedding(text, model_name)
            if cached is not None:
                return cached

            # Compute and cache
            result = await func(text, *args, **kwargs)
            await optimizer.cache_embedding(text, model_name, result, ttl)
            return result

        return wrapper

    return decorator


def gpu_memory_cleanup():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class MemoryMonitor:
    """Monitor and log memory usage"""

    def __init__(self, threshold_gb: float = 0.5):
        self.threshold_gb = threshold_gb
        self.last_cleanup = time.time()

    def check_and_cleanup(self, force: bool = False):
        """Check memory usage and cleanup if needed"""
        if not torch.cuda.is_available():
            return

        allocated_gb = torch.cuda.memory_allocated() / 1024**3

        if allocated_gb > self.threshold_gb or force:
            if time.time() - self.last_cleanup > 30:  # Don't cleanup too frequently
                print(f"GPU memory: {allocated_gb:.2f}GB, performing cleanup")
                gpu_memory_cleanup()
                self.last_cleanup = time.time()

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory

        return {
            "gpu_available": True,
            "allocated_gb": allocated / 1024**3,
            "cached_gb": cached / 1024**3,
            "total_gb": total / 1024**3,
            "utilization_percent": (allocated / total) * 100,
        }


# Global memory monitor
memory_monitor = MemoryMonitor()
