# core/rag/vector_store.py

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import faiss
import pickle
import json
from datetime import datetime
from dataclasses import dataclass, asdict

from ..config import get_config
from ..shared_cache import get_shared_cache
from ..exceptions import RAGError

logger = logging.getLogger(__name__)


@dataclass
class VectorMetadata:
    """Metadata for vector entries"""

    doc_id: str
    chunk_id: str
    content_hash: str
    timestamp: datetime
    metadata: Dict[str, Any]


class VectorStore:
    """Vector storage and retrieval using FAISS"""

    def __init__(
        self, dimension: int = 768, index_type: str = "flat", metric: str = "cosine"
    ):
        """
        Initialize vector store

        Args:
            dimension: Vector dimension
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
            metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.config = get_config()
        self.cache = get_shared_cache()

        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric

        # Storage
        self.index: Optional[faiss.Index] = None
        self.metadata: Dict[int, VectorMetadata] = {}  # index_id -> metadata
        self.id_to_index: Dict[str, int] = {}  # doc_id -> index_id
        self.next_index_id = 0

        # Initialize index
        self._create_index()

    def _create_index(self) -> None:
        """Create FAISS index based on configuration"""
        try:
            if self.metric == "cosine":
                # Use Inner Product with normalized vectors for cosine similarity
                if self.index_type == "flat":
                    self.index = faiss.IndexFlatIP(self.dimension)
                elif self.index_type == "ivf":
                    # IVF index for large datasets
                    quantizer = faiss.IndexFlatIP(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                elif self.index_type == "hnsw":
                    # HNSW for fast approximate search
                    self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                else:
                    raise ValueError(f"Unsupported index type: {self.index_type}")

            elif self.metric == "l2":
                if self.index_type == "flat":
                    self.index = faiss.IndexFlatL2(self.dimension)
                elif self.index_type == "ivf":
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                elif self.index_type == "hnsw":
                    self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                else:
                    raise ValueError(f"Unsupported index type: {self.index_type}")

            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            logger.info(
                f"Created FAISS index: {self.index_type}, metric={self.metric}, dim={self.dimension}"
            )

        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise RAGError(f"Index creation failed: {e}")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return vectors / norms
        return vectors

    def add_vector(
        self, doc_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add vector to store

        Args:
            doc_id: Document identifier
            vector: Vector to add
            metadata: Additional metadata

        Returns:
            Index ID of added vector
        """
        try:
            if vector.shape[0] != self.dimension:
                raise ValueError(
                    f"Vector dimension {vector.shape[0]} != expected {self.dimension}"
                )

            # Normalize if needed
            vector = self._normalize_vectors(vector.reshape(1, -1))

            # Train index if needed (for IVF)
            if hasattr(self.index, "is_trained") and not self.index.is_trained:
                if self.index.ntotal < 256:  # Need enough vectors to train
                    logger.warning(
                        "Not enough vectors to train IVF index, adding to flat index temporarily"
                    )
                else:
                    self.index.train(vector)

            # Add to index
            self.index.add(vector.astype(np.float32))

            # Store metadata
            index_id = self.next_index_id
            content_hash = self._compute_hash(vector)

            vector_metadata = VectorMetadata(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_{index_id}",
                content_hash=content_hash,
                timestamp=datetime.now(),
                metadata=metadata or {},
            )

            self.metadata[index_id] = vector_metadata
            self.id_to_index[doc_id] = index_id
            self.next_index_id += 1

            logger.debug(f"Added vector for doc_id={doc_id}, index_id={index_id}")
            return index_id

        except Exception as e:
            logger.error(f"Failed to add vector for {doc_id}: {e}")
            raise RAGError(f"Vector addition failed: {e}")

    def add_vectors_batch(
        self,
        doc_ids: List[str],
        vectors: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """
        Add multiple vectors in batch

        Args:
            doc_ids: List of document identifiers
            vectors: Array of vectors (n_vectors, dimension)
            metadata_list: List of metadata dicts

        Returns:
            List of index IDs
        """
        try:
            if len(doc_ids) != vectors.shape[0]:
                raise ValueError("Number of doc_ids must match number of vectors")

            if vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"Vector dimension {vectors.shape[1]} != expected {self.dimension}"
                )

            # Normalize if needed
            vectors = self._normalize_vectors(vectors)

            # Train index if needed
            if hasattr(self.index, "is_trained") and not self.index.is_trained:
                if vectors.shape[0] >= 256:
                    self.index.train(vectors.astype(np.float32))
                else:
                    logger.warning("Not enough vectors to train IVF index")

            # Add to index
            start_index = self.next_index_id
            self.index.add(vectors.astype(np.float32))

            # Store metadata
            index_ids = []
            for i, doc_id in enumerate(doc_ids):
                index_id = start_index + i
                content_hash = self._compute_hash(vectors[i])

                vector_metadata = VectorMetadata(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_{index_id}",
                    content_hash=content_hash,
                    timestamp=datetime.now(),
                    metadata=metadata_list[i] if metadata_list else {},
                )

                self.metadata[index_id] = vector_metadata
                self.id_to_index[doc_id] = index_id
                index_ids.append(index_id)

            self.next_index_id += len(doc_ids)

            logger.info(f"Added {len(doc_ids)} vectors in batch")
            return index_ids

        except Exception as e:
            logger.error(f"Batch vector addition failed: {e}")
            raise RAGError(f"Batch addition failed: {e}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        min_score: Optional[float] = None,
    ) -> List[Tuple[int, float, VectorMetadata]]:
        """
        Search for similar vectors

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of (index_id, score, metadata) tuples
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("No vectors in index")
                return []

            if query_vector.shape[0] != self.dimension:
                raise ValueError(
                    f"Query vector dimension {query_vector.shape[0]} != expected {self.dimension}"
                )

            # Normalize query vector
            query_vector = self._normalize_vectors(query_vector.reshape(1, -1))

            # Search
            scores, indices = self.index.search(query_vector.astype(np.float32), top_k)

            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx in self.metadata:  # Valid index
                    if min_score is None or score >= min_score:
                        results.append((int(idx), float(score), self.metadata[idx]))

            logger.debug(f"Search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RAGError(f"Search failed: {e}")

    def remove_vector(self, doc_id: str) -> bool:
        """
        Remove vector by doc_id
        Note: FAISS doesn't support efficient removal, so we mark as deleted
        """
        try:
            if doc_id not in self.id_to_index:
                logger.warning(f"Doc ID {doc_id} not found in index")
                return False

            index_id = self.id_to_index[doc_id]

            # Mark as deleted in metadata
            if index_id in self.metadata:
                self.metadata[index_id].metadata["deleted"] = True
                self.metadata[index_id].metadata[
                    "deleted_at"
                ] = datetime.now().isoformat()

            # Remove from mapping
            del self.id_to_index[doc_id]

            logger.info(f"Marked vector {doc_id} as deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to remove vector {doc_id}: {e}")
            return False

    def rebuild_index(self) -> bool:
        """Rebuild index excluding deleted vectors"""
        try:
            if not self.metadata:
                logger.warning("No vectors to rebuild")
                return False

            # Collect non-deleted vectors
            active_vectors = []
            active_metadata = {}
            active_mappings = {}
            new_index_id = 0

            for old_index_id, metadata in self.metadata.items():
                if not metadata.metadata.get("deleted", False):
                    # We need to extract the vector from current index
                    # This is a limitation - FAISS doesn't allow direct extraction
                    # In practice, you'd need to store vectors separately for rebuild
                    logger.warning("Index rebuild requires storing original vectors")
                    return False

            logger.info("Index rebuild completed")
            return True

        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        active_count = sum(
            1
            for meta in self.metadata.values()
            if not meta.metadata.get("deleted", False)
        )

        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "active_vectors": active_count,
            "deleted_vectors": len(self.metadata) - active_count,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "is_trained": (
                getattr(self.index, "is_trained", True) if self.index else False
            ),
        }

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save vector store to disk"""
        if filepath is None:
            filepath = self.cache.get_output_path("rag") / "vector_store"

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        try:
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, str(filepath / "faiss.index"))

            # Save metadata and mappings
            store_data = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
                "next_index_id": self.next_index_id,
                "id_to_index": self.id_to_index,
                "metadata": {
                    str(k): {
                        "doc_id": v.doc_id,
                        "chunk_id": v.chunk_id,
                        "content_hash": v.content_hash,
                        "timestamp": v.timestamp.isoformat(),
                        "metadata": v.metadata,
                    }
                    for k, v in self.metadata.items()
                },
                "created_at": datetime.now().isoformat(),
            }

            with open(filepath / "store_data.json", "w", encoding="utf-8") as f:
                json.dump(store_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Vector store saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise RAGError(f"Save failed: {e}")

    def load(self, filepath: Path) -> bool:
        """Load vector store from disk"""
        try:
            filepath = Path(filepath)

            if not filepath.exists():
                logger.warning(f"Vector store path does not exist: {filepath}")
                return False

            # Load FAISS index
            index_file = filepath / "faiss.index"
            if index_file.exists():
                self.index = faiss.read_index(str(index_file))

            # Load metadata and mappings
            store_file = filepath / "store_data.json"
            if store_file.exists():
                with open(store_file, "r", encoding="utf-8") as f:
                    store_data = json.load(f)

                self.dimension = store_data["dimension"]
                self.index_type = store_data["index_type"]
                self.metric = store_data["metric"]
                self.next_index_id = store_data["next_index_id"]
                self.id_to_index = store_data["id_to_index"]

                # Reconstruct metadata
                self.metadata = {}
                for k, v in store_data["metadata"].items():
                    self.metadata[int(k)] = VectorMetadata(
                        doc_id=v["doc_id"],
                        chunk_id=v["chunk_id"],
                        content_hash=v["content_hash"],
                        timestamp=datetime.fromisoformat(v["timestamp"]),
                        metadata=v["metadata"],
                    )

            logger.info(f"Vector store loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False

    def _compute_hash(self, vector: np.ndarray) -> str:
        """Compute hash of vector for deduplication"""
        import hashlib

        vector_bytes = vector.astype(np.float32).tobytes()
        return hashlib.md5(vector_bytes).hexdigest()[:16]


class HybridVectorStore:
    """Hybrid vector store with multiple indices for different purposes"""

    def __init__(self):
        self.stores: Dict[str, VectorStore] = {}
        self.default_store = "main"

    def create_store(
        self,
        name: str,
        dimension: int = 768,
        index_type: str = "flat",
        metric: str = "cosine",
    ) -> VectorStore:
        """Create a new vector store"""
        store = VectorStore(dimension, index_type, metric)
        self.stores[name] = store
        return store

    def get_store(self, name: str = None) -> VectorStore:
        """Get vector store by name"""
        store_name = name or self.default_store

        if store_name not in self.stores:
            # Create default store
            self.stores[store_name] = VectorStore()

        return self.stores[store_name]

    def list_stores(self) -> List[str]:
        """List all store names"""
        return list(self.stores.keys())

    def save_all(self, base_path: Optional[Path] = None) -> Dict[str, Path]:
        """Save all stores"""
        if base_path is None:
            cache = get_shared_cache()
            base_path = cache.get_output_path("rag") / "vector_stores"

        base_path = Path(base_path)
        saved_paths = {}

        for name, store in self.stores.items():
            store_path = base_path / name
            try:
                saved_paths[name] = store.save(store_path)
            except Exception as e:
                logger.error(f"Failed to save store {name}: {e}")

        return saved_paths

    def load_all(self, base_path: Path) -> Dict[str, bool]:
        """Load all stores from base path"""
        base_path = Path(base_path)
        loaded_stores = {}

        if not base_path.exists():
            logger.warning(f"Base path does not exist: {base_path}")
            return {}

        for store_dir in base_path.iterdir():
            if store_dir.is_dir():
                store_name = store_dir.name
                try:
                    store = VectorStore()
                    success = store.load(store_dir)
                    if success:
                        self.stores[store_name] = store
                        loaded_stores[store_name] = True
                    else:
                        loaded_stores[store_name] = False
                except Exception as e:
                    logger.error(f"Failed to load store {store_name}: {e}")
                    loaded_stores[store_name] = False

        return loaded_stores
