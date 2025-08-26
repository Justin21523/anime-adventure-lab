# core/rag/retrievers.py
"""Document retrieval utilities"""
import json
import numpy as np
from pathlib import Path
from typing import List, Optional
from .schemas import RetrievalItem
from .embedders import SimpleEmbedder
from ..shared_cache import get_shared_cache


class SimpleRetriever:
    """Simple cosine similarity retriever"""

    def __init__(self):
        self.cache = get_shared_cache()
        self.embedder = SimpleEmbedder()
        self._index = {}
        self._documents = {}

    def add_documents(self, texts: List[str], doc_ids: List[str], chunk_ids: List[str]):
        """Add documents to retrieval index"""
        try:
            embeddings = self.embedder.embed_texts(texts)

            for i, (text, doc_id, chunk_id) in enumerate(
                zip(texts, doc_ids, chunk_ids)
            ):
                self._documents[chunk_id] = {
                    "text": text,
                    "doc_id": doc_id,
                    "embedding": embeddings[i],
                }

                if doc_id not in self._index:
                    self._index[doc_id] = []
                self._index[doc_id].append(chunk_id)

        except Exception as e:
            raise RuntimeError(f"Failed to add documents: {str(e)}")

    def retrieve(
        self, query: str, world_id: Optional[str] = None, top_k: int = 8
    ) -> List[RetrievalItem]:
        """Retrieve relevant documents"""
        try:
            if not self._documents:
                return []

            query_embedding = self.embedder.embed_query(query)

            # Calculate similarities
            similarities = []
            for chunk_id, doc_data in self._documents.items():
                similarity = np.dot(query_embedding, doc_data["embedding"])
                similarities.append((chunk_id, similarity))

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)

            results = []
            for chunk_id, score in similarities[:top_k]:
                doc_data = self._documents[chunk_id]
                item = RetrievalItem(
                    chunk_id=chunk_id,
                    doc_id=doc_data["doc_id"],
                    text=doc_data["text"],
                    score=float(score),
                    metadata={"world_id": world_id} if world_id else {},
                )
                results.append(item)

            return results

        except Exception as e:
            raise RuntimeError(f"Retrieval failed: {str(e)}")


class VectorRetriever:
    """Vector-based document retrieval"""

    def __init__(self, embedding_model: EmbeddingModel, index_name: str = "default"):
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.index_path = Path(INDEX_DIR) / index_name
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None

        self._load_index()

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the retriever"""
        if not documents:
            return

        # Extract text content
        texts = [doc["content"] for doc in documents]

        # Generate embeddings
        new_embeddings = self.embedding_model.encode(texts)

        # Add to storage
        self.documents.extend(documents)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Update FAISS index
        self._update_faiss_index()

        # Save index
        self._save_index()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        if not self.documents:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode_single(query)

        if FAISS_AVAILABLE and self.faiss_index:
            # Use FAISS for fast search
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                min(top_k, len(self.documents)),
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))

            return results
        else:
            # Fallback to cosine similarity
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1)
                * np.linalg.norm(query_embedding)
            )

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            return [
                (self.documents[idx], float(similarities[idx])) for idx in top_indices
            ]

    def _update_faiss_index(self):
        """Update FAISS index with current embeddings"""
        if FAISS_AVAILABLE and self.embeddings is not None:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(
                dim
            )  # Inner product for cosine similarity

            # Normalize embeddings for cosine similarity
            normalized_embeddings = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )

            self.faiss_index.add(normalized_embeddings.astype(np.float32))

    def _save_index(self):
        """Save index to disk"""
        try:
            # Save documents and embeddings
            with open(self.index_path / "documents.json", "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)

            if self.embeddings is not None:
                np.save(self.index_path / "embeddings.npy", self.embeddings)

            # Save FAISS index
            if FAISS_AVAILABLE and self.faiss_index:
                faiss.write_index(
                    self.faiss_index, str(self.index_path / "faiss.index")
                )

        except Exception as e:
            print(f"Failed to save index: {e}")

    def _load_index(self):
        """Load index from disk"""
        try:
            docs_file = self.index_path / "documents.json"
            embeddings_file = self.index_path / "embeddings.npy"
            faiss_file = self.index_path / "faiss.index"

            if docs_file.exists():
                with open(docs_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)

            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)

            if FAISS_AVAILABLE and faiss_file.exists() and self.embeddings is not None:
                self.faiss_index = faiss.read_index(str(faiss_file))

            print(
                f"Loaded {len(self.documents)} documents from index {self.index_name}"
            )

        except Exception as e:
            print(f"Failed to load index: {e}")
