# core/rag/engine.py

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

try:
    import torch
except ImportError:  # API profile can use lexical retrieval without torch
    torch = None  # type: ignore[assignment]
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # noqa: BLE001
    BM25Okapi = None  # type: ignore
from .reranker import CrossEncoderReranker
from ..exceptions import (
    DocumentIndexError,
    EmbeddingError,
    RAGError,
    handle_model_error,
)
from ..config import get_config
from ..shared_cache import get_shared_cache
from ..model_registry import resolve_model_path

SentenceTransformer = None
AutoModel = None
AutoTokenizer = None

logger = logging.getLogger(__name__)

try:
    import opencc  # type: ignore
except Exception:  # noqa: BLE001
    opencc = None  # type: ignore


def _resolve_device(device: str) -> str:
    raw = str(device or "").strip().lower()
    if raw == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if raw.startswith("cuda"):
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DocumentMetadata:
    doc_id: str
    title: str
    source: str
    world_id: str
    upload_time: str
    language: str = "zh-TW"
    license: str = ""
    tags: List[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class Document:
    """Document structure for RAG."""

    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ChunkResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    section_title: str = ""
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """Search result with relevance score."""

    document: Document
    score: float
    rank: int
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    combined_score: Optional[float] = None
    rerank_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Text pre-processing
# ---------------------------------------------------------------------------


class ChineseTextProcessor:
    """Chinese text processing utilities."""

    def __init__(self):
        try:
            self.cc = opencc.OpenCC("t2s")  # Traditional to Simplified
        except Exception:  # noqa: BLE001
            logger.warning("OpenCC not available, using basic text processing")
            self.cc = None

    def normalize_text(self, text: str) -> str:
        """Normalize Chinese/English mixed text."""
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        # Keep Chinese, English, digits and basic punctuation
        text = re.sub(r"[^\w\s\u4e00-\u9fff.,!?;:]", "", text)

        return text.strip()

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> List[str]:
        """Split text into overlapping chunks (sentence-based for Chinese)."""
        if not text:
            return []

        text = self.normalize_text(text)

        # Split by sentence / newline
        sentences = re.split(r"[。！？\n]", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[str] = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Very simple overlap: reuse tail of previous chunk
        final_chunks: List[str] = []
        for i, chunk in enumerate(chunks):
            if i > 0 and overlap > 0:
                prev_words = chunks[i - 1].split()[-overlap:]
                overlap_text = " ".join(prev_words)
                chunk = (overlap_text + " " + chunk).strip()
            final_chunks.append(chunk)

        return final_chunks


# ---------------------------------------------------------------------------
# RAG engine
# ---------------------------------------------------------------------------


class ChineseRAGEngine:
    """RAG engine optimized for Chinese/English mixed content."""

    def __init__(self, embedding_model: Optional[str] = None):
        self.config = get_config()
        self.cache = get_shared_cache()

        # Model configuration
        rag_cfg = getattr(self.config, "rag", None)
        self.embedding_model_name = (
            embedding_model
            or getattr(rag_cfg, "embedding_model", None)
            or self.config.model.embedding_model
        )  # type: ignore[attr-defined]
        self.reranker_model_name = str(getattr(rag_cfg, "reranker_model", "") or "").strip()
        self.enable_rerank = bool(getattr(rag_cfg, "enable_rerank", False))
        self.rerank_top_k = int(getattr(rag_cfg, "rerank_top_k", 0) or 0)
        self.hybrid_alpha = float(getattr(rag_cfg, "hybrid_alpha", 1.0) or 1.0)
        self._embedding_device = _resolve_device(getattr(rag_cfg, "device", "cpu"))
        self._reranker = CrossEncoderReranker(
            self.reranker_model_name,
            cache_dir=Path(self.cache.get_path("MODELS_RERANKER")),
            device=str(getattr(rag_cfg, "reranker_device", "cpu") or "cpu"),
            max_seq_length=int(getattr(rag_cfg, "max_seq_length", 512) or 512),
        )
        self._embedding_model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._loaded: bool = False

        # Document storage
        self.documents: Dict[str, Document] = {}
        self.index: Optional[faiss.IndexFlatIP] = None
        # List mapping FAISS index row -> chunk doc_id
        self.doc_id_map: List[str] = []

        # Index parameters
        self.embedding_dim: int = 768  # default, will be overwritten on model load
        self.max_chunk_size: int = int(
            getattr(rag_cfg, "chunk_size", 500) or 500
        )  # characters per chunk

        # BM25 for optional lexical / hybrid search
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[str] = []

        # Text processor
        self.text_processor = ChineseTextProcessor()

    # ------------------------- Model loading / encoding ---------------------

    @handle_model_error
    def _load_model(self) -> bool:
        """Load embedding model if not already loaded."""
        global AutoModel, AutoTokenizer, SentenceTransformer
        if self._loaded:
            return True

        try:
            cache_dir = self.cache.get_path("MODELS_EMBEDDING")
            local_embedding_model = resolve_model_path(
                self.embedding_model_name,
                kind="embedding",
            )
            logger.info(
                "Loading embedding model: %s -> %s",
                self.embedding_model_name,
                local_embedding_model,
            )

            # Prefer SentenceTransformer for retrieval
            try:
                if SentenceTransformer is None:
                    from sentence_transformers import (
                        SentenceTransformer as _SentenceTransformer,
                    )

                    SentenceTransformer = _SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    local_embedding_model,
                    cache_folder=str(cache_dir),
                    device=self._embedding_device,
                )
                self.embedding_dim = self._embedding_model.get_sentence_embedding_dimension()
                self._tokenizer = None
                logger.info("Loaded SentenceTransformer model, dim=%d", self.embedding_dim)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "SentenceTransformer load failed (%s), falling back to AutoModel",
                    e,
                )

                if AutoTokenizer is None or AutoModel is None:
                    from transformers import AutoModel as _AutoModel
                    from transformers import AutoTokenizer as _AutoTokenizer

                    AutoModel = _AutoModel
                    AutoTokenizer = _AutoTokenizer
                if torch is None:
                    raise RuntimeError("torch embedding runtime is unavailable")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    local_embedding_model,
                    cache_dir=str(cache_dir),
                    local_files_only=True,
                )
                self._embedding_model = AutoModel.from_pretrained(
                    local_embedding_model,
                    cache_dir=str(cache_dir),
                    local_files_only=True,
                    torch_dtype=(torch.float16 if self.config.model.use_fp16 else torch.float32),
                    device_map="auto"
                    if torch.cuda.is_available() and self._embedding_device != "cpu"
                    else None,
                )

                # Use hidden_size as embedding dimension
                self.embedding_dim = int(self._embedding_model.config.hidden_size)
                logger.info("Loaded AutoModel, dim=%d", self.embedding_dim)

            self._loaded = True
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load embedding model: %s", e)
            raise EmbeddingError(f"Model loading failed: {e}")

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to a normalized embedding vector."""
        if not self._loaded:
            self._load_model()

        def _do_encode() -> np.ndarray:
            try:
                if SentenceTransformer is not None and isinstance(
                    self._embedding_model, SentenceTransformer
                ):
                    embedding = self._embedding_model.encode(
                        text,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                    return embedding.astype(np.float32)

                # Transformers path
                assert self._tokenizer is not None
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )

                device = next(self._embedding_model.parameters()).device  # type: ignore[union-attr]
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._embedding_model(**inputs)  # type: ignore[operator]
                    # Mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

                return embedding.cpu().numpy().astype(np.float32)
            except Exception as e:  # noqa: BLE001
                logger.error("Text encoding failed: %s", e)
                raise EmbeddingError(f"Encoding failed: {e}")

        if self._embedding_device == "cuda" and torch.cuda.is_available():
            try:
                from core.runtime import get_model_runtime

                runtime = get_model_runtime()
                with runtime.exclusive_gpu(reason="rag.embed", device="cuda"):
                    return _do_encode()
            except Exception:
                return _do_encode()

        return _do_encode()

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Backwards‑compatible wrapper for generating an embedding.

        以前這個 function 直接用 tokenizer + AutoModel，
        現在統一走 _encode_text，避免 SentenceTransformer 路徑炸掉。
        """
        return self._encode_text(text)

    def _create_index(self) -> faiss.IndexFlatIP:
        """Create a new FAISS index (Inner Product for cosine similarity)."""
        if not self._loaded:
            self._load_model()

        index = faiss.IndexFlatIP(self.embedding_dim)
        logger.info("Created FAISS index with dimension %d", self.embedding_dim)
        return index

    # ----------------------------- Ingestion --------------------------------

    def _chunk_document(self, content: str, doc_id: str) -> List[Document]:
        """
        Split document into chunks for indexing.

        目前維持原本的「字數 chunk + 句號切割」，讓舊資料行為一致。
        如果之後要改用 ChineseTextProcessor.chunk_text 也可以在這裡切換。
        """
        chunks: List[Document] = []

        overlap = 50
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + self.max_chunk_size
            chunk_text = content[start:end]

            # Try to break at sentence boundary
            if end < len(content):
                last_period = chunk_text.rfind("。")
                last_question = chunk_text.rfind("？")
                last_exclamation = chunk_text.rfind("！")
                last_newline = chunk_text.rfind("\n")

                break_point = max(last_period, last_question, last_exclamation, last_newline)
                if break_point > 100:  # minimum chunk size
                    chunk_text = chunk_text[: break_point + 1]
                    end = start + len(chunk_text)

            chunk_id = f"{doc_id}_chunk_{chunk_idx}"
            chunks.append(
                Document(
                    doc_id=chunk_id,
                    content=chunk_text.strip(),
                    metadata={
                        "parent_doc_id": doc_id,
                        "chunk_index": chunk_idx,
                        "start_char": start,
                        "end_char": end,
                    },
                    created_at=datetime.now(),
                )
            )

            start = end - overlap
            chunk_idx += 1

        return chunks

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a logical document (split into chunks) to the RAG index."""
        try:
            if not content.strip():
                raise DocumentIndexError(doc_id, "Empty content")

            base_metadata: Dict[str, Any] = dict(metadata or {})
            if base_metadata and base_metadata.get("title") is None:
                # Prefer original filename as a display title when available.
                try:
                    base_metadata["title"] = (
                        base_metadata.get("original_filename")
                        or base_metadata.get("file_name")
                        or doc_id
                    )
                except Exception:
                    base_metadata["title"] = doc_id

            # If doc_id already exists (as a logical parent), remove old chunks first to avoid FAISS/doc_id_map drift.
            replaced = False
            try:
                replaced = bool(self.remove_document(doc_id))
                if replaced:
                    logger.info(
                        "Replacing existing document %s (removed old chunks, will rebuild index)",
                        doc_id,
                    )
            except Exception:
                replaced = False

            # Split into chunks
            chunks = self._chunk_document(content, doc_id)
            if not chunks:
                raise DocumentIndexError(doc_id, "No chunks generated from content")

            # Generate embeddings for chunks
            chunk_embeddings: List[np.ndarray] = []
            for chunk in chunks:
                # Merge caller metadata (world_id, tags, source...) into every chunk.
                # Keep chunk-specific fields (parent_doc_id, chunk_index...) taking precedence.
                try:
                    chunk.metadata = {**base_metadata, **(chunk.metadata or {})}
                except Exception:
                    pass
                embedding = self._generate_embedding(chunk.content)
                chunk.embedding = embedding
                chunk_embeddings.append(embedding)

                # Store chunk in document store
                self.documents[chunk.doc_id] = chunk

            # Keep FAISS/BM25/doc_id_map consistent.
            # If we replaced an existing document, we must rebuild to remove stale embeddings.
            if replaced or self.index is None or not self.doc_id_map:
                self.rebuild_index()
            else:
                # Normalize embeddings for cosine similarity
                embeddings_matrix = np.asarray(chunk_embeddings, dtype=np.float32)
                faiss.normalize_L2(embeddings_matrix)

                # Add to index & doc_id_map
                self.index.add(embeddings_matrix)  # type: ignore[union-attr]
                for chunk in chunks:
                    self.doc_id_map.append(chunk.doc_id)

                # Keep BM25 in sync
                self._update_bm25()

            logger.info("Added document %s with %d chunks", doc_id, len(chunks))
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to add document %s: %s", doc_id, e)
            raise DocumentIndexError(doc_id, str(e))

    # ----------------------------- Retrieval --------------------------------

    def _update_bm25(self) -> None:
        """Update BM25 index with current documents."""
        try:
            self.bm25_corpus = [
                self.text_processor.normalize_text(doc.content) for doc in self.documents.values()
            ]

            if self.bm25_corpus:
                tokenized_corpus = [doc.split() for doc in self.bm25_corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
                logger.debug("Updated BM25 with %d documents", len(self.bm25_corpus))
            else:
                self.bm25 = None
        except Exception as e:  # noqa: BLE001
            logger.warning("BM25 update failed: %s", e)
            self.bm25 = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
        world_id: Optional[str] = None,
        enable_rerank: Optional[bool] = None,
        rerank_top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search for relevant chunks using:
        - semantic (FAISS)
        - optional BM25 hybrid fusion
        - optional reranker (CrossEncoder)
        """
        if self.index is None or not self.documents or not self.doc_id_map:
            logger.warning("No documents indexed for search")
            return []

        try:
            requested_k = max(1, int(top_k or 0))
            enable_rerank_flag = (
                bool(self.enable_rerank) if enable_rerank is None else bool(enable_rerank)
            )
            enable_rerank_stage = bool(enable_rerank_flag) and bool(self.reranker_model_name)
            rerank_top = (
                int(self.rerank_top_k or 0) if rerank_top_k is None else int(rerank_top_k or 0)
            )
            rerank_top = max(0, rerank_top)
            rerank_k = max(requested_k, rerank_top) if enable_rerank_stage else requested_k
            max_candidates = len(self.doc_id_map)
            candidate_k = max(1, min(int(rerank_k), int(max_candidates)))

            target_world = str(world_id or "").strip() or None

            # When world_id is specified, we need to over-fetch candidates because FAISS cannot
            # filter by metadata. Keep rerank window bounded by candidate_k, but increase the
            # retrieval window to avoid "0 hit" caused by cross-world top-k domination.
            retrieval_k = candidate_k
            if target_world:
                oversample = 8
                retrieval_k = min(max_candidates, max(retrieval_k, int(candidate_k * oversample)))

            semantic_results = self._semantic_search(query, top_k=retrieval_k)
            use_bm25 = bool(self.bm25) and float(self.hybrid_alpha) < 1.0
            bm25_results = self._bm25_search(query, top_k=retrieval_k) if use_bm25 else []

            merged: Dict[str, SearchResult] = {}

            def _accept(doc: Document) -> bool:
                if not target_world:
                    return True
                metadata = doc.metadata or {}
                return str(metadata.get("world_id", "default")).strip() == target_world

            for res in semantic_results:
                doc = res.document
                if not _accept(doc):
                    continue
                merged[doc.doc_id] = SearchResult(
                    document=doc,
                    score=float(res.score),
                    rank=0,
                    semantic_score=float(res.score),
                )

            for res in bm25_results:
                doc = res.document
                if not _accept(doc):
                    continue
                item = merged.get(doc.doc_id)
                if item is None:
                    merged[doc.doc_id] = SearchResult(
                        document=doc,
                        score=float(res.score),
                        rank=0,
                        semantic_score=None,
                        bm25_score=float(res.score),
                    )
                else:
                    item.bm25_score = float(res.score)

            if not merged:
                return []

            max_bm25 = 0.0
            if use_bm25:
                max_bm25 = max((x.bm25_score or 0.0) for x in merged.values())
                if not np.isfinite(max_bm25):
                    max_bm25 = 0.0

            alpha = float(self.hybrid_alpha)
            if not np.isfinite(alpha):
                alpha = 1.0
            alpha = min(1.0, max(0.0, alpha))

            candidates: List[SearchResult] = []
            for item in merged.values():
                semantic_score = float(item.semantic_score or 0.0)
                if not np.isfinite(semantic_score):
                    semantic_score = 0.0
                semantic_score = max(0.0, semantic_score)

                bm25_score = float(item.bm25_score or 0.0)
                if not np.isfinite(bm25_score):
                    bm25_score = 0.0
                bm25_norm = (bm25_score / max_bm25) if (use_bm25 and max_bm25 > 0) else 0.0

                combined = (alpha * semantic_score) + ((1.0 - alpha) * bm25_norm)

                item.semantic_score = semantic_score
                item.bm25_score = bm25_score if use_bm25 else None
                item.combined_score = combined
                item.score = combined

                if combined >= float(min_score or 0.0):
                    candidates.append(item)

            if not candidates:
                return []

            candidates.sort(key=lambda x: float(x.combined_score or x.score), reverse=True)

            # Optional reranker stage (best-effort; falls back to combined score)
            if enable_rerank_stage and self._reranker.is_enabled():
                rerank_window = min(candidate_k, len(candidates))
                passages = [c.document.content for c in candidates[:rerank_window]]
                rerank_scores = self._reranker.rerank(query, passages)
                if rerank_scores and len(rerank_scores) == len(passages):
                    for item, score in zip(candidates[:rerank_window], rerank_scores):
                        item.rerank_score = float(score)
                        item.score = float(score)
                    candidates[:rerank_window] = sorted(
                        candidates[:rerank_window],
                        key=lambda x: float(x.rerank_score or x.score),
                        reverse=True,
                    )

            results = candidates[:requested_k]
            for i, item in enumerate(results, start=1):
                item.rank = i

            logger.info("Search '%s...' -> %d results", query[:50], len(results))
            return results
        except Exception as e:  # noqa: BLE001
            logger.error("Search failed for query '%s...': %s", query[:50], e)
            raise RAGError(f"Search failed: {str(e)}")

    def _semantic_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Semantic search using FAISS (internal helper)."""
        if not self.index or not self.documents or not self.doc_id_map:
            return []

        try:
            query_embedding = self._encode_text(query)
            query_vector = np.asarray([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            scores, indices = self.index.search(query_vector, top_k)  # type: ignore[arg-type]
            results: List[SearchResult] = []

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.doc_id_map):
                    continue
                chunk_id = self.doc_id_map[idx]
                document = self.documents.get(chunk_id)
                if not document:
                    continue
                results.append(
                    SearchResult(
                        document=document,
                        score=float(score),
                        rank=0,
                    )
                )
            return results
        except Exception as e:  # noqa: BLE001
            logger.error("Semantic search failed: %s", e)
            return []

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        """BM25 lexical search over raw chunk texts."""
        if not self.bm25 or not self.bm25_corpus or not self.documents:
            return []

        try:
            query_tokens = self.text_processor.normalize_text(query).split()
            scores = self.bm25.get_scores(query_tokens)  # type: ignore[arg-type]

            top_indices = np.argsort(scores)[-top_k:][::-1]
            doc_ids = list(self.documents.keys())

            results: List[SearchResult] = []
            for idx in top_indices:
                if idx < 0 or idx >= len(doc_ids):
                    continue
                if scores[idx] <= 0:
                    continue

                chunk_id = doc_ids[idx]
                document = self.documents.get(chunk_id)
                if not document:
                    continue

                results.append(
                    SearchResult(
                        document=document,
                        score=float(scores[idx]),
                        rank=0,
                    )
                )

            return results
        except Exception as e:  # noqa: BLE001
            logger.error("BM25 search failed: %s", e)
            return []

    def generate_context(
        self,
        query: str,
        max_context_length: int = 2000,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate context for RAG-enhanced generation.

        目前使用 semantic search 的 top_k 結果，串成加註編號的 context，
        並附上 sources metadata。
        """
        try:
            search_results = self.search(query, top_k=top_k)
            if not search_results:
                return {"context": "", "sources": [], "total_chars": 0}

            context_parts: List[str] = []
            sources: List[Dict[str, Any]] = []
            total_chars = 0

            for idx, result in enumerate(search_results, start=1):
                content = result.document.content

                # Check length limit
                if total_chars + len(content) > max_context_length:
                    remaining = max_context_length - total_chars
                    if remaining > 100:
                        content = content[:remaining] + "..."
                    else:
                        break

                context_parts.append(f"[相關資料 {idx}]\n{content}\n")
                sources.append(
                    {
                        "doc_id": result.document.doc_id,
                        "score": result.score,
                        "metadata": result.document.metadata,
                    }
                )
                total_chars += len(content)

            context = "\n".join(context_parts)
            return {"context": context, "sources": sources, "total_chars": total_chars}
        except Exception as e:  # noqa: BLE001
            logger.error("Context generation failed: %s", e)
            return {"context": "", "sources": [], "total_chars": 0}

    # ----------------------------- Maintenance ------------------------------

    def rebuild_index(self) -> bool:
        """Rebuild FAISS index and BM25 from current documents."""
        try:
            if not self.documents:
                logger.warning("No documents to rebuild index")
                self.index = None
                self.doc_id_map = []
                self.bm25 = None
                self.bm25_corpus = []
                return False

            self.index = self._create_index()
            self.doc_id_map = []

            embeddings: List[np.ndarray] = []
            doc_ids: List[str] = []

            for doc_id, document in self.documents.items():
                if document.embedding is None:
                    document.embedding = self._encode_text(document.content)
                embeddings.append(document.embedding)
                doc_ids.append(doc_id)

            if embeddings:
                embeddings_array = np.vstack(embeddings).astype(np.float32)
                faiss.normalize_L2(embeddings_array)
                self.index.add(embeddings_array)  # type: ignore[arg-type]
                self.doc_id_map.extend(doc_ids)

            self._update_bm25()

            logger.info("Rebuilt index with %d chunks", len(embeddings))
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Index rebuild failed: %s", e)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics."""
        unique_docs = {
            doc.metadata.get("parent_doc_id", doc.doc_id) for doc in self.documents.values()
        }

        return {
            "total_documents": len(unique_docs),
            "total_chunks": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "model_loaded": self._loaded,
            "bm25_enabled": self.bm25 is not None,
            "enable_rerank": bool(self.enable_rerank),
            "reranker_model": self.reranker_model_name,
        }

    def save_index(self, filepath: Optional[Path] = None) -> Path:
        """Save FAISS index and metadata to disk."""
        if filepath is None:
            filepath = self.cache.get_output_path("rag") / "index"

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        try:
            if self.index is not None:
                faiss.write_index(self.index, str(filepath / "faiss.index"))

            metadata = {
                "doc_id_map": self.doc_id_map,
                "embedding_model": self.embedding_model_name,
                "embedding_dim": self.embedding_dim,
                "created_at": datetime.now().isoformat(),
                "total_documents": len(self.documents),
            }

            with open(filepath / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            documents_data: Dict[str, Any] = {}
            for doc_id, doc in self.documents.items():
                documents_data[doc_id] = {
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                }

            with open(filepath / "documents.json", "w", encoding="utf-8") as f:
                json.dump(documents_data, f, indent=2, ensure_ascii=False)

            logger.info("RAG index saved to %s", filepath)
            return filepath
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to save RAG index: %s", e)
            raise RAGError(f"Index save failed: {e}")

    def load_index(self, filepath: Path) -> bool:
        """Load FAISS index and metadata from disk."""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.warning("Index path does not exist: %s", filepath)
                return False

            metadata_file = filepath / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                doc_id_map_raw = metadata.get("doc_id_map", [])
                if isinstance(doc_id_map_raw, dict):
                    # Old format: {index: doc_id}
                    self.doc_id_map = [
                        doc_id_map_raw[str(i)]
                        for i in sorted((int(k) for k in doc_id_map_raw.keys()))
                    ]
                else:
                    self.doc_id_map = list(doc_id_map_raw)

                self.embedding_dim = metadata.get("embedding_dim", self.embedding_dim)

                if metadata.get("embedding_model") != self.embedding_model_name:
                    logger.warning(
                        "Embedding model mismatch: loaded=%s, current=%s. "
                        "Index rebuild is recommended.",
                        metadata.get("embedding_model"),
                        self.embedding_model_name,
                    )

            faiss_file = filepath / "faiss.index"
            if faiss_file.exists():
                self.index = faiss.read_index(str(faiss_file))

            documents_file = filepath / "documents.json"
            if documents_file.exists():
                with open(documents_file, "r", encoding="utf-8") as f:
                    documents_data = json.load(f)

                for doc_id, doc_data in documents_data.items():
                    created_at = (
                        datetime.fromisoformat(doc_data["created_at"])
                        if doc_data.get("created_at")
                        else None
                    )
                    self.documents[doc_id] = Document(
                        doc_id=doc_data["doc_id"],
                        content=doc_data["content"],
                        metadata=doc_data["metadata"],
                        created_at=created_at,
                    )

            self._update_bm25()

            logger.info(
                "RAG index loaded from %s, %d documents",
                filepath,
                len(self.documents),
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load RAG index from %s: %s", filepath, e)
            return False

    # ----------------------------- Utilities --------------------------------

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by chunk ID (or full doc if stored as single chunk)."""
        return self.documents.get(doc_id)

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a logical document and its chunks from the in-memory store.

        注意：FAISS index 需要另外呼叫 rebuild_index() 才會同步。
        """
        chunks_to_remove = [
            chunk_id
            for chunk_id, doc in self.documents.items()
            if doc.metadata.get("parent_doc_id") == doc_id or doc.doc_id == doc_id
        ]

        if not chunks_to_remove:
            return False

        for chunk_id in chunks_to_remove:
            self.documents.pop(chunk_id, None)

        logger.warning(
            "Removed %d chunks for doc %s from documents store. "
            "Call rebuild_index() to refresh FAISS index.",
            len(chunks_to_remove),
            doc_id,
        )
        return True


# ---------------------------------------------------------------------------
# Global RAG engine instance
# ---------------------------------------------------------------------------

_rag_engine: Optional[ChineseRAGEngine] = None


def get_rag_engine(embedding_model: Optional[str] = None) -> ChineseRAGEngine:
    """Get global RAG engine instance (singleton)."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = ChineseRAGEngine(embedding_model)

        # Try to load existing index from shared cache
        cache = get_shared_cache()
        index_path = cache.get_output_path("rag") / "index"  # type: ignore[attr-defined]

        if index_path.exists():
            try:
                _rag_engine.load_index(index_path)
                logger.info("Existing RAG index loaded from %s", index_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to load existing index: %s", e)

    return _rag_engine


class DocumentMemory:
    """Simple document memory wrapper around the global RAG engine."""

    def __init__(self):
        self.rag_engine = get_rag_engine()

    def add(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add document to memory."""
        self.rag_engine.add_document(doc_id, content, metadata or {})

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents and return simplified dict results."""
        results = self.rag_engine.search(query, top_k=top_k)
        return [
            {
                "content": result.document.content,
                "score": result.score,
                "metadata": result.document.metadata,
            }
            for result in results
        ]
