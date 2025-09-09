# core/rag/retriever.py

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

from .vector_store import VectorStore, VectorMetadata
from .engine import ChineseTextProcessor, SearchResult, Document
from ..exceptions import RAGError

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result"""

    doc_id: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any]
    retrieval_method: str  # 'semantic', 'bm25', 'hybrid'


@dataclass
class RetrievalQuery:
    """Retrieval query with parameters"""

    text: str
    top_k: int = 10
    min_score: float = 0.1
    filters: Optional[Dict[str, Any]] = None
    boost_factors: Optional[Dict[str, float]] = None


class BaseRetriever(ABC):
    """Base class for retrievers"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve relevant documents"""
        pass


class SemanticRetriever(BaseRetriever):
    """Semantic retrieval using vector similarity"""

    def __init__(self, vector_store: VectorStore, embedding_function: callable):
        super().__init__("semantic")
        self.vector_store = vector_store
        self.embedding_function = embedding_function

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve using semantic similarity"""
        try:
            # Generate query embedding
            query_vector = self.embedding_function(query.text)

            # Search vector store
            search_results = self.vector_store.search(
                query_vector=query_vector, top_k=query.top_k, min_score=query.min_score
            )

            # Convert to retrieval results
            results = []
            for rank, (index_id, score, metadata) in enumerate(search_results):
                # Apply filters if specified
                if self._passes_filters(metadata, query.filters):
                    # Apply boost factors
                    boosted_score = self._apply_boost(
                        score, metadata, query.boost_factors
                    )

                    results.append(
                        RetrievalResult(
                            doc_id=metadata.doc_id,
                            content=self._get_content_from_metadata(metadata),
                            score=boosted_score,
                            rank=rank + 1,
                            metadata=metadata.metadata,
                            retrieval_method="semantic",
                        )
                    )

            # Re-sort if boost factors were applied
            if query.boost_factors:
                results.sort(key=lambda x: x.score, reverse=True)
                for i, result in enumerate(results):
                    result.rank = i + 1

            return results

        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            raise RAGError(f"Semantic retrieval error: {e}")

    def _passes_filters(
        self, metadata: VectorMetadata, filters: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if result passes filters"""
        if not filters:
            return True

        for key, expected_value in filters.items():
            if key in metadata.metadata:
                actual_value = metadata.metadata[key]

                # Handle different filter types
                if isinstance(expected_value, list):
                    # Value must be in list
                    if actual_value not in expected_value:
                        return False
                elif isinstance(expected_value, dict):
                    # Range or comparison filters
                    if "min" in expected_value and actual_value < expected_value["min"]:
                        return False
                    if "max" in expected_value and actual_value > expected_value["max"]:
                        return False
                else:
                    # Exact match
                    if actual_value != expected_value:
                        return False

        return True

    def _apply_boost(
        self,
        score: float,
        metadata: VectorMetadata,
        boost_factors: Optional[Dict[str, float]],
    ) -> float:
        """Apply boost factors to score"""
        if not boost_factors:
            return score

        boosted_score = score

        for factor_key, boost_value in boost_factors.items():
            if factor_key in metadata.metadata:
                # Apply multiplicative boost
                boosted_score *= boost_value
            elif factor_key == "recency":
                # Recency boost based on timestamp
                days_old = (datetime.now() - metadata.timestamp).days
                recency_factor = max(0.1, 1.0 - (days_old / 365.0))  # Decay over a year
                boosted_score *= 1.0 + boost_value * recency_factor

        return boosted_score

    def _get_content_from_metadata(self, metadata: VectorMetadata) -> str:
        """Extract content from metadata - this would need to be implemented based on your data structure"""
        # This is a placeholder - you'd need to implement based on how you store content
        return metadata.metadata.get("content", f"[Content for {metadata.doc_id}]")


class BM25Retriever(BaseRetriever):
    """BM25 lexical retrieval"""

    def __init__(self, documents: Dict[str, Document]):
        super().__init__("bm25")
        self.documents = documents
        self.text_processor = ChineseTextProcessor()
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index from documents"""
        try:
            from rank_bm25 import BM25Okapi

            self.doc_ids = list(self.documents.keys())
            corpus = []

            for doc_id in self.doc_ids:
                doc = self.documents[doc_id]
                processed_text = self.text_processor.normalize_text(doc.content)
                corpus.append(processed_text.split())

            self.bm25 = BM25Okapi(corpus)
            logger.info(f"Built BM25 index with {len(corpus)} documents")

        except ImportError:
            logger.error("rank_bm25 not available")
            self.bm25 = None
        except Exception as e:
            logger.error(f"BM25 index building failed: {e}")
            self.bm25 = None

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve using BM25 scoring"""
        if not self.bm25:
            logger.warning("BM25 not available")
            return []

        try:
            # Tokenize query
            query_tokens = self.text_processor.normalize_text(query.text).split()

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Get top results
            top_indices = np.argsort(scores)[-query.top_k :][::-1]

            results = []
            for rank, idx in enumerate(top_indices):
                if idx < len(self.doc_ids) and scores[idx] >= query.min_score:
                    doc_id = self.doc_ids[idx]
                    doc = self.documents[doc_id]

                    # Apply filters
                    if self._passes_filters(doc, query.filters):
                        # Apply boost factors
                        boosted_score = self._apply_boost(
                            scores[idx], doc, query.boost_factors
                        )

                        results.append(
                            RetrievalResult(
                                doc_id=doc_id,
                                content=doc.content,
                                score=boosted_score,
                                rank=rank + 1,
                                metadata=doc.metadata,
                                retrieval_method="bm25",
                            )
                        )

            return results

        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            raise RAGError(f"BM25 retrieval error: {e}")

    def _passes_filters(self, doc: Document, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if document passes filters"""
        if not filters:
            return True

        for key, expected_value in filters.items():
            if key in doc.metadata:
                actual_value = doc.metadata[key]

                if isinstance(expected_value, list):
                    if actual_value not in expected_value:
                        return False
                elif isinstance(expected_value, dict):
                    if "min" in expected_value and actual_value < expected_value["min"]:
                        return False
                    if "max" in expected_value and actual_value > expected_value["max"]:
                        return False
                else:
                    if actual_value != expected_value:
                        return False

        return True

    def _apply_boost(
        self, score: float, doc: Document, boost_factors: Optional[Dict[str, float]]
    ) -> float:
        """Apply boost factors to BM25 score"""
        if not boost_factors:
            return score

        boosted_score = score

        for factor_key, boost_value in boost_factors.items():
            if factor_key in doc.metadata:
                boosted_score *= boost_value
            elif factor_key == "recency" and doc.created_at:
                days_old = (datetime.now() - doc.created_at).days
                recency_factor = max(0.1, 1.0 - (days_old / 365.0))
                boosted_score *= 1.0 + boost_value * recency_factor

        return boosted_score


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining multiple methods"""

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        fusion_method: str = "rrf",  # "rrf", "weighted", "max"
        weights: Optional[List[float]] = None,
    ):
        super().__init__("hybrid")
        self.retrievers = retrievers
        self.fusion_method = fusion_method
        self.weights = weights or [1.0] * len(retrievers)

        if len(self.weights) != len(self.retrievers):
            raise ValueError("Number of weights must match number of retrievers")

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve using hybrid fusion"""
        try:
            # Get results from all retrievers
            all_results = {}  # doc_id -> {retriever_name: result}

            for retriever, weight in zip(self.retrievers, self.weights):
                try:
                    results = retriever.retrieve(query)

                    for result in results:
                        doc_id = result.doc_id
                        if doc_id not in all_results:
                            all_results[doc_id] = {}

                        # Store result with weight
                        all_results[doc_id][retriever.name] = {
                            "result": result,
                            "weight": weight,
                        }

                except Exception as e:
                    logger.warning(f"Retriever {retriever.name} failed: {e}")
                    continue

            # Fuse results
            if self.fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(all_results, query.top_k)
            elif self.fusion_method == "weighted":
                fused_results = self._weighted_fusion(all_results, query.top_k)
            elif self.fusion_method == "max":
                fused_results = self._max_fusion(all_results, query.top_k)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

            return fused_results

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            raise RAGError(f"Hybrid retrieval error: {e}")

    def _reciprocal_rank_fusion(
        self, all_results: Dict[str, Dict], top_k: int, k: int = 60
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion (RRF)"""
        fused_scores = {}

        for doc_id, retriever_results in all_results.items():
            fused_score = 0.0
            best_result = None

            for retriever_name, data in retriever_results.items():
                result = data["result"]
                weight = data["weight"]

                # RRF score: weight / (k + rank)
                rrf_score = weight / (k + result.rank)
                fused_score += rrf_score

                # Keep the best result for metadata
                if best_result is None or result.score > best_result.score:
                    best_result = result

            if best_result:
                fused_scores[doc_id] = {"score": fused_score, "result": best_result}

        # Sort and create final results
        sorted_results = sorted(
            fused_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )

        final_results = []
        for rank, (doc_id, data) in enumerate(sorted_results[:top_k]):
            result = data["result"]
            result.score = data["score"]
            result.rank = rank + 1
            result.retrieval_method = "hybrid_rrf"
            final_results.append(result)

        return final_results

    def _weighted_fusion(
        self, all_results: Dict[str, Dict], top_k: int
    ) -> List[RetrievalResult]:
        """Weighted score fusion"""
        fused_scores = {}

        for doc_id, retriever_results in all_results.items():
            fused_score = 0.0
            best_result = None

            for retriever_name, data in retriever_results.items():
                result = data["result"]
                weight = data["weight"]

                # Weighted score
                weighted_score = weight * result.score
                fused_score += weighted_score

                if best_result is None or result.score > best_result.score:
                    best_result = result

            if best_result:
                fused_scores[doc_id] = {"score": fused_score, "result": best_result}

        # Sort and create final results
        sorted_results = sorted(
            fused_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )

        final_results = []
        for rank, (doc_id, data) in enumerate(sorted_results[:top_k]):
            result = data["result"]
            result.score = data["score"]
            result.rank = rank + 1
            result.retrieval_method = "hybrid_weighted"
            final_results.append(result)

        return final_results

    def _max_fusion(
        self, all_results: Dict[str, Dict], top_k: int
    ) -> List[RetrievalResult]:
        """Max score fusion"""
        fused_scores = {}

        for doc_id, retriever_results in all_results.items():
            max_score = 0.0
            best_result = None

            for retriever_name, data in retriever_results.items():
                result = data["result"]

                if result.score > max_score:
                    max_score = result.score
                    best_result = result

            if best_result:
                fused_scores[doc_id] = {"score": max_score, "result": best_result}

        # Sort and create final results
        sorted_results = sorted(
            fused_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )

        final_results = []
        for rank, (doc_id, data) in enumerate(sorted_results[:top_k]):
            result = data["result"]
            result.score = data["score"]
            result.rank = rank + 1
            result.retrieval_method = "hybrid_max"
            final_results.append(result)

        return final_results


class AdvancedRetriever:
    """Advanced retriever with multiple strategies and reranking"""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: Optional[callable] = None,
        query_expansion: bool = False,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.query_expansion = query_expansion
        self.text_processor = ChineseTextProcessor()

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Advanced retrieval with optional query expansion and reranking"""
        try:
            # Query expansion
            if self.query_expansion:
                expanded_query = self._expand_query(query.text)
                query.text = expanded_query

            # Base retrieval
            results = self.base_retriever.retrieve(query)

            # Reranking
            if self.reranker and results:
                results = self._rerank_results(query.text, results)

            return results

        except Exception as e:
            logger.error(f"Advanced retrieval failed: {e}")
            raise RAGError(f"Advanced retrieval error: {e}")

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        # Simple query expansion - could be enhanced with word embeddings
        expanded_terms = []

        # Add original query
        expanded_terms.append(query)

        # Add common synonyms for Chinese terms
        synonym_map = {
            "问题": ["問題", "疑問", "課題"],
            "方法": ["辦法", "途徑", "手段"],
            "结果": ["結果", "成果", "效果"],
            "分析": ["解析", "研究", "檢視"],
            "数据": ["資料", "數據", "信息"],
        }

        for term, synonyms in synonym_map.items():
            if term in query:
                expanded_terms.extend(synonyms)

        return " ".join(expanded_terms)

    def _rerank_results(
        self, query: str, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank results using custom scoring"""
        if not self.reranker:
            return results

        try:
            # Apply reranker
            for result in results:
                rerank_score = self.reranker(query, result.content)
                # Combine original score with rerank score
                result.score = 0.7 * result.score + 0.3 * rerank_score

            # Re-sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)

            # Update ranks
            for i, result in enumerate(results):
                result.rank = i + 1

            return results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results
