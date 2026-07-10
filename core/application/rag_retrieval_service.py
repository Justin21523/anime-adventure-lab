from __future__ import annotations

import os
from typing import Any

from sqlalchemy import select

from core.application.embedding_service import cosine_similarity, embed_texts
from core.persistence.models import DocumentChunkRecord, DocumentRecord
from core.persistence.unit_of_work import UnitOfWork


class RagRetrievalService:
    def __init__(self, uow_factory=UnitOfWork) -> None:
        self.uow_factory = uow_factory

    def retrieve(
        self, *, world_id: str, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        limit = top_k or int(os.getenv("RAG_STORY_TOP_K", "4"))
        limit = min(10, max(1, limit))
        vector = embed_texts([query])[0]

        with self.uow_factory() as uow:
            assert uow.session is not None
            dialect = uow.session.bind.dialect.name if uow.session.bind else ""
            if dialect == "postgresql":
                distance = DocumentChunkRecord.embedding.cosine_distance(vector)
                rows = uow.session.execute(
                    select(
                        DocumentChunkRecord, DocumentRecord, distance.label("distance")
                    )
                    .join(
                        DocumentRecord,
                        DocumentRecord.id == DocumentChunkRecord.document_id,
                    )
                    .where(
                        DocumentChunkRecord.world_id == world_id,
                        DocumentRecord.status == "ready",
                        DocumentChunkRecord.embedding.is_not(None),
                    )
                    .order_by(distance)
                    .limit(limit)
                ).all()
                ranked = [
                    (chunk, document, 1.0 - float(distance_value))
                    for chunk, document, distance_value in rows
                ]
            else:
                rows = uow.session.execute(
                    select(DocumentChunkRecord, DocumentRecord)
                    .join(
                        DocumentRecord,
                        DocumentRecord.id == DocumentChunkRecord.document_id,
                    )
                    .where(
                        DocumentChunkRecord.world_id == world_id,
                        DocumentRecord.status == "ready",
                    )
                ).all()
                ranked = sorted(
                    (
                        (
                            chunk,
                            document,
                            cosine_similarity(vector, list(chunk.embedding or [])),
                        )
                        for chunk, document in rows
                    ),
                    key=lambda item: item[2],
                    reverse=True,
                )[:limit]

            return [
                {
                    "document_id": document.id,
                    "filename": document.filename,
                    "chunk_id": chunk.id,
                    "position": chunk.position,
                    "excerpt": chunk.content[:400],
                    "score": round(max(0.0, min(1.0, score)), 4),
                }
                for chunk, document, score in ranked
            ]
