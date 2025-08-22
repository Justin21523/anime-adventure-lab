# core/rag/chunkle.py
import os, pathlib, torch, json, hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Cell 2: Dependencies & Model Setup
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import opencc
import re
from pathlib import Path
import pickle
from urllib.parse import quote
import uuid

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")


class ChineseHierarchicalChunker:
    """Hierarchical chunker optimized for Chinese text"""

    def __init__(self, target_chars=700, overlap_chars=120):
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

        # Chinese sentence separators
        self.sent_separators = ["。", "！", "？", "；", "…", "\n\n"]
        self.para_separators = ["\n\n", "\n", "。", "！", "？"]

    def normalize_text(self, text: str) -> str:
        """Normalize Chinese text"""
        # Remove control characters but keep structure
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        # Normalize whitespace but preserve paragraphs
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n\n+", "\n\n", text)
        return text.strip()

    def split_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """Split text by markdown headers first"""
        sections = []
        current_section = {"level": 0, "title": "", "content": ""}

        lines = text.split("\n")
        for line in lines:
            # Detect markdown headers
            header_match = re.match(r"^(#{1,6})\s*(.+)", line)
            if header_match:
                # Save previous section
                if current_section["content"].strip():
                    sections.append(current_section.copy())

                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {"level": level, "title": title, "content": ""}
            else:
                current_section["content"] += line + "\n"

        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Main chunking function with hierarchical strategy"""
        normalized_text = self.normalize_text(text)
        sections = self.split_by_headers(normalized_text)

        chunks = []
        chunk_id = 0

        for section in sections:
            section_chunks = self._chunk_section(
                section["content"],
                doc_id,
                section_title=section["title"],
                start_chunk_id=chunk_id,
            )
            chunks.extend(section_chunks)
            chunk_id += len(section_chunks)

        return chunks

    def _chunk_section(
        self, text: str, doc_id: str, section_title: str = "", start_chunk_id: int = 0
    ) -> List[Dict[str, Any]]:
        """Chunk a single section with overlap"""
        if len(text) <= self.target_chars:
            return [
                {
                    "chunk_id": f"{doc_id}@{start_chunk_id:04d}",
                    "doc_id": doc_id,
                    "section_title": section_title,
                    "text": text,
                    "char_count": len(text),
                    "chunk_index": start_chunk_id,
                }
            ]

        chunks = []
        start = 0
        chunk_idx = start_chunk_id

        while start < len(text):
            end = min(start + self.target_chars, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                for sep in self.sent_separators:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + self.target_chars // 2:
                        end = last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}@{chunk_idx:04d}",
                        "doc_id": doc_id,
                        "section_title": section_title,
                        "text": chunk_text,
                        "char_count": len(chunk_text),
                        "chunk_index": chunk_idx,
                    }
                )
                chunk_idx += 1

            # Calculate next start with overlap
            start = max(start + 1, end - self.overlap_chars)

        return chunks


if __name__ == "__main__":
    # Test chunker
    chunker = ChineseHierarchicalChunker()
    sample_text = """
    # 繁體中文測試文檔

    ## 第一章：背景介紹
    這是一個測試文檔，用來驗證中文分段功能。文檔包含多個段落和章節，每個部分都有不同的內容長度。

    ## 第二章：技術細節
    RAG 系統需要能夠處理中文文本的特殊性，包括沒有空格的句子邊界、繁簡轉換、以及語義相似性匹配。

    系統架構包括：文檔解析、分段處理、向量化、索引建立、檢索匹配等步驟。每個步驟都需要針對中文進行優化。
    """

    test_chunks = chunker.chunk_text(sample_text, "test_doc")
    print(f"[chunker] Generated {len(test_chunks)} chunks")
    for i, chunk in enumerate(test_chunks[:2]):
        print(f"Chunk {i}: {chunk['chunk_id']} | chars: {chunk['char_count']}")
        print(f"Text preview: {chunk['text'][:100]}...")
