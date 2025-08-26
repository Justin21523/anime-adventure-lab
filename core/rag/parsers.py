# core/rag/parsers.py
import os
from pathlib import Path
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import PyPDF2


class DocumentParser(ABC):
    """Base class for document parsing"""

    @abstractmethod
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        pass


class SimpleParser:
    """Simple text document parser"""

    def parse_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """Parse text into chunks"""
        chunks = []

        # Simple sentence-based chunking
        sentences = text.split(". ")
        current_chunk = ""
        chunk_id = 0

        for sentence in sentences:
            if len(current_chunk + sentence) > chunk_size and current_chunk:
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id}",
                        "text": current_chunk.strip(),
                        "start_pos": len(text)
                        - len(". ".join(sentences[len(chunks) :])),
                        "end_pos": len(text) - len(". ".join(sentences[len(chunks) :])),
                    }
                )

                # Add overlap
                if overlap > 0:
                    current_chunk = sentence[-overlap:] + ". "
                else:
                    current_chunk = ""
                chunk_id += 1

            current_chunk += sentence + ". "

        # Add final chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_id}",
                    "text": current_chunk.strip(),
                    "start_pos": 0,
                    "end_pos": len(text),
                }
            )

        return chunks


class TextParser(DocumentParser):
    """Plain text parser"""

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple chunking by paragraphs
        chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

        return [
            {
                "content": chunk,
                "metadata": {"source": file_path, "chunk_id": i, "type": "text"},
            }
            for i, chunk in enumerate(chunks)
        ]


class PDFParser(DocumentParser):
    """PDF parser using PyPDF2"""

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            chunks = []

            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        chunks.append(
                            {
                                "content": text.strip(),
                                "metadata": {
                                    "source": file_path,
                                    "page": page_num + 1,
                                    "type": "pdf",
                                },
                            }
                        )
            return chunks
        except ImportError:
            # Fallback to text parsing if PyPDF2 not available
            return TextParser().parse(file_path)


def get_parser(file_path: str) -> DocumentParser:
    """Get appropriate parser based on file extension"""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return PDFParser()
    else:
        return TextParser()
