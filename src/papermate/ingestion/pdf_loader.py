"""PDF text loader backed by PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz

from papermate.ingestion.base import DocumentLoader
from papermate.schemas import PageText


class PDFLoader(DocumentLoader):
    """Extract page-level text from PDF files."""

    def load(self, file_path: str, doc_id: str) -> list[PageText]:
        """Load a PDF and return one PageText object per page."""

        if not file_path or not file_path.strip():
            raise ValueError("file_path must not be empty")
        if not doc_id or not doc_id.strip():
            raise ValueError("doc_id must not be empty")
        doc_id = doc_id.strip()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {file_path}")
        if not path.is_file():
            raise ValueError(f"PDF path is not a file: {file_path}")

        try:
            with fitz.open(str(path)) as document:
                page_count = document.page_count
                metadata = {
                    "file_path": str(path),
                    "file_name": path.name,
                    "page_count": page_count,
                }
                return [
                    PageText(
                        doc_id=doc_id,
                        page_number=page_index + 1,
                        text=page.get_text().strip(),
                        metadata=metadata.copy(),
                    )
                    for page_index, page in enumerate(document)
                ]
        except FileNotFoundError:
            raise
        except Exception as error:
            raise ValueError(f"Could not open PDF file: {file_path}") from error
