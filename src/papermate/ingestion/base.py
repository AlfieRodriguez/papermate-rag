"""Base interfaces for document ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod

from papermate.schemas import PageText


class DocumentLoader(ABC):
    """Interface for loading documents into page-level text."""

    @abstractmethod
    def load(self, file_path: str, doc_id: str) -> list[PageText]:
        """Load a document and return text for each page."""
        raise NotImplementedError
