"""Vector store interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from papermate.schemas import DocumentChunk, RetrievedChunk


class BaseVectorStore(ABC):
    """Interface for storing and searching embedded chunks."""

    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their corresponding embeddings."""

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return the most similar chunks for a query embedding."""
