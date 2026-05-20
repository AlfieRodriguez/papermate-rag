"""Coordinate query embedding and vector search."""

from __future__ import annotations

from papermate.embeddings.base import BaseEmbedder
from papermate.schemas import RetrievedChunk
from papermate.vectorstores.base import BaseVectorStore


class Retriever:
    """Retrieve relevant chunks for a text query."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed a query and search the vector store."""

        if not query.strip():
            raise ValueError("query must not be empty")

        effective_top_k = self.top_k if top_k is None else top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be > 0")

        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.similarity_search(
            query_embedding,
            top_k=effective_top_k,
        )
