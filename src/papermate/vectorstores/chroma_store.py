"""Chroma-backed vector store."""

from __future__ import annotations

import json
from typing import Any

import chromadb

from papermate.schemas import DocumentChunk, RetrievedChunk
from papermate.vectorstores.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """Store and search document chunks with Chroma."""

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "papermate_chunks",
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Add chunks and embeddings to Chroma."""

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._metadata_from_chunk(chunk) for chunk in chunks],
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Search by vector similarity.

        Chroma returns distances, so the distance is used directly as the score.
        """

        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not query_embedding or self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=str(metadata["doc_id"]),
                text=documents[index],
                page_start=int(metadata["page_start"]),
                page_end=int(metadata["page_end"]),
                source=str(metadata["source"]),
                metadata=self._chunk_metadata_from_chroma(metadata),
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(distances[index]),
                )
            )

        return retrieved

    def _metadata_from_chunk(self, chunk: DocumentChunk) -> dict[str, Any]:
        return {
            "doc_id": chunk.doc_id,
            "source": chunk.source,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "metadata_json": json.dumps(chunk.metadata, sort_keys=True, default=str),
        }

    def _chunk_metadata_from_chroma(self, metadata: dict[str, Any]) -> dict[str, Any]:
        metadata_json = metadata.get("metadata_json")
        if not metadata_json:
            return {}
        loaded = json.loads(str(metadata_json))
        if isinstance(loaded, dict):
            return loaded
        return {}
