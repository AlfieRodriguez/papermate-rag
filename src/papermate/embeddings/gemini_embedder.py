"""Gemini embedding provider."""

from __future__ import annotations

from typing import Any

from papermate.embeddings.base import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):
    """Create embeddings with the Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of non-empty texts."""

        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("text must not be empty")

        response = self._get_client().models.embed_content(
            model=self.model,
            contents=texts,
        )
        return self._vectors_from_response(response)

    def embed_query(self, query: str) -> list[float]:
        """Embed one query string."""

        if not query.strip():
            raise ValueError("query must not be empty")

        return self.embed_texts([query])[0]

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            if self.api_key is None:
                self._client = genai.Client()
            else:
                self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _vectors_from_response(self, response: Any) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            embedding = getattr(response, "embedding", None)
            embeddings = [embedding] if embedding is not None else []

        vectors: list[list[float]] = []
        for embedding in embeddings:
            if isinstance(embedding, dict):
                values = embedding.get("values", [])
            else:
                values = getattr(embedding, "values", embedding)
            vectors.append([float(value) for value in values])
        return vectors
