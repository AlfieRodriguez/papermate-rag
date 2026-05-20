"""OpenAI embedding provider."""

from __future__ import annotations

from openai import OpenAI

from papermate.embeddings.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """Create embeddings with the OpenAI embeddings API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client: OpenAI | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with OpenAI."""

        if not texts:
            return []

        response = self._get_client().embeddings.create(
            model=self.model,
            input=texts,
        )
        return [list(item.embedding) for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        """Embed one query string."""

        if not query.strip():
            raise ValueError("query must not be empty")

        return self.embed_texts([query])[0]

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if self.api_key is None:
                self._client = OpenAI()
            else:
                self._client = OpenAI(api_key=self.api_key)
        return self._client
