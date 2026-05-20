import pytest

from papermate.embeddings import BaseEmbedder, OpenAIEmbedder


class FakeEmbedder(BaseEmbedder):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("query must not be empty")
        return self.embed_texts([query])[0]


def test_base_embedder_can_be_subclassed() -> None:
    embedder = FakeEmbedder()

    assert embedder.embed_texts(["abc", "d"]) == [[3.0, 1.0], [1.0, 1.0]]
    assert embedder.embed_query("abc") == [3.0, 1.0]


def test_openai_embed_texts_empty_returns_empty_without_client() -> None:
    embedder = OpenAIEmbedder()

    assert embedder.embed_texts([]) == []
    assert embedder._client is None


def test_openai_embed_query_empty_raises_without_client() -> None:
    embedder = OpenAIEmbedder()

    with pytest.raises(ValueError, match="query must not be empty"):
        embedder.embed_query("   ")
    assert embedder._client is None


def test_openai_embed_query_uses_embed_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = OpenAIEmbedder()

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        assert texts == ["search text"]
        return [[0.1, 0.2]]

    monkeypatch.setattr(embedder, "embed_texts", fake_embed_texts)

    assert embedder.embed_query("search text") == [0.1, 0.2]
