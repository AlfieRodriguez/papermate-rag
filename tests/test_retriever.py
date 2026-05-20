import pytest

from papermate.embeddings.base import BaseEmbedder
from papermate.retrieval import Retriever
from papermate.schemas import DocumentChunk, RetrievedChunk
from papermate.vectorstores.base import BaseVectorStore


class FakeEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.5, 0.25]


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.query_embedding: list[float] | None = None
        self.top_k: int | None = None
        self.results = [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="paper-1:chunk:0",
                    doc_id="paper-1",
                    text="Relevant text.",
                    page_start=1,
                    page_end=1,
                    source="paper.pdf",
                    metadata={"chunk_index": 0},
                ),
                score=0.1,
            )
        ]

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        pass

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.query_embedding = query_embedding
        self.top_k = top_k
        return self.results


def test_retrieve_calls_embed_query_with_query() -> None:
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    Retriever(embedder, vector_store).retrieve("What is the method?")

    assert embedder.queries == ["What is the method?"]


def test_retrieve_calls_similarity_search_with_embedding_and_top_k() -> None:
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    Retriever(embedder, vector_store, top_k=3).retrieve("What is the method?")

    assert vector_store.query_embedding == [0.5, 0.25]
    assert vector_store.top_k == 3


def test_retrieve_returns_retrieved_chunks() -> None:
    results = Retriever(FakeEmbedder(), FakeVectorStore()).retrieve("query")

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert isinstance(results[0].chunk, DocumentChunk)


def test_default_top_k_is_used() -> None:
    vector_store = FakeVectorStore()

    Retriever(FakeEmbedder(), vector_store, top_k=7).retrieve("query")

    assert vector_store.top_k == 7


def test_method_top_k_overrides_default_top_k() -> None:
    vector_store = FakeVectorStore()

    Retriever(FakeEmbedder(), vector_store, top_k=7).retrieve("query", top_k=2)

    assert vector_store.top_k == 2


def test_empty_query_raises_value_error() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        Retriever(FakeEmbedder(), FakeVectorStore()).retrieve("   ")


def test_invalid_constructor_top_k_raises_value_error() -> None:
    with pytest.raises(ValueError, match="top_k must be > 0"):
        Retriever(FakeEmbedder(), FakeVectorStore(), top_k=0)


def test_invalid_method_top_k_raises_value_error() -> None:
    with pytest.raises(ValueError, match="top_k must be > 0"):
        Retriever(FakeEmbedder(), FakeVectorStore()).retrieve("query", top_k=0)
