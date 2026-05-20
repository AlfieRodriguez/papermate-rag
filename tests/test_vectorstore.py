import pytest

from papermate.schemas import DocumentChunk, RetrievedChunk
from papermate.vectorstores import ChromaVectorStore


def chunk(
    chunk_id: str,
    text: str,
    page_start: int = 1,
    page_end: int = 1,
    metadata: dict[str, int] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="paper-1",
        text=text,
        page_start=page_start,
        page_end=page_end,
        source="paper.pdf",
        metadata=metadata or {"chunk_index": 0, "char_start": 0, "char_end": len(text)},
    )


def store(tmp_path) -> ChromaVectorStore:
    return ChromaVectorStore(persist_directory=str(tmp_path / "chroma"))


def test_add_chunks_and_similarity_search(tmp_path) -> None:
    vectorstore = store(tmp_path)
    chunks = [
        chunk("paper-1:chunk:0", "alpha text"),
        chunk("paper-1:chunk:1", "beta text"),
    ]
    vectorstore.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = vectorstore.similarity_search([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert isinstance(results[0].chunk, DocumentChunk)
    assert results[0].chunk.chunk_id == "paper-1:chunk:0"
    assert results[0].chunk.text == "alpha text"


def test_empty_chunks_do_nothing(tmp_path) -> None:
    vectorstore = store(tmp_path)

    vectorstore.add_chunks([], [])

    assert vectorstore.similarity_search([1.0, 0.0]) == []


def test_mismatched_chunks_and_embeddings_raises_value_error(tmp_path) -> None:
    vectorstore = store(tmp_path)

    with pytest.raises(ValueError, match="same length"):
        vectorstore.add_chunks([chunk("paper-1:chunk:0", "alpha text")], [])


def test_invalid_top_k_raises_value_error(tmp_path) -> None:
    vectorstore = store(tmp_path)

    with pytest.raises(ValueError, match="top_k must be > 0"):
        vectorstore.similarity_search([1.0, 0.0], top_k=0)


def test_empty_query_embedding_returns_empty(tmp_path) -> None:
    vectorstore = store(tmp_path)
    vectorstore.add_chunks([chunk("paper-1:chunk:0", "alpha text")], [[1.0, 0.0]])

    assert vectorstore.similarity_search([], top_k=1) == []


def test_metadata_page_and_source_are_preserved(tmp_path) -> None:
    vectorstore = store(tmp_path)
    original = chunk(
        "paper-1:chunk:0",
        "spans pages",
        page_start=2,
        page_end=3,
        metadata={"chunk_index": 4, "char_start": 10, "char_end": 21},
    )
    vectorstore.add_chunks([original], [[1.0, 0.0]])

    result = vectorstore.similarity_search([1.0, 0.0], top_k=1)[0].chunk

    assert result.doc_id == "paper-1"
    assert result.source == "paper.pdf"
    assert result.page_start == 2
    assert result.page_end == 3
    assert result.metadata == {"chunk_index": 4, "char_start": 10, "char_end": 21}
