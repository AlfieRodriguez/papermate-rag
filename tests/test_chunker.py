import pytest

from papermate.chunking import TextChunker
from papermate.schemas import PageText


def page(
    text: str,
    page_number: int = 1,
    doc_id: str = "paper-1",
    metadata: dict | None = None,
) -> PageText:
    return PageText(
        doc_id=doc_id,
        page_number=page_number,
        text=text,
        metadata=metadata or {"file_name": "paper.pdf"},
    )


def test_short_single_page_text_creates_one_chunk() -> None:
    chunks = TextChunker(chunk_size=100, chunk_overlap=10).chunk_pages(
        [page("Short page text.")]
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "paper-1:chunk:0"
    assert chunks[0].doc_id == "paper-1"
    assert chunks[0].text == "Short page text."
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["char_start"] == 0
    assert chunks[0].metadata["char_end"] == len("Short page text.")


def test_empty_pages_do_not_create_empty_chunks() -> None:
    chunks = TextChunker(chunk_size=20, chunk_overlap=5).chunk_pages(
        [page("   "), page("", page_number=2)]
    )

    assert chunks == []


def test_long_text_creates_multiple_chunks() -> None:
    chunks = TextChunker(chunk_size=10, chunk_overlap=2).chunk_pages(
        [page("abcdefghijklmnopqrstuvwxyz")]
    )

    assert len(chunks) == 3
    assert [chunk.text for chunk in chunks] == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"]


def test_overlap_behavior() -> None:
    chunks = TextChunker(chunk_size=6, chunk_overlap=2).chunk_pages(
        [page("abcdefghijkl")]
    )

    assert chunks[0].text[-2:] == chunks[1].text[:2]
    assert chunks[1].metadata["char_start"] == 4


def test_multiple_pages_preserve_page_start_and_page_end() -> None:
    chunks = TextChunker(chunk_size=15, chunk_overlap=0).chunk_pages(
        [
            page("first page", page_number=1),
            page("second page", page_number=2),
        ]
    )

    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert chunks[1].page_start == 2
    assert chunks[1].page_end == 2


def test_chunk_id_is_deterministic() -> None:
    pages = [page("abcdefghijklmnopqrstuvwxyz")]
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)

    first_ids = [chunk.chunk_id for chunk in chunker.chunk_pages(pages)]
    second_ids = [chunk.chunk_id for chunk in chunker.chunk_pages(pages)]

    assert first_ids == second_ids
    assert first_ids == ["paper-1:chunk:0", "paper-1:chunk:1", "paper-1:chunk:2"]


def test_invalid_chunk_size_raises_value_error() -> None:
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        TextChunker(chunk_size=0)


@pytest.mark.parametrize("chunk_overlap", [-1, 10])
def test_invalid_chunk_overlap_raises_value_error(chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=chunk_overlap)


def test_source_is_taken_from_metadata_file_name() -> None:
    chunks = TextChunker(chunk_size=100, chunk_overlap=10).chunk_pages(
        [page("Source page.", metadata={"file_name": "source.pdf", "source": "fallback"})]
    )

    assert chunks[0].source == "source.pdf"
    assert chunks[0].metadata["file_name"] == "source.pdf"
