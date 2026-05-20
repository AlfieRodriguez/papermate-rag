from pathlib import Path

from papermate.schemas import Citation, DocumentChunk, RetrievedChunk
from papermate.ui import streamlit_app


def test_streamlit_app_imports() -> None:
    assert streamlit_app is not None
    assert callable(streamlit_app.build_service)


def test_build_doc_upload_path_uses_file_name_only(tmp_path: Path) -> None:
    upload_path = streamlit_app.build_doc_upload_path(
        "../paper.pdf",
        upload_dir=tmp_path,
    )

    assert upload_path == tmp_path / "paper.pdf"


def test_format_page_range_for_single_page() -> None:
    assert streamlit_app.format_page_range(3, 3) == "p. 3"


def test_format_page_range_for_multiple_pages() -> None:
    assert streamlit_app.format_page_range(3, 5) == "pp. 3-5"


def test_citation_label_includes_source_and_pages() -> None:
    citation = Citation(
        doc_id="paper-1",
        source="paper.pdf",
        page_start=2,
        page_end=4,
        chunk_id="paper-1:chunk:0",
    )

    assert streamlit_app.citation_label(citation) == "paper.pdf - pp. 2-4"


def test_find_retrieved_chunk_matches_by_chunk_id() -> None:
    citation = Citation(
        doc_id="paper-1",
        source="paper.pdf",
        page_start=1,
        page_end=1,
        chunk_id="paper-1:chunk:0",
    )
    retrieved = RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="paper-1:chunk:0",
            doc_id="paper-1",
            text="Relevant chunk text.",
            page_start=1,
            page_end=1,
            source="paper.pdf",
            metadata={},
        ),
        score=0.1,
    )

    assert streamlit_app.find_retrieved_chunk(citation, [retrieved]) == retrieved


def test_chunk_preview_compacts_long_text() -> None:
    preview = streamlit_app.chunk_preview("one   two\nthree four", max_length=10)

    assert preview == "one two..."


def test_sample_questions_are_available_for_empty_state() -> None:
    assert len(streamlit_app.SAMPLE_QUESTIONS) == 4
    assert "What is the main contribution of this paper?" in streamlit_app.SAMPLE_QUESTIONS
