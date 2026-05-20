from papermate.schemas import (
    Citation,
    DocumentChunk,
    PageText,
    PaperDocument,
    PaperSummary,
    RAGAnswer,
    RetrievedChunk,
)


def test_paper_document_schema_creation() -> None:
    page = PageText(doc_id="paper-1", page_number=1, text="Introduction text.")
    document = PaperDocument(
        doc_id="paper-1",
        title="A Useful Paper",
        file_name="paper.pdf",
        file_path="data/papers/paper.pdf",
    )

    assert page.doc_id == document.doc_id
    assert document.file_name == "paper.pdf"
    assert document.metadata == {}


def test_chunk_and_retrieved_chunk_schema_creation() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="paper-1",
        text="Chunk text.",
        page_start=1,
        page_end=2,
        source="paper.pdf",
    )
    retrieved = RetrievedChunk(chunk=DocumentChunk(**chunk.model_dump()), score=0.91)

    assert retrieved.chunk.chunk_id == "chunk-1"
    assert retrieved.chunk.doc_id == "paper-1"
    assert retrieved.score == 0.91


def test_answer_and_summary_schema_creation() -> None:
    citation = Citation(
        doc_id="paper-1",
        chunk_id="chunk-1",
        source="paper.pdf",
        page_start=3,
        page_end=4,
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="paper-1",
        text="Evidence text.",
        page_start=3,
        page_end=4,
        source="paper.pdf",
    )
    answer = RAGAnswer(
        question="What did the paper find?",
        answer="It found a useful result.",
        citations=[citation],
        retrieved_chunks=[RetrievedChunk(chunk=chunk, score=0.87)],
    )
    summary = PaperSummary(
        doc_id="paper-1",
        title="A Useful Paper",
        research_problem="A research problem.",
        method="A method.",
        dataset="A dataset.",
        experiment="An experiment.",
        results="Useful results.",
        limitations="Some limitations.",
        relevance_to_user_research="Relevant to the user's research.",
        citations=[citation],
    )

    assert answer.citations[0].page_start == 3
    assert answer.retrieved_chunks[0].chunk.chunk_id == "chunk-1"
    assert summary.doc_id == "paper-1"
    assert summary.results == "Useful results."
