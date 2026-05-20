import pytest

from papermate.schemas import DocumentChunk, PageText, RAGAnswer
from papermate.services import PaperService


class FakePDFLoader:
    def __init__(self) -> None:
        self.file_path: str | None = None
        self.doc_id: str | None = None
        self.pages: list[PageText] = []

    def load(self, file_path: str, doc_id: str) -> list[PageText]:
        self.file_path = file_path
        self.doc_id = doc_id
        return self.pages


class FakeChunker:
    def __init__(self) -> None:
        self.pages: list[PageText] | None = None
        self.chunks: list[DocumentChunk] = []

    def chunk_pages(self, pages: list[PageText]) -> list[DocumentChunk]:
        self.pages = pages
        return self.chunks


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] | None = None
        self.embeddings = [[1.0, 0.0]]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return self.embeddings


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] | None = None
        self.embeddings: list[list[float]] | None = None

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings


class FakeQAChain:
    def __init__(self) -> None:
        self.question: str | None = None
        self.top_k: int | None = None
        self.answer_result = RAGAnswer(question="question", answer="answer")

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        self.question = question
        self.top_k = top_k
        return self.answer_result


def page(doc_id: str = "paper-1") -> PageText:
    return PageText(
        doc_id=doc_id,
        page_number=1,
        text="Page text.",
        metadata={"file_name": "paper.pdf"},
    )


def chunk(doc_id: str = "paper-1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_id}:chunk:0",
        doc_id=doc_id,
        text="Chunk text.",
        page_start=1,
        page_end=1,
        source="paper.pdf",
        metadata={"chunk_index": 0},
    )


def make_service() -> tuple[
    PaperService,
    FakePDFLoader,
    FakeChunker,
    FakeEmbedder,
    FakeVectorStore,
    FakeQAChain,
]:
    pdf_loader = FakePDFLoader()
    chunker = FakeChunker()
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    qa_chain = FakeQAChain()
    return (
        PaperService(pdf_loader, chunker, embedder, vector_store, qa_chain),
        pdf_loader,
        chunker,
        embedder,
        vector_store,
        qa_chain,
    )


def test_ingest_pdf_calls_pdf_loader_with_generated_doc_id() -> None:
    service, pdf_loader, chunker, _, _, _ = make_service()
    pdf_loader.pages = [page()]
    chunker.chunks = [chunk()]

    doc_id = service.ingest_pdf("papers/example.pdf")

    assert pdf_loader.file_path == "papers/example.pdf"
    assert pdf_loader.doc_id == doc_id
    assert doc_id.startswith("paper:example:")


def test_ingest_pdf_calls_chunker_with_pages() -> None:
    service, pdf_loader, chunker, _, _, _ = make_service()
    pdf_loader.pages = [page()]

    service.ingest_pdf("papers/example.pdf")

    assert chunker.pages == pdf_loader.pages


def test_ingest_pdf_calls_embedder_with_chunk_texts() -> None:
    service, pdf_loader, chunker, embedder, _, _ = make_service()
    pdf_loader.pages = [page()]
    chunker.chunks = [chunk()]

    service.ingest_pdf("papers/example.pdf")

    assert embedder.texts == ["Chunk text."]


def test_ingest_pdf_calls_vector_store_with_chunks_and_embeddings() -> None:
    service, pdf_loader, chunker, embedder, vector_store, _ = make_service()
    pdf_loader.pages = [page()]
    chunker.chunks = [chunk()]
    embedder.embeddings = [[0.1, 0.2]]

    service.ingest_pdf("papers/example.pdf")

    assert vector_store.chunks == chunker.chunks
    assert vector_store.embeddings == [[0.1, 0.2]]


def test_ingest_pdf_returns_doc_id() -> None:
    service, pdf_loader, _, _, _, _ = make_service()
    pdf_loader.pages = [page()]

    doc_id = service.ingest_pdf("papers/example.pdf")

    assert doc_id.startswith("paper:example:")


def test_provided_doc_id_is_used() -> None:
    service, pdf_loader, chunker, _, _, _ = make_service()
    pdf_loader.pages = [page("custom-doc")]
    chunker.chunks = [chunk("custom-doc")]

    doc_id = service.ingest_pdf("papers/example.pdf", doc_id="custom-doc")

    assert doc_id == "custom-doc"
    assert pdf_loader.doc_id == "custom-doc"


def test_empty_file_path_raises_value_error() -> None:
    service, _, _, _, _, _ = make_service()

    with pytest.raises(ValueError, match="file_path must not be empty"):
        service.ingest_pdf("   ")


def test_empty_provided_doc_id_raises_value_error() -> None:
    service, _, _, _, _, _ = make_service()

    with pytest.raises(ValueError, match="doc_id must not be empty"):
        service.ingest_pdf("papers/example.pdf", doc_id="   ")


def test_no_chunks_skips_embedder_and_vector_store() -> None:
    service, pdf_loader, chunker, embedder, vector_store, _ = make_service()
    pdf_loader.pages = [page()]
    chunker.chunks = []

    doc_id = service.ingest_pdf("papers/example.pdf")

    assert doc_id.startswith("paper:example:")
    assert embedder.texts is None
    assert vector_store.chunks is None
    assert vector_store.embeddings is None


def test_ask_delegates_to_qa_chain_answer() -> None:
    service, _, _, _, _, qa_chain = make_service()

    result = service.ask("What is the method?")

    assert result == qa_chain.answer_result
    assert qa_chain.question == "What is the method?"


def test_ask_passes_method_level_top_k() -> None:
    service, _, _, _, _, qa_chain = make_service()

    service.ask("What is the method?", top_k=3)

    assert qa_chain.top_k == 3


def test_empty_question_raises_value_error() -> None:
    service, _, _, _, _, _ = make_service()

    with pytest.raises(ValueError, match="question must not be empty"):
        service.ask("   ")


def test_list_documents_returns_ingested_documents() -> None:
    service, pdf_loader, _, _, _, _ = make_service()
    pdf_loader.pages = [page()]

    doc_id = service.ingest_pdf("papers/example.pdf")
    documents = service.list_documents()

    assert len(documents) == 1
    assert documents[0].doc_id == doc_id
    assert documents[0].title == "example"
    assert documents[0].file_name == "example.pdf"


def test_get_document_returns_matching_document_or_none() -> None:
    service, pdf_loader, _, _, _, _ = make_service()
    pdf_loader.pages = [page()]

    doc_id = service.ingest_pdf("papers/example.pdf")

    assert service.get_document(doc_id) is not None
    assert service.get_document("missing") is None
