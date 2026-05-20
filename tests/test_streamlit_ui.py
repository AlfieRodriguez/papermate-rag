from pathlib import Path

import pytest

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


def test_default_provider_prefers_gemini_when_key_exists(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert streamlit_app.default_provider() == "gemini"


def test_default_provider_uses_openai_when_only_openai_key_exists(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert streamlit_app.default_provider() == "openai"


def test_build_service_has_provider_parameter() -> None:
    assert "provider" in streamlit_app.build_service.__annotations__


def test_build_service_gemini_can_use_monkeypatched_components(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(streamlit_app, "UPLOAD_DIR", tmp_path / "papers")
    monkeypatch.setattr(streamlit_app, "CHROMA_DIR", tmp_path / "chroma")
    calls: dict[str, object] = {}

    class FakePDFLoader:
        pass

    class FakeTextChunker:
        pass

    class FakeGeminiEmbedder:
        def __init__(self, model: str) -> None:
            calls["embedder_model"] = model

    class FakeVectorStore:
        def __init__(self, persist_directory: str) -> None:
            calls["persist_directory"] = persist_directory

    class FakeRetriever:
        def __init__(self, embedder, vector_store, top_k: int) -> None:
            calls["retriever_top_k"] = top_k

    class FakeGeminiLLM:
        def __init__(self, model: str) -> None:
            calls["llm_model"] = model

    class FakeQAChain:
        def __init__(self, retriever, llm, top_k: int) -> None:
            calls["qa_top_k"] = top_k

    class FakePaperService:
        def __init__(self, **kwargs) -> None:
            calls["service_kwargs"] = kwargs

    monkeypatch.setattr(streamlit_app, "PDFLoader", FakePDFLoader)
    monkeypatch.setattr(streamlit_app, "TextChunker", FakeTextChunker)
    monkeypatch.setattr(streamlit_app, "GeminiEmbedder", FakeGeminiEmbedder)
    monkeypatch.setattr(streamlit_app, "ChromaVectorStore", FakeVectorStore)
    monkeypatch.setattr(streamlit_app, "Retriever", FakeRetriever)
    monkeypatch.setattr(streamlit_app, "GeminiLLM", FakeGeminiLLM)
    monkeypatch.setattr(streamlit_app, "QAChain", FakeQAChain)
    monkeypatch.setattr(streamlit_app, "PaperService", FakePaperService)

    service = streamlit_app.build_service(
        top_k=3,
        provider="gemini",
        llm_model="gemini-test",
        embedding_model="gemini-embed-test",
    )

    assert isinstance(service, FakePaperService)
    assert calls["llm_model"] == "gemini-test"
    assert calls["embedder_model"] == "gemini-embed-test"
    assert calls["retriever_top_k"] == 3
    assert calls["qa_top_k"] == 3
    assert str(calls["persist_directory"]).endswith("chroma\\gemini") or str(
        calls["persist_directory"]
    ).endswith("chroma/gemini")


def test_build_service_openai_can_use_monkeypatched_components(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(streamlit_app, "UPLOAD_DIR", tmp_path / "papers")
    monkeypatch.setattr(streamlit_app, "CHROMA_DIR", tmp_path / "chroma")
    calls: dict[str, object] = {}

    class FakePDFLoader:
        pass

    class FakeTextChunker:
        pass

    class FakeOpenAIEmbedder:
        def __init__(self, model: str) -> None:
            calls["embedder_model"] = model

    class FakeVectorStore:
        def __init__(self, persist_directory: str) -> None:
            calls["persist_directory"] = persist_directory

    class FakeRetriever:
        def __init__(self, embedder, vector_store, top_k: int) -> None:
            calls["retriever_top_k"] = top_k

    class FakeOpenAILLM:
        def __init__(self, model: str) -> None:
            calls["llm_model"] = model

    class FakeQAChain:
        def __init__(self, retriever, llm, top_k: int) -> None:
            calls["qa_top_k"] = top_k

    class FakePaperService:
        def __init__(self, **kwargs) -> None:
            calls["service_kwargs"] = kwargs

    monkeypatch.setattr(streamlit_app, "PDFLoader", FakePDFLoader)
    monkeypatch.setattr(streamlit_app, "TextChunker", FakeTextChunker)
    monkeypatch.setattr(streamlit_app, "OpenAIEmbedder", FakeOpenAIEmbedder)
    monkeypatch.setattr(streamlit_app, "ChromaVectorStore", FakeVectorStore)
    monkeypatch.setattr(streamlit_app, "Retriever", FakeRetriever)
    monkeypatch.setattr(streamlit_app, "OpenAILLM", FakeOpenAILLM)
    monkeypatch.setattr(streamlit_app, "QAChain", FakeQAChain)
    monkeypatch.setattr(streamlit_app, "PaperService", FakePaperService)

    service = streamlit_app.build_service(
        top_k=4,
        provider="openai",
        llm_model="openai-test",
        embedding_model="openai-embed-test",
    )

    assert isinstance(service, FakePaperService)
    assert calls["llm_model"] == "openai-test"
    assert calls["embedder_model"] == "openai-embed-test"
    assert calls["retriever_top_k"] == 4
    assert calls["qa_top_k"] == 4
    assert str(calls["persist_directory"]).endswith("chroma\\openai") or str(
        calls["persist_directory"]
    ).endswith("chroma/openai")


def test_build_service_unsupported_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported provider: anthropic"):
        streamlit_app.build_service(provider="anthropic")
