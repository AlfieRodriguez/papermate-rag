"""Streamlit app for the PaperMate RAG MVP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from papermate.chains import QAChain
from papermate.chunking import TextChunker
from papermate.embeddings import GeminiEmbedder, OpenAIEmbedder
from papermate.ingestion.pdf_loader import PDFLoader
from papermate.llm import GeminiLLM, OpenAILLM
from papermate.retrieval import Retriever
from papermate.schemas import Citation, RetrievedChunk
from papermate.services import PaperService
from papermate.vectorstores import ChromaVectorStore

load_dotenv()

UPLOAD_DIR = Path("data/raw/papers")
CHROMA_DIR = Path("data/chroma")
DEFAULT_TOP_K = 5
PROVIDER_GEMINI = "gemini"
PROVIDER_OPENAI = "openai"
DEFAULT_OPENAI_LLM_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_GEMINI_LLM_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
SAMPLE_QUESTIONS = [
    "What is the main contribution of this paper?",
    "What dataset does the paper use?",
    "How does the proposed method work?",
    "What are the limitations of this study?",
]


def build_service(
    top_k: int = DEFAULT_TOP_K,
    provider: str = PROVIDER_GEMINI,
    llm_model: str | None = None,
    embedding_model: str | None = None,
) -> PaperService:
    """Build the PaperMate service graph for the Streamlit app."""

    normalized_provider = normalize_provider(provider)
    if normalized_provider == PROVIDER_GEMINI:
        effective_llm_model = llm_model or default_gemini_model()
        effective_embedding_model = embedding_model or default_gemini_embedding_model()
        embedder = GeminiEmbedder(model=effective_embedding_model)
        llm = GeminiLLM(model=effective_llm_model)
        persist_directory = CHROMA_DIR / PROVIDER_GEMINI
    elif normalized_provider == PROVIDER_OPENAI:
        effective_llm_model = llm_model or DEFAULT_OPENAI_LLM_MODEL
        effective_embedding_model = embedding_model or DEFAULT_OPENAI_EMBEDDING_MODEL
        embedder = OpenAIEmbedder(model=effective_embedding_model)
        llm = OpenAILLM(model=effective_llm_model)
        persist_directory = CHROMA_DIR / PROVIDER_OPENAI
    else:
        raise ValueError(f"unsupported provider: {provider}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    persist_directory.mkdir(parents=True, exist_ok=True)
    pdf_loader = PDFLoader()
    chunker = TextChunker()
    vector_store = ChromaVectorStore(persist_directory=str(persist_directory))
    retriever = Retriever(embedder=embedder, vector_store=vector_store, top_k=top_k)
    qa_chain = QAChain(retriever=retriever, llm=llm, top_k=top_k)
    return PaperService(
        pdf_loader=pdf_loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        qa_chain=qa_chain,
    )


def normalize_provider(provider: str) -> str:
    """Normalize a provider label from UI or config."""

    return provider.strip().lower()


def default_provider() -> str:
    """Choose the default provider from available API keys."""

    if os.getenv("GEMINI_API_KEY"):
        return PROVIDER_GEMINI
    if os.getenv("OPENAI_API_KEY"):
        return PROVIDER_OPENAI
    return PROVIDER_GEMINI


def provider_display_name(provider: str) -> str:
    """Return the display name for a provider."""

    normalized_provider = normalize_provider(provider)
    if normalized_provider == PROVIDER_GEMINI:
        return "Gemini"
    if normalized_provider == PROVIDER_OPENAI:
        return "OpenAI"
    return provider


def default_gemini_model() -> str:
    """Return Gemini LLM model default from the environment."""

    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_LLM_MODEL)


def default_gemini_embedding_model() -> str:
    """Return Gemini embedding model default from the environment."""

    return os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL)


def build_doc_upload_path(file_name: str, upload_dir: Path = UPLOAD_DIR) -> Path:
    """Return a safe upload path for a PDF file name."""

    return upload_dir / Path(file_name).name


def format_page_range(page_start: int, page_end: int) -> str:
    """Format a citation page range."""

    if page_start == page_end:
        return f"p. {page_start}"
    return f"pp. {page_start}-{page_end}"


def citation_label(citation: Citation) -> str:
    """Format a compact citation label."""

    return f"{citation.source} - {format_page_range(citation.page_start, citation.page_end)}"


def find_retrieved_chunk(
    citation: Citation,
    retrieved_chunks: list[RetrievedChunk] | None,
) -> RetrievedChunk | None:
    """Find the retrieved chunk matching a citation."""

    if not retrieved_chunks:
        return None
    for retrieved in retrieved_chunks:
        if retrieved.chunk.chunk_id == citation.chunk_id:
            return retrieved
    return None


def chunk_preview(text: str, max_length: int = 500) -> str:
    """Return a compact preview of retrieved text."""

    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def apply_styles() -> None:
    """Apply light styling for the Streamlit app."""

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 980px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .pm-subtitle {
            color: #64748b;
            font-size: 1.05rem;
            margin-top: -0.75rem;
            margin-bottom: 1.25rem;
        }
        .pm-status-row {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: 1rem 0 1.5rem;
        }
        .pm-status-card {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            background: #ffffff;
        }
        .pm-status-label {
            color: #64748b;
            font-size: 0.78rem;
            margin-bottom: 0.2rem;
        }
        .pm-status-value {
            color: #0f172a;
            font-weight: 650;
            font-size: 1rem;
        }
        .pm-muted {
            color: #64748b;
        }
        .pm-empty-state {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.25rem;
            background: #ffffff;
            margin: 1rem 0 1.5rem;
        }
        .pm-empty-title {
            color: #0f172a;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .pm-empty-subtitle {
            color: #64748b;
            margin-bottom: 1rem;
        }
        .pm-workflow {
            display: grid;
            gap: 0.5rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin-bottom: 1rem;
        }
        .pm-workflow-step,
        .pm-question-card {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.7rem 0.75rem;
            background: #f8fafc;
            color: #334155;
            font-size: 0.9rem;
        }
        .pm-question-grid {
            display: grid;
            gap: 0.5rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Initialize Streamlit session state."""

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("indexed_doc_ids", [])
    st.session_state.setdefault("service", None)
    st.session_state.setdefault("service_config", None)


def get_service(
    top_k: int,
    provider: str,
    llm_model: str,
    embedding_model: str,
) -> PaperService:
    """Return a cached service, rebuilding when settings change."""

    config = {
        "top_k": top_k,
        "provider": normalize_provider(provider),
        "llm_model": llm_model,
        "embedding_model": embedding_model,
    }
    if st.session_state.service is None or st.session_state.service_config != config:
        st.session_state.service = build_service(
            top_k=top_k,
            provider=provider,
            llm_model=llm_model,
            embedding_model=embedding_model,
        )
        st.session_state.service_config = config
    return st.session_state.service


def save_uploaded_pdf(uploaded_file: Any) -> Path:
    """Save an uploaded PDF and return its path."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = build_doc_upload_path(uploaded_file.name)
    upload_path.write_bytes(uploaded_file.getbuffer())
    return upload_path


def render_citations(
    citations: list[Citation],
    retrieved_chunks: list[RetrievedChunk] | None = None,
) -> None:
    """Render citations under an assistant answer."""

    if not citations:
        st.caption("No citations available.")
        return

    st.caption("Citations")
    for index, citation in enumerate(citations, start=1):
        with st.expander(f"[{index}] {citation_label(citation)}"):
            st.markdown(f"**Chunk:** `{citation.chunk_id}`")
            retrieved = find_retrieved_chunk(citation, retrieved_chunks)
            if retrieved is not None:
                st.markdown(chunk_preview(retrieved.chunk.text))


def render_status_row(indexed_count: int, top_k: int, provider: str) -> None:
    """Render compact app status cards."""

    st.markdown(
        f"""
        <div class="pm-status-row">
          <div class="pm-status-card">
            <div class="pm-status-label">Indexed papers</div>
            <div class="pm-status-value">{indexed_count}</div>
          </div>
          <div class="pm-status-card">
            <div class="pm-status-label">Provider</div>
            <div class="pm-status-value">{provider_display_name(provider)}</div>
          </div>
          <div class="pm-status-card">
            <div class="pm-status-label">Top K</div>
            <div class="pm-status-value">{top_k}</div>
          </div>
          <div class="pm-status-card">
            <div class="pm-status-label">Vector store</div>
            <div class="pm-status-value">Chroma</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_messages() -> None:
    """Render chat history."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_citations(
                    message.get("citations", []),
                    message.get("retrieved_chunks", []),
                )


def render_empty_state() -> None:
    """Render the first-screen empty state."""

    question_cards = "\n".join(
        f'<div class="pm-question-card">{question}</div>'
        for question in SAMPLE_QUESTIONS
    )
    st.markdown(
        f"""
        <div class="pm-empty-state">
          <div class="pm-empty-title">Start with a paper</div>
          <div class="pm-empty-subtitle">
            Upload a PDF, index it, then ask questions with citation-backed answers.
          </div>
          <div class="pm-workflow">
            <div class="pm-workflow-step">Upload a PDF from the sidebar.</div>
            <div class="pm-workflow-step">Click Index paper.</div>
            <div class="pm-workflow-step">Ask grounded questions.</div>
            <div class="pm-workflow-step">Check citations under answers.</div>
          </div>
          <div class="pm-question-grid">
            {question_cards}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(page_title="PaperMate", layout="centered")
    apply_styles()
    init_session_state()

    with st.sidebar:
        st.markdown("## PaperMate RAG")
        st.caption("Ask questions grounded in academic papers.")

        st.markdown("### PDF upload")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        index_clicked = st.button("Index paper", use_container_width=True)

        st.markdown("### Indexed papers")
        if st.session_state.indexed_doc_ids:
            for doc_id in st.session_state.indexed_doc_ids:
                st.caption(doc_id)
        else:
            st.caption("No papers indexed yet.")

        with st.expander("Advanced settings"):
            provider_options = ["Gemini", "OpenAI"]
            default_provider_name = provider_display_name(default_provider())
            provider = st.selectbox(
                "Provider",
                provider_options,
                index=provider_options.index(default_provider_name),
            )
            normalized_provider = normalize_provider(provider)
            if normalized_provider == PROVIDER_GEMINI:
                llm_default = default_gemini_model()
                embedding_default = default_gemini_embedding_model()
                llm_label = "Gemini model"
                embedding_label = "Gemini embedding model"
                if not os.getenv("GEMINI_API_KEY"):
                    st.warning("Set GEMINI_API_KEY before real Gemini answering.")
            else:
                llm_default = DEFAULT_OPENAI_LLM_MODEL
                embedding_default = DEFAULT_OPENAI_EMBEDDING_MODEL
                llm_label = "OpenAI model"
                embedding_label = "OpenAI embedding model"
                if not os.getenv("OPENAI_API_KEY"):
                    st.warning("Set OPENAI_API_KEY before real OpenAI answering.")

            top_k = st.slider("top_k", min_value=1, max_value=10, value=DEFAULT_TOP_K)
            llm_model = st.text_input(llm_label, value=llm_default)
            embedding_model = st.text_input(
                embedding_label,
                value=embedding_default,
            )
            st.caption("Settings changes rebuild the service automatically.")

    try:
        service = get_service(top_k, provider, llm_model, embedding_model)
    except Exception as error:
        st.error(f"Could not initialize PaperMate: {error}")
        return

    if index_clicked:
        if uploaded_file is None:
            st.sidebar.error("Upload a PDF before indexing.")
        else:
            try:
                saved_path = save_uploaded_pdf(uploaded_file)
                doc_id = service.ingest_pdf(str(saved_path))
                st.session_state.indexed_doc_ids.append(doc_id)
                st.sidebar.success(f"Indexed paper: {doc_id}")
            except Exception as error:
                st.sidebar.error(f"Could not index paper: {error}")

    st.title("PaperMate")
    st.markdown(
        '<div class="pm-subtitle">Ask questions, get answers backed by paper evidence.</div>',
        unsafe_allow_html=True,
    )
    render_status_row(len(st.session_state.indexed_doc_ids), top_k, provider)

    with st.container():
        if st.session_state.messages:
            render_messages()
        else:
            render_empty_state()

    question = st.chat_input("Ask a question about your indexed papers")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                rag_answer = service.ask(question, top_k=top_k)
                st.markdown(rag_answer.answer)
                render_citations(rag_answer.citations, rag_answer.retrieved_chunks)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": rag_answer.answer,
                        "citations": rag_answer.citations,
                        "retrieved_chunks": rag_answer.retrieved_chunks,
                    }
                )
            except Exception as error:
                st.error(f"Could not answer question: {error}")


if __name__ == "__main__":
    main()
