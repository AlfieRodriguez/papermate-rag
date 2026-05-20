"""Application-level coordination for papers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from papermate.schemas import DocumentChunk, PageText, PaperDocument, RAGAnswer


class PaperService:
    """Coordinate paper ingestion and question answering."""

    def __init__(
        self,
        pdf_loader: Any,
        chunker: Any,
        embedder: Any,
        vector_store: Any,
        qa_chain: Any,
    ) -> None:
        self.pdf_loader = pdf_loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.qa_chain = qa_chain
        self.documents: dict[str, PaperDocument] = {}

    def ingest_pdf(self, file_path: str, doc_id: str | None = None) -> str:
        """Load, chunk, embed, and store one PDF."""

        if not file_path.strip():
            raise ValueError("file_path must not be empty")
        if doc_id is not None and not doc_id.strip():
            raise ValueError("doc_id must not be empty")

        path = Path(file_path)
        effective_doc_id = doc_id.strip() if doc_id is not None else self._generate_doc_id(path)

        pages: list[PageText] = self.pdf_loader.load(file_path, effective_doc_id)
        chunks: list[DocumentChunk] = self.chunker.chunk_pages(pages)
        if chunks:
            embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
            self.vector_store.add_chunks(chunks, embeddings)

        self.documents[effective_doc_id] = PaperDocument(
            doc_id=effective_doc_id,
            title=path.stem,
            file_name=path.name,
            file_path=file_path,
            metadata={"page_count": len(pages), "chunk_count": len(chunks)},
        )
        return effective_doc_id

    def ask(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """Answer a question using the configured QA chain."""

        if not question.strip():
            raise ValueError("question must not be empty")

        return self.qa_chain.answer(question, top_k=top_k)

    def list_documents(self) -> list[PaperDocument]:
        """Return documents ingested during this service lifetime."""

        return list(self.documents.values())

    def get_document(self, doc_id: str) -> PaperDocument | None:
        """Return one ingested document by id."""

        return self.documents.get(doc_id)

    def _generate_doc_id(self, path: Path) -> str:
        file_stem = path.stem or "paper"
        short_hash = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
        return f"paper:{file_stem}:{short_hash}"
