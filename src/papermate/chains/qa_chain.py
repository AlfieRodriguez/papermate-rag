"""Question-answering chain for retrieved paper chunks."""

from __future__ import annotations

from papermate.llm.base import BaseLLM
from papermate.retrieval.retriever import Retriever
from papermate.schemas import Citation, RAGAnswer, RetrievedChunk

NO_EVIDENCE_ANSWER = (
    "I do not have enough evidence in the provided papers to answer this question."
)


class QAChain:
    """Answer questions using retrieved paper chunks and an LLM."""

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """Answer a question from retrieved paper context."""

        if not question.strip():
            raise ValueError("question must not be empty")

        effective_top_k = self.top_k if top_k is None else top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be > 0")

        retrieved_chunks = self.retriever.retrieve(question, top_k=effective_top_k)
        if not retrieved_chunks:
            return RAGAnswer(
                question=question,
                answer=NO_EVIDENCE_ANSWER,
                citations=[],
                retrieved_chunks=[],
            )

        prompt = self._build_prompt(question, retrieved_chunks)
        answer_text = self.llm.generate(prompt)
        citations = [self._citation_from_retrieved(chunk) for chunk in retrieved_chunks]

        return RAGAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
        )

    def _build_prompt(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        context = "\n\n".join(
            self._format_context_chunk(index, retrieved)
            for index, retrieved in enumerate(retrieved_chunks, start=1)
        )
        return (
            "You are a research paper assistant.\n"
            "Answer using only the provided context.\n"
            "Do not invent facts.\n"
            "If the context does not contain enough evidence, say so.\n"
            "Keep the answer clear and concise.\n"
            "Mention citations in the answer if useful.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer:"
        )

    def _format_context_chunk(self, index: int, retrieved: RetrievedChunk) -> str:
        chunk = retrieved.chunk
        return (
            f"[{index}] Source: {chunk.source}, pages {chunk.page_start}-{chunk.page_end}, "
            f"chunk_id: {chunk.chunk_id}\n"
            f"{chunk.text}"
        )

    def _citation_from_retrieved(self, retrieved: RetrievedChunk) -> Citation:
        chunk = retrieved.chunk
        return Citation(
            doc_id=chunk.doc_id,
            source=chunk.source,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_id=chunk.chunk_id,
        )
