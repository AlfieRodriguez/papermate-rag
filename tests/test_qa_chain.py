import pytest

from papermate.chains.qa_chain import NO_EVIDENCE_ANSWER, QAChain
from papermate.llm.base import BaseLLM
from papermate.schemas import DocumentChunk, RAGAnswer, RetrievedChunk


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.results = results or []
        self.question: str | None = None
        self.top_k: int | None = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.question = query
        self.top_k = top_k
        return self.results


class FakeLLM(BaseLLM):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Generated answer."


def retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="paper-1:chunk:0",
            doc_id="paper-1",
            text="The paper evaluates a retrieval augmented reading workflow.",
            page_start=2,
            page_end=3,
            source="paper.pdf",
            metadata={"chunk_index": 0},
        ),
        score=0.2,
    )


def test_answer_empty_question_raises_value_error() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        QAChain(FakeRetriever(), FakeLLM()).answer("   ")


def test_constructor_invalid_top_k_raises_value_error() -> None:
    with pytest.raises(ValueError, match="top_k must be > 0"):
        QAChain(FakeRetriever(), FakeLLM(), top_k=0)


def test_method_invalid_top_k_raises_value_error() -> None:
    with pytest.raises(ValueError, match="top_k must be > 0"):
        QAChain(FakeRetriever(), FakeLLM()).answer("What is tested?", top_k=0)


def test_answer_calls_retriever_with_question_and_effective_top_k() -> None:
    retriever = FakeRetriever([retrieved_chunk()])

    QAChain(retriever, FakeLLM(), top_k=4).answer("What is tested?")

    assert retriever.question == "What is tested?"
    assert retriever.top_k == 4


def test_answer_calls_llm_generate_when_chunks_exist() -> None:
    llm = FakeLLM()

    QAChain(FakeRetriever([retrieved_chunk()]), llm).answer("What is tested?")

    assert len(llm.prompts) == 1


def test_answer_returns_rag_answer() -> None:
    answer = QAChain(FakeRetriever([retrieved_chunk()]), FakeLLM()).answer(
        "What is tested?"
    )

    assert isinstance(answer, RAGAnswer)
    assert answer.question == "What is tested?"
    assert answer.answer == "Generated answer."
    assert len(answer.retrieved_chunks) == 1


def test_answer_includes_citations_from_retrieved_chunks() -> None:
    answer = QAChain(FakeRetriever([retrieved_chunk()]), FakeLLM()).answer(
        "What is tested?"
    )

    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.doc_id == "paper-1"
    assert citation.source == "paper.pdf"
    assert citation.page_start == 2
    assert citation.page_end == 3
    assert citation.chunk_id == "paper-1:chunk:0"


def test_no_retrieved_chunks_returns_safe_answer_and_does_not_call_llm() -> None:
    llm = FakeLLM()

    answer = QAChain(FakeRetriever([]), llm).answer("What is tested?")

    assert answer.answer == NO_EVIDENCE_ANSWER
    assert answer.citations == []
    assert answer.retrieved_chunks == []
    assert llm.prompts == []


def test_method_top_k_overrides_default_top_k() -> None:
    retriever = FakeRetriever([retrieved_chunk()])

    QAChain(retriever, FakeLLM(), top_k=4).answer("What is tested?", top_k=2)

    assert retriever.top_k == 2


def test_prompt_contains_question_and_retrieved_context() -> None:
    llm = FakeLLM()

    QAChain(FakeRetriever([retrieved_chunk()]), llm).answer("What is tested?")

    prompt = llm.prompts[0]
    assert "What is tested?" in prompt
    assert "The paper evaluates a retrieval augmented reading workflow." in prompt
    assert "Source: paper.pdf, pages 2-3" in prompt
    assert "Answer using only the provided context." in prompt
