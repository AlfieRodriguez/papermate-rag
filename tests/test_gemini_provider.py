import pytest

from papermate.embeddings import GeminiEmbedder
from papermate.llm import GeminiLLM


class FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class FakeEmbeddingResponse:
    def __init__(self) -> None:
        self.embeddings = [
            FakeEmbedding([1.0, 0.0]),
            FakeEmbedding([0.0, 1.0]),
        ]


class FakeGenerateResponse:
    text = "Generated Gemini answer."


class FakeModels:
    def __init__(self) -> None:
        self.embed_content_calls: list[dict] = []
        self.generate_content_calls: list[dict] = []

    def embed_content(self, **kwargs):
        self.embed_content_calls.append(kwargs)
        return FakeEmbeddingResponse()

    def generate_content(self, **kwargs):
        self.generate_content_calls.append(kwargs)
        return FakeGenerateResponse()


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()


def test_gemini_embed_texts_empty_returns_empty() -> None:
    embedder = GeminiEmbedder()

    assert embedder.embed_texts([]) == []
    assert embedder._client is None


def test_gemini_embed_query_empty_raises_value_error() -> None:
    embedder = GeminiEmbedder()

    with pytest.raises(ValueError, match="query must not be empty"):
        embedder.embed_query("   ")
    assert embedder._client is None


def test_gemini_embed_texts_empty_text_raises_value_error() -> None:
    embedder = GeminiEmbedder()

    with pytest.raises(ValueError, match="text must not be empty"):
        embedder.embed_texts(["valid", "   "])
    assert embedder._client is None


def test_gemini_embedder_client_is_created_lazily() -> None:
    embedder = GeminiEmbedder()

    assert embedder._client is None


def test_gemini_embedder_returns_vectors_from_fake_response() -> None:
    embedder = GeminiEmbedder(model="gemini-embedding-001")
    fake_client = FakeClient()
    embedder._client = fake_client

    vectors = embedder.embed_texts(["alpha", "beta"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert fake_client.models.embed_content_calls == [
        {"model": "gemini-embedding-001", "contents": ["alpha", "beta"]}
    ]


def test_gemini_llm_generate_empty_raises_value_error() -> None:
    llm = GeminiLLM()

    with pytest.raises(ValueError, match="prompt must not be empty"):
        llm.generate("")
    assert llm._client is None


def test_gemini_llm_generate_whitespace_raises_value_error() -> None:
    llm = GeminiLLM()

    with pytest.raises(ValueError, match="prompt must not be empty"):
        llm.generate("   ")
    assert llm._client is None


def test_gemini_llm_client_is_created_lazily() -> None:
    llm = GeminiLLM()

    assert llm._client is None


def test_gemini_llm_returns_text_from_fake_response() -> None:
    llm = GeminiLLM(model="gemini-3.5-flash", temperature=0.2)
    fake_client = FakeClient()
    llm._client = fake_client

    text = llm.generate("Answer this.")

    assert text == "Generated Gemini answer."
    assert fake_client.models.generate_content_calls == [
        {
            "model": "gemini-3.5-flash",
            "contents": "Answer this.",
            "config": {"temperature": 0.2},
        }
    ]


def test_gemini_llm_missing_text_raises_runtime_error() -> None:
    class MissingTextResponse:
        pass

    class MissingTextModels:
        def generate_content(self, **kwargs):
            return MissingTextResponse()

    class MissingTextClient:
        models = MissingTextModels()

    llm = GeminiLLM()
    llm._client = MissingTextClient()

    with pytest.raises(RuntimeError, match="Gemini response did not include text"):
        llm.generate("Answer this.")
