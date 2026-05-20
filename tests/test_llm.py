import pytest

from papermate.llm import BaseLLM, OpenAILLM


class FakeLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        return f"generated: {prompt}"


def test_base_llm_can_be_subclassed() -> None:
    llm = FakeLLM()

    assert llm.generate("prompt") == "generated: prompt"


def test_openai_llm_generate_empty_raises_value_error() -> None:
    llm = OpenAILLM()

    with pytest.raises(ValueError, match="prompt must not be empty"):
        llm.generate("")
    assert llm._client is None


def test_openai_llm_generate_whitespace_raises_value_error() -> None:
    llm = OpenAILLM()

    with pytest.raises(ValueError, match="prompt must not be empty"):
        llm.generate("   ")
    assert llm._client is None


def test_openai_llm_client_is_created_lazily() -> None:
    llm = OpenAILLM(api_key="test-key")

    assert llm._client is None
