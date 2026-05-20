"""OpenAI LLM provider."""

from __future__ import annotations

from openai import OpenAI

from papermate.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """Generate text with the OpenAI Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._client: OpenAI | None = None

    def generate(self, prompt: str) -> str:
        """Generate text for a non-empty prompt."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = self._get_client().chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content or ""

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if self.api_key is None:
                self._client = OpenAI()
            else:
                self._client = OpenAI(api_key=self.api_key)
        return self._client
