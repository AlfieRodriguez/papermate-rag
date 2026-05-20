"""Gemini LLM provider."""

from __future__ import annotations

from typing import Any

from papermate.llm.base import BaseLLM


class GeminiLLM(BaseLLM):
    """Generate text with the Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-flash",
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._client: Any | None = None

    def generate(self, prompt: str) -> str:
        """Generate text for a non-empty prompt."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = self._get_client().models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": self.temperature},
        )
        text = getattr(response, "text", None)
        if text is None:
            raise RuntimeError("Gemini response did not include text")
        return str(text)

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            if self.api_key is None:
                self._client = genai.Client()
            else:
                self._client = genai.Client(api_key=self.api_key)
        return self._client
