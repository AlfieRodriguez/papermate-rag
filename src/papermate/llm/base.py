"""LLM provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Interface for text generation providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
