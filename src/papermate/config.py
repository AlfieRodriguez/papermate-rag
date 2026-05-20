"""Application configuration for PaperMate RAG."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_CHROMA_DIR = "data/chroma"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5


class AppConfig(BaseModel):
    """Runtime configuration loaded from environment variables."""

    openai_api_key: str | None = None
    llm_provider: str = DEFAULT_LLM_PROVIDER
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    chroma_dir: str = DEFAULT_CHROMA_DIR
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "AppConfig":
        """Ensure chunk overlap leaves room for new text in each chunk."""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @classmethod
    def from_env(cls, *, env_file: str | None = None) -> "AppConfig":
        """Create configuration from environment variables."""

        load_dotenv(dotenv_path=env_file)
        values: dict[str, Any] = {
            "openai_api_key": os.getenv("OPENAI_API_KEY") or None,
            "llm_provider": os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
            "embedding_provider": os.getenv(
                "EMBEDDING_PROVIDER",
                DEFAULT_EMBEDDING_PROVIDER,
            ),
            "chroma_dir": os.getenv("CHROMA_DIR", DEFAULT_CHROMA_DIR),
            "chunk_size": os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
            "chunk_overlap": os.getenv(
                "CHUNK_OVERLAP",
                DEFAULT_CHUNK_OVERLAP,
            ),
            "top_k": os.getenv("TOP_K", DEFAULT_TOP_K),
        }
        return cls(**values)


def load_config(*, env_file: str | None = None) -> AppConfig:
    """Load application configuration from the current environment."""

    return AppConfig.from_env(env_file=env_file)
