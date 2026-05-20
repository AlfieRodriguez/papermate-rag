"""Embedding providers for PaperMate RAG."""

from papermate.embeddings.base import BaseEmbedder
from papermate.embeddings.gemini_embedder import GeminiEmbedder
from papermate.embeddings.openai_embedder import OpenAIEmbedder

__all__ = ["BaseEmbedder", "GeminiEmbedder", "OpenAIEmbedder"]
