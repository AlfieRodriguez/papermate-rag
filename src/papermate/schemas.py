"""Core data schemas for PaperMate RAG."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PageText(BaseModel):
    """Text extracted from a single page in a paper."""

    doc_id: str
    page_number: int = Field(..., ge=1)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperDocument(BaseModel):
    """Metadata for a paper available to the RAG system."""

    doc_id: str
    title: str | None = None
    file_name: str
    file_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A retrievable chunk of paper text."""

    chunk_id: str
    doc_id: str
    text: str
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A document chunk returned by retrieval."""

    chunk: DocumentChunk
    score: float


class Citation(BaseModel):
    """Citation metadata for text used in an answer."""

    doc_id: str
    source: str
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    chunk_id: str


class RAGAnswer(BaseModel):
    """Answer returned from a RAG pipeline."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)


class PaperSummary(BaseModel):
    """A generated summary for a paper."""

    doc_id: str
    title: str
    research_problem: str
    method: str
    dataset: str
    experiment: str
    results: str
    limitations: str
    relevance_to_user_research: str
    citations: list[Citation] = Field(default_factory=list)
