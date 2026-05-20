"""Convert page-level paper text into retrievable chunks."""

from __future__ import annotations

from typing import Any

from papermate.schemas import DocumentChunk, PageText


class TextChunker:
    """Create deterministic overlapping chunks from page text."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[PageText]) -> list[DocumentChunk]:
        """Convert pages into overlapping DocumentChunk objects."""

        page_spans, combined_text = self._combine_pages(pages)
        if not combined_text:
            return []

        chunks: list[DocumentChunk] = []
        step = self.chunk_size - self.chunk_overlap
        doc_id = page_spans[0]["page"].doc_id
        source = self._source_for_page(page_spans[0]["page"])
        source_metadata = self._source_metadata(page_spans[0]["page"])

        start = 0
        while start < len(combined_text):
            end = min(start + self.chunk_size, len(combined_text))
            chunk_text = combined_text[start:end]
            leading_space_count = len(chunk_text) - len(chunk_text.lstrip())
            trailing_space_count = len(chunk_text) - len(chunk_text.rstrip())
            char_start = start + leading_space_count
            char_end = end - trailing_space_count
            stripped_text = combined_text[char_start:char_end]

            if stripped_text:
                page_start, page_end = self._page_range(page_spans, char_start, char_end)
                chunk_index = len(chunks)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}:chunk:{chunk_index}",
                        doc_id=doc_id,
                        text=stripped_text,
                        page_start=page_start,
                        page_end=page_end,
                        source=source,
                        metadata={
                            **source_metadata,
                            "chunk_index": chunk_index,
                            "char_start": char_start,
                            "char_end": char_end,
                        },
                    )
                )

            if end == len(combined_text):
                break
            start += step

        return chunks

    def _combine_pages(self, pages: list[PageText]) -> tuple[list[dict[str, Any]], str]:
        page_spans: list[dict[str, Any]] = []
        parts: list[str] = []
        cursor = 0

        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            if parts:
                parts.append("\n\n")
                cursor += 2

            start = cursor
            parts.append(text)
            cursor += len(text)
            page_spans.append({"page": page, "start": start, "end": cursor})

        return page_spans, "".join(parts)

    def _page_range(
        self,
        page_spans: list[dict[str, Any]],
        char_start: int,
        char_end: int,
    ) -> tuple[int, int]:
        covered_pages = [
            span["page"].page_number
            for span in page_spans
            if span["start"] < char_end and span["end"] > char_start
        ]
        return min(covered_pages), max(covered_pages)

    def _source_for_page(self, page: PageText) -> str:
        file_name = page.metadata.get("file_name")
        if file_name:
            return str(file_name)

        source = page.metadata.get("source")
        if source:
            return str(source)

        return "unknown"

    def _source_metadata(self, page: PageText) -> dict[str, Any]:
        return {
            key: value
            for key, value in page.metadata.items()
            if key in {"file_name", "file_path", "page_count", "source"}
        }
