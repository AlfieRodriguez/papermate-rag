# PaperMate RAG Demo Guide

## 3-Minute Demo Script

1. Open the Streamlit app and briefly explain the goal: PaperMate lets a user ask citation-grounded questions over academic papers.
2. Point out the sidebar workflow: choose a provider, upload a PDF, and click `Index paper`.
3. Upload a paper and index it. Explain that the pipeline extracts page text, chunks it, embeds chunks, and stores them in Chroma.
4. Ask a question in the chat. Show that the answer is grounded in retrieved chunks.
5. Expand citations below the answer and point out source, page range, chunk ID, and text preview.
6. Ask an unsupported question and explain that the QA chain is instructed to say when the context lacks enough evidence.

## Suggested Questions

- What is the main contribution of this paper?
- What dataset does the paper use?
- What are the limitations?
- What information is not contained in this paper?

## What To Point Out

- The system is built as a modular pipeline rather than one large script.
- Answers include citations so the user can inspect supporting evidence.
- Provider selection supports Gemini and OpenAI without changing the UI workflow.
- Unit tests use fake dependencies and avoid real API calls.
- Unsupported questions should trigger no-hallucination behavior instead of invented facts.
