# PaperMate RAG

PaperMate RAG is a modular RAG system for asking citation-grounded questions over academic papers.

It is designed as a lightweight MVP for academic paper reading: upload a PDF, index it, ask questions, and inspect citations tied back to retrieved paper chunks.

## Key Features

- PDF upload and text extraction
- Text chunking
- Embedding generation
- Chroma vector search
- Citation-grounded QA
- Streamlit chat UI
- Provider support: Gemini and OpenAI
- Modular architecture and tests

## Tech Stack

- Python
- Streamlit
- ChromaDB
- Gemini API
- OpenAI API
- pytest

## Quick Start

Activate the existing conda environment:

```powershell
conda activate papermate
```

Install dependencies if needed:

```powershell
pip install -e .
```

Create a local `.env` file:

```powershell
Copy-Item .env.example .env
```

Add provider credentials to `.env`:

```text
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Run the Streamlit app:

```powershell
streamlit run src/papermate/ui/streamlit_app.py
```

## Demo Workflow

Gemini is the recommended provider for interviews and free-tier demos. OpenAI is also supported when `OPENAI_API_KEY` is configured.

1. Start the app.
2. Choose the Gemini provider in Advanced settings.
3. Upload a PDF from the sidebar.
4. Click `Index paper`.
5. Ask questions in the chat.
6. Check citations under the answer.

## Run Tests

```powershell
python -m pytest
```

## Notes

- Do not commit `.env`.
- Uploaded PDFs and Chroma data are local runtime artifacts.
- Gemini free tier may be rate-limited.
- OpenAI API billing is separate from a ChatGPT subscription.
