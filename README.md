# PaperMate RAG

PaperMate RAG is a lightweight Retrieval-Augmented Generation (RAG) system for reading academic papers.

It allows users to upload a PDF, index its content, ask questions about the paper, and inspect citations linked to the retrieved text chunks. The project is designed as a simple MVP for academic paper reading and question answering.

## Features

- PDF upload and text extraction
- Text chunking for long documents
- Embedding generation
- Chroma-based vector search
- Question answering with citations from retrieved chunks
- Streamlit chat interface
- Provider support for Gemini and OpenAI
- Modular project structure
- Basic tests with pytest

## Tech Stack

- Python
- Streamlit
- ChromaDB
- Gemini API
- OpenAI API
- pytest

## Project Structure

```text
papermate-rag/
├── src/
│   └── papermate/
│       ├── ui/
│       ├── ingestion/
│       ├── retrieval/
│       ├── generation/
│       └── config/
├── tests/
├── .env.example
├── README.md
└── pyproject.toml
```

## Quick Start

### 1. Create and activate a conda environment

```bash
conda create -n papermate python=3.11
conda activate papermate
```

If you already have the environment, simply activate it:

```bash
conda activate papermate
```

### 2. Install dependencies

```bash
pip install -e .
```

### 3. Create a local `.env` file

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

### 4. Add API credentials

Edit the `.env` file and add the provider credentials you want to use.

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

You only need to configure the provider you plan to use.

### 5. Run the Streamlit app

```bash
streamlit run src/papermate/ui/streamlit_app.py
```

## Demo Workflow

1. Start the Streamlit app.
2. Select a provider in the advanced settings.
3. Upload a PDF from the sidebar.
4. Click **Index paper**.
5. Ask questions in the chat.
6. Check the cited chunks under the generated answer.

For simple demos, Gemini is a convenient default option. OpenAI is also supported when `OPENAI_API_KEY` is configured.

## Run Tests

```bash
python -m pytest
```

## Notes

- Do not commit `.env`.
- Uploaded PDFs are local runtime artifacts.
- Chroma vector data is stored locally.
- API usage may be subject to provider rate limits.
- OpenAI API usage is separate from a ChatGPT subscription.

## Limitations

- Scanned PDFs without selectable text may not be extracted correctly.
- Citation quality depends on PDF text extraction and retrieval quality.
- The system currently focuses on single-PDF reading.
- This project is designed as a local MVP, not a production deployment.

## Future Improvements

- Support for multiple papers
- Better PDF parsing for tables and figures
- Citation preview with page numbers
- Conversation history persistence
- Evaluation scripts for retrieval quality
- Deployment-ready configuration