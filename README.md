# PaperMate RAG

PaperMate RAG is an early scaffold for a modular retrieval-augmented generation system focused on reading and exploring academic papers.

## Setup

Create and activate a virtual environment, then install the project in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Copy the example environment file and add your API key:

```powershell
Copy-Item .env.example .env
```

## Run Tests

```powershell
python -m pytest
```

## Run the Streamlit app

Set a provider API key before using real embedding and answering.

```powershell
conda activate papermate
cd D:\Development\papermate-rag
streamlit run src/papermate/ui/streamlit_app.py
```

Upload a PDF from the sidebar, index it, then ask questions in the chat.

## Gemini provider

Create a Gemini API key in Google AI Studio, then add this to `.env`:

```text
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Run the app:

```powershell
streamlit run src/papermate/ui/streamlit_app.py
```

Choose Gemini in Advanced settings. Upload a PDF, index it, and ask questions.

The OpenAI provider is still available if `OPENAI_API_KEY` is configured. The Gemini free tier is rate-limited.
