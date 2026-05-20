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

Set `OPENAI_API_KEY` before using real embedding and answering.

```powershell
conda activate papermate
cd D:\Development\papermate-rag
streamlit run src/papermate/ui/streamlit_app.py
```

Upload a PDF from the sidebar, index it, then ask questions in the chat.
