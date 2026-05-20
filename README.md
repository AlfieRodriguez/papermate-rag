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
pytest
```

## Run the App

The Streamlit app package is scaffolded, but application behavior has not been implemented yet.

```powershell
streamlit run src/papermate_rag/app.py
```
