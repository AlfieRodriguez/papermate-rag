# AGENTS.md

## Project

This repository is **PaperMate RAG**, a modular Python RAG system for academic paper reading.

The goal is to support:

- PDF ingestion
- Page-level text extraction
- Chunking
- Embeddings
- Vector search
- Retrieval-augmented question answering
- Citation-based answers

## Environment

Use the existing conda environment:

```powershell
conda activate papermate
```

Run tests with:

```powershell
python -m pytest
```

Do **not** use plain `pytest`.

The project is developed on Windows with PowerShell, so prefer commands that work in PowerShell.

## Architecture Rules

Keep modules decoupled.

- `ingestion` only loads documents and extracts page-level text.
- `chunking` only converts page text into chunks.
- `embeddings` only converts text into vectors.
- `vectorstores` only stores and searches chunks.
- `retrieval` only coordinates embedding and vector search.
- `llm` only wraps model calls.
- `chains` only builds prompts and produces structured outputs.
- `services` coordinates modules for the UI.
- `ui` only handles Streamlit display and user interaction.

Do not put PDF parsing, vector database logic, prompt construction, or LLM calls inside the UI.

## Current Schema Rules

Use the schemas in:

```text
src/papermate/schemas.py
```

Use `doc_id` consistently.

Do **not** introduce `document_id`.

Important schemas:

- `PageText`
- `PaperDocument`
- `DocumentChunk`
- `RetrievedChunk`
- `Citation`
- `RAGAnswer`
- `PaperSummary`

## Coding Rules

- Use Python 3.11.
- Use type hints.
- Keep implementations simple and testable.
- Add or update tests for every feature.
- Do not implement future tasks unless explicitly asked.
- Do not modify unrelated files.
- Prefer small focused changes.
- Avoid over-engineering.
- Do not add heavy dependencies unless explicitly requested.
- Do not move logic across module boundaries without explaining why.

## Dependency Rules

Keep the MVP lightweight.

Core dependencies should stay focused on:

- `streamlit`
- `pydantic`
- `python-dotenv`
- `pymupdf`
- `chromadb`
- `openai`
- `pytest` for development/testing

Do not add `torch`, `sentence-transformers`, or other heavy local-model dependencies unless explicitly asked.

Local embedding support can be added later as an optional feature.

## Testing Rules

The preferred test commands are:

```powershell
python -m pytest tests/test_xxx.py
python -m pytest

Do not use plain pytest.

The user usually runs tests manually in Windows PowerShell with the papermate conda environment activated:

conda activate papermate
cd D:\Development\papermate-rag
python -m pytest

Codex may run in a shell that does not share the user's Windows conda PATH.

If this Codex environment cannot find python, py, conda, or pytest, do not modify the environment and do not install packages unless explicitly asked.

In that case:

State clearly that tests could not be run because the test runner is unavailable in this shell.
Provide the exact commands the user should run manually in PowerShell.
Do not claim tests passed.
Continue to summarize the changed files and assumptions.

For a single module task, ask the user to run the relevant test file first, then the full suite.

Example:

python -m pytest tests/test_chunker.py
python -m pytest

A task is complete only after the user confirms the relevant tests pass.

## Git / Review Rules

After each task, summarize:

1. Files changed
2. Main design decisions
3. Tests run
4. Test results
5. Any assumptions made

Do not commit automatically unless explicitly asked.

## Task Discipline

When given a task:

1. Read this file first.
2. Follow the allowed file list from the user prompt.
3. Implement only the requested task.
4. Do not implement future modules early.
5. Keep changes minimal.
6. Add tests for the new behavior.
7. Run the requested tests.

If a change requires editing a file outside the allowed list, explain why before doing it.

## Review Packet Rules

After each task, create a temporary review packet under:

```text
.review/
```

Example:

```text
.review/TASK_05_REVIEW.md
```

The review packet is for human review only and should not be committed.

The review packet must include:

1. Task Summary
   - What was implemented
   - What was intentionally not implemented

2. Files Changed
   - List every changed file
   - Explain the role of each file

3. Code Review Content
   - Include the full content of every new file created in this task
   - For modified existing files, include the relevant changed sections
   - If a file is short, include the full file content

4. Tests
   - List exact test commands
   - Include exact test output if tests were run
   - If tests could not be run, clearly explain why and provide PowerShell commands for the user

5. Design Notes
   - Important design decisions
   - Assumptions
   - Limitations or follow-up tasks

6. Review Focus
   - Specific things the human reviewer should check

Do not commit `.review/` files.
Do not use ellipses (`...`) or truncated code in review packets. Include full file contents for new files.