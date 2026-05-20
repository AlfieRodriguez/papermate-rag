# Development

This project is organized as a modular RAG system. Business logic is intentionally left unimplemented in the initial scaffold.

```text
papermate-rag/
├── data/
│   ├── chroma/
│   ├── papers/
│   └── processed/
├── docs/
│   └── DEVELOPMENT.md
├── src/
│   └── papermate_rag/
│       ├── app.py
│       ├── chunking/
│       ├── config/
│       ├── embeddings/
│       ├── ingest/
│       ├── llm/
│       ├── parsing/
│       ├── pipelines/
│       ├── retrieval/
│       ├── ui/
│       └── vectorstore/
└── tests/
```

## Module Boundaries

- `config`: environment and runtime configuration.
- `ingest`: paper intake workflows.
- `parsing`: PDF and document parsing.
- `chunking`: text chunking strategies.
- `embeddings`: embedding provider integrations.
- `vectorstore`: Chroma and vector index integrations.
- `retrieval`: retrieval and reranking workflows.
- `llm`: language model provider integrations.
- `pipelines`: orchestration across modules.
- `ui`: Streamlit user interface components.
