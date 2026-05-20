# PaperMate RAG Architecture

## Pipeline

```text
PDFLoader -> TextChunker -> Embedder -> VectorStore -> Retriever -> QAChain -> PaperService -> Streamlit UI
```

## Modules

### ingestion

Loads PDFs and extracts page-level text. The ingestion layer returns `PageText` objects and does not know about chunking, embeddings, retrieval, prompts, or UI.

### chunking

Converts page-level text into `DocumentChunk` objects. Chunks preserve document ID, source, page range, and metadata needed for later citations.

### embeddings

Converts text into vectors. The embedding interface allows multiple providers while keeping the rest of the pipeline provider-neutral.

### vector store

Stores embedded chunks and performs similarity search. The current implementation uses ChromaDB and returns `RetrievedChunk` objects.

### retriever

Coordinates query embedding and vector search. It takes a question, embeds it, asks the vector store for similar chunks, and returns retrieved evidence.

### LLM wrapper

Wraps provider-specific text generation APIs behind a small `BaseLLM` interface. The QA chain depends on this interface rather than on a provider SDK.

### QA chain

Builds the prompt from the question and retrieved context, calls the LLM, and returns a structured `RAGAnswer` with citations.

### service layer

`PaperService` coordinates ingestion and question answering for the UI. It uses dependency injection so tests can pass fake loaders, chunkers, embedders, vector stores, and QA chains.

### UI

The Streamlit UI handles upload, settings, chat messages, status display, and citation rendering. It delegates business logic to `PaperService`.

## Provider Abstraction

PaperMate supports separate provider implementations:

- `OpenAIEmbedder`
- `GeminiEmbedder`
- `OpenAILLM`
- `GeminiLLM`

The rest of the pipeline depends on small interfaces (`BaseEmbedder` and `BaseLLM`) rather than directly depending on provider SDKs. This keeps provider-specific code isolated and makes it easier to test.

## Why Dependency Injection Is Used

`PaperService` receives its dependencies in the constructor instead of constructing them internally. This keeps the service focused on orchestration, avoids hidden API or database setup, and makes unit tests fast and deterministic.

In tests, fake dependencies can verify calls and returned values without touching real PDFs, Chroma, Gemini, or OpenAI.

## Testing Strategy

- Unit tests use fake dependencies for orchestration modules.
- Tests do not call real OpenAI or Gemini APIs.
- Tests do not require a running Streamlit server.
- Module-level tests cover schemas, config, ingestion, chunking, embeddings, vector store behavior, retrieval, QA chain behavior, service orchestration, and UI helpers.
