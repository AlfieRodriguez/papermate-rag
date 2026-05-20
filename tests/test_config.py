import pytest
from pydantic import ValidationError

from papermate.config import AppConfig, load_config


def test_config_defaults(monkeypatch) -> None:
    for key in (
        "OPENAI_API_KEY",
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "CHROMA_DIR",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
    ):
        monkeypatch.delenv(key, raising=False)

    config = AppConfig.from_env(env_file="missing.env")

    assert config.openai_api_key is None
    assert config.llm_provider == "openai"
    assert config.embedding_provider == "openai"
    assert config.chroma_dir == "data/chroma"
    assert config.chunk_size == 800
    assert config.chunk_overlap == 150
    assert config.top_k == 5


def test_config_loads_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setenv("CHROMA_DIR", "tmp/chroma")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv("TOP_K", "8")

    config = load_config(env_file="missing.env")

    assert config.openai_api_key == "test-key"
    assert config.llm_provider == "local"
    assert config.embedding_provider == "sentence-transformers"
    assert config.chroma_dir == "tmp/chroma"
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200
    assert config.top_k == 8


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(
        ValidationError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        AppConfig(chunk_size=100, chunk_overlap=100)
