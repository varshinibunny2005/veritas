"""Central configuration (reads .env)."""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # ---- LLM ----
    llm_provider: str = "groq"  # groq | openai | anthropic | ollama | offline
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-120b"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-haiku-4-5-20251001"
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_s: int = 90
    llm_max_retries: int = 4

    # ---- Embeddings / RAG ----
    embedding_provider: str = "fastembed"  # fastembed | openai | hash
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_dir: Path = BASE_DIR / "storage" / "chroma"
    fastembed_cache_dir: Path = BASE_DIR / "storage" / "fastembed"
    kb_dir: Path = BASE_DIR / "data" / "knowledge_base"
    chunk_size: int = 700
    chunk_overlap: int = 100
    retrieval_top_k: int = 4
    claim_top_k: int = 3
    min_relevance: Optional[float] = None  # None = sensible default per embedding provider

    # ---- Pipeline ----
    max_claims: int = 8
    selfcheck_samples: int = 2
    max_concurrency: int = 4
    request_timeout_s: int = 240

    # ---- Arbiter weights ----
    weight_accuracy: float = 0.30
    weight_grounding: float = 0.30
    weight_completeness: float = 0.20
    weight_consistency: float = 0.20

    @property
    def weights(self) -> dict:
        return {
            "accuracy": self.weight_accuracy,
            "grounding": self.weight_grounding,
            "completeness": self.weight_completeness,
            "consistency": self.weight_consistency,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
