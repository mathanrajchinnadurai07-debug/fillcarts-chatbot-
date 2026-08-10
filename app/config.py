"""
app/config.py
─────────────
Centralised configuration loaded from environment variables / .env file.
All other modules import from here — never read os.environ directly.
"""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings validated by Pydantic."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = "Curify AI Advisor"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    port: int = 8000
    host: str = "0.0.0.0"

    # ── API Security ───────────────────────────────────────────────────────────
    train_api_key: str = "change_me"
    chatbot_api_key: str = "change_me"

    # ── Gemini LLM ─────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # ── Database ───────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./curify_chat.db"

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "curify_knowledge"

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Classifier ─────────────────────────────────────────────────────────────
    model_dir: str = "./models/saved"
    confidence_threshold: float = 0.6
    top_k_retrieval: int = 3

    # ── Training Data ──────────────────────────────────────────────────────────
    training_data_path: str = "./data/raw/sample_training_data.csv"

    # ── CORS ───────────────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://localhost:8080"

    # ── LLM Response Cache ─────────────────────────────────────────────────────
    llm_cache_max_size: int = 256
    llm_cache_ttl_seconds: int = 300

    # ── Retrain ────────────────────────────────────────────────────────────────
    min_new_samples_for_retrain: int = 10

    @field_validator("gemini_api_key")
    @classmethod
    def warn_if_empty_api_key(cls, v: str) -> str:
        """Emit a warning (not an error) if the Gemini key is missing."""
        if not v:
            import warnings
            warnings.warn(
                "GEMINI_API_KEY is not set. LLM calls will fall back to "
                "rule-based responses.",
                RuntimeWarning,
                stacklevel=2,
            )
        return v

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse the comma-separated CORS_ORIGINS string into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        """True when running in production environment."""
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.

    Using lru_cache ensures the .env file is parsed only once per process,
    improving startup performance and avoiding repeated I/O.
    """
    return Settings()


# Module-level shortcut so callers can do: from app.config import settings
settings: Settings = get_settings()
