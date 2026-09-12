from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Doc Investigator", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(
        default="http://localhost:5175,http://127.0.0.1:5175,*",
        alias="CORS_ORIGINS",
    )
    database_url: str = Field(
        default="postgresql+psycopg://docs:docs@localhost:5434/doc_investigator",
        alias="DATABASE_URL",
    )
    qdrant_url: str = Field(default="http://127.0.0.1:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="doc_passages", alias="QDRANT_COLLECTION")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash-lite", alias="GEMINI_MODEL")
    embedding_model: str = Field(default="models/gemini-embedding-001", alias="EMBEDDING_MODEL")
    agent_max_steps: int = Field(default=8, alias="AGENT_MAX_STEPS")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def resolved_embedding_model(self) -> str:
        retired = {
            "text-embedding-004": "models/gemini-embedding-001",
            "models/text-embedding-004": "models/gemini-embedding-001",
            "embedding-001": "models/gemini-embedding-001",
            "models/embedding-001": "models/gemini-embedding-001",
        }
        name = (self.embedding_model or "").strip()
        return retired.get(name, name) or "models/gemini-embedding-001"

    @property
    def demo_policy_path(self) -> Path:
        return PROJECT_ROOT / "demo" / "vendor_policy.txt"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
