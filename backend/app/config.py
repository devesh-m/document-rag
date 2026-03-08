from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Document Intelligence Assistant", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    database_url: str = Field(
        default="sqlite+aiosqlite:///./backend/rag_metadata.db",
        alias="DATABASE_URL",
    )
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="phi3", alias="OLLAMA_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    upload_dir: str = Field(default="storage/uploads", alias="UPLOAD_DIR")
    faiss_index_path: str = Field(default="vector_store/data/faiss.index", alias="FAISS_INDEX_PATH")
    faiss_metadata_path: str = Field(
        default="vector_store/data/faiss_metadata.json",
        alias="FAISS_METADATA_PATH",
    )

    @property
    def upload_path(self) -> Path:
        return PROJECT_ROOT / self.upload_dir

    @property
    def faiss_index_file(self) -> Path:
        return PROJECT_ROOT / self.faiss_index_path

    @property
    def faiss_metadata_file(self) -> Path:
        return PROJECT_ROOT / self.faiss_metadata_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
