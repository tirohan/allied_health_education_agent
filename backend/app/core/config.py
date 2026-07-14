from functools import lru_cache
from typing import Literal

from pydantic import AnyUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    api_key: SecretStr | None = None

    database_url: str = "postgresql://tanbinislam@localhost:5432/allied_health_edu"
    postgres_min_size: int = Field(default=1, ge=1)
    postgres_max_size: int = Field(default=10, ge=1)

    qdrant_url: AnyUrl = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_vector_size: int = Field(default=1536, ge=1)

    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    cache_ttl_seconds: int = Field(default=86_400, ge=0)
    mindmap_max_nodes: int = Field(default=50, ge=1, le=250)

    @property
    def qdrant_url_str(self) -> str:
        return str(self.qdrant_url).rstrip("/")


@lru_cache
def get_settings() -> Settings:
    return Settings()
