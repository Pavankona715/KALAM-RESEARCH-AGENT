"""
Configuration Management
========================
Single source of truth for all system configuration.
Uses Pydantic Settings to validate and load from environment variables.

Design: Every component reads from this module. Never import os.getenv() directly
in business logic - always go through settings.
"""

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMModel(str, Enum):
    """Supported LLM models via LiteLLM naming convention."""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    CLAUDE_SONNET = "claude-sonnet-4-5"
    CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
    GEMINI_PRO = "gemini/gemini-pro"
    GROQ_LLAMA = "groq/llama3-70b-8192"


class AppSettings(BaseSettings):
    """Core application settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Universal AI Research Agent"
    app_env: Environment = Environment.DEVELOPMENT
    app_port: int = 8000
    app_debug: bool = False
    secret_key: str = "change-this-in-production"

    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    default_llm_model: str = "gemini/gemini-2.0-flash"
    default_llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    default_llm_max_tokens: int = Field(default=4096, ge=1)

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_agent"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_session_ttl: int = 86400  # 24 hours

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_documents: str = "documents"
    qdrant_collection_memory: str = "long_term_memory"

    # Web Search
    tavily_api_key: Optional[str] = None

    # Observability
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "universal-ai-agent"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # MCP Integrations
    notion_api_key: Optional[str] = None
    slack_bot_token: Optional[str] = None
    google_drive_credentials_path: Optional[str] = None

    # Security
    jwt_secret: str = "change-this-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24

    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_day: int = 1_000_000

    # File Upload
    max_upload_size_mb: int = 50
    upload_dir: str = "./uploads"
    allowed_extensions: str = "pdf,docx,txt,md,html"

    @field_validator("default_llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.app_env == Environment.DEVELOPMENT

    @property
    def configured_llm_providers(self) -> list[str]:
        """Returns list of LLM providers that have API keys configured."""
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.gemini_api_key:
            providers.append("gemini")
        if self.groq_api_key:
            providers.append("groq")
        return providers


@lru_cache()
def get_settings() -> AppSettings:
    """
    Returns cached application settings.
    
    Uses lru_cache so settings are loaded once and reused.
    In tests, call get_settings.cache_clear() to reload.
    
    Usage:
        from backend.config.settings import get_settings
        settings = get_settings()
        print(settings.default_llm_model)
    """
    return AppSettings()


# Module-level singleton for convenience
settings = get_settings()