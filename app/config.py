"""Centralized, Pydantic-validated application settings."""

import os
from urllib.parse import quote, urlunsplit

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load and validate environment variables from `.env`.

    - `extra="ignore"` lets `.env` keep legacy keys (`POSTGRES_URI`, `REDIS_URL`,
      old Groq keys, etc.) without failing startup.
    - Everything except the LLM endpoint is optional, so the project can run with
      zero paid services (free hosted LLM like Groq, or a local Ollama model).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM: any OpenAI-compatible endpoint (Groq, Gemini, OpenRouter, Ollama, ...) ---
    # Defaults target Groq's free tier. For Ollama use:
    #   LLM_BASE_URL=http://localhost:11434/v1  LLM_API_KEY=ollama  LLM_MODEL=llama3.1:8b
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    # Max characters of retrieved context sent to the LLM. Lower this for free-tier
    # token-per-minute limits or small local context windows (e.g. 8000 for Ollama).
    MAX_CONTEXT_CHARS: int = 12000

    # Optional separate judge model for RAGAS evals (defaults to the LLM above).
    JUDGE_LLM_BASE_URL: str | None = None
    JUDGE_LLM_API_KEY: str | None = None
    JUDGE_LLM_MODEL: str | None = None

    # --- JINA AI (optional). Empty -> local mxbai embeddings; reranking is skipped. ---
    JINA_API_KEY: str = ""
    LOCAL_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LEGACY / OPTIONAL: OpenAI, Groq key alias, and the Portkey gateway ---
    # GROQ_API_KEY is accepted as an alias for LLM_API_KEY (see validator below).
    GROQ_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    JUDGE_OPENAI_API_KEY: str | None = None

    # --- PORTKEY LLM GATEWAY (only used when LLM_API_KEY is empty AND both Portkey values are set) ---
    PORTKEY_API_KEY: str = ""
    PORTKEY_PRIMARY_SLUG: str = "marathon-api"
    PORTKEY_FALLBACK_SLUG: str = "anthropic-fallback"
    # Portkey saved config is referenced by its system-generated `pc-...` ID.
    # Required when block_inline_config is enabled on the workspace.
    PORTKEY_PRIMARY_CONFIG_ID: str = ""

    # --- QDRANT VECTOR DB (local Docker: http://localhost:6333) ---
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("QDRANT_URL", "QDRANT_CLUSTER_ENDPOINT"),
    )
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "enterprise_rag"

    # --- NEON SERVERLESS POSTGRES (optional; empty -> in-memory LangGraph checkpointer) ---
    NEON_DB_URL: str = ""

    # --- UPSTASH REDIS (optional; empty -> in-memory rate limiting) ---
    UPSTASH_REDIS_REST_URL: str = ""
    UPSTASH_REDIS_REST_TOKEN: str = ""

    # --- API SAFETY ---
    API_KEY: str | None = Field(default=None, alias="RAG_API_KEY")
    RATE_LIMIT_PER_MINUTE: int = 20
    STRICT_STARTUP: bool = False

    # --- OBSERVABILITY ---
    LOGFIRE_TOKEN: str | None = None
    LOGFIRE_BASE_URL: str | None = None  # e.g. https://logfire-eu.pydantic.dev for EU tokens
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "rag_scale_test"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    @field_validator("QDRANT_API_KEY", mode="before")
    @classmethod
    def _empty_qdrant_key_as_none(cls, v):
        """Treat empty QDRANT_API_KEY as unset so local Qdrant doesn't receive a blank header."""
        if v == "" or v is None:
            return None
        return v

    @field_validator("QDRANT_URL", mode="before")
    @classmethod
    def _empty_qdrant_url_as_default(cls, v):
        """A blank QDRANT_URL / QDRANT_CLUSTER_ENDPOINT in .env falls back to local Qdrant."""
        if v is None or str(v).strip() == "":
            return "http://localhost:6333"
        return v

    @field_validator("JUDGE_LLM_BASE_URL", "JUDGE_LLM_API_KEY", "JUDGE_LLM_MODEL", "JUDGE_OPENAI_API_KEY", mode="before")
    @classmethod
    def _empty_judge_as_none(cls, v):
        return None if v == "" else v

    @model_validator(mode="after")
    def _groq_key_as_llm_key(self):
        """Existing .env files that only have GROQ_API_KEY keep working."""
        if not self.LLM_API_KEY and self.GROQ_API_KEY:
            self.LLM_API_KEY = self.GROQ_API_KEY
        return self

    @property
    def use_portkey(self) -> bool:
        """Legacy mode: route through Portkey only when no direct LLM key is configured."""
        return bool(not self.LLM_API_KEY and self.PORTKEY_API_KEY and self.PORTKEY_PRIMARY_CONFIG_ID)

    @property
    def has_postgres(self) -> bool:
        return bool(self.NEON_DB_URL)

    @property
    def has_redis(self) -> bool:
        return bool(self.UPSTASH_REDIS_REST_URL and self.UPSTASH_REDIS_REST_TOKEN)

    @property
    def judge_api_key(self) -> str:
        """Dedicated judge key, falling back to the main OpenAI key."""
        return self.JUDGE_OPENAI_API_KEY or self.OPENAI_API_KEY or ""

    @property
    def judge_config(self) -> tuple[str, str, str]:
        """(base_url, api_key, model) for the RAGAS judge; defaults to the main LLM."""
        return (
            self.JUDGE_LLM_BASE_URL or self.LLM_BASE_URL,
            self.JUDGE_LLM_API_KEY or self.LLM_API_KEY or self.JUDGE_OPENAI_API_KEY or "not-needed",
            self.JUDGE_LLM_MODEL or self.LLM_MODEL,
        )

    @property
    def postgres_uri(self) -> str:
        """LangGraph Postgres checkpointer URI (Neon).

        Serverless Postgres closes idle connections, so append TCP keepalive
        options to keep the connection pool healthy between requests.
        """
        base = self.NEON_DB_URL.rstrip("/")
        keepalive = "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
        if "?" in base:
            return f"{base}&{keepalive}"
        return f"{base}?{keepalive}"

    @property
    def redis_url(self) -> str:
        """TLS Redis URL derived from Upstash REST credentials.

        Upstash exposes the same host for REST and TLS Redis. The REST token is
        used as the Redis password under the default username. The result is
        passed to `limits` for rate limiting and to the health checker.
        """
        host = self.UPSTASH_REDIS_REST_URL.replace("https://", "").rstrip("/")
        token = quote(self.UPSTASH_REDIS_REST_TOKEN, safe="")
        netloc = f"default:{token}@{host}"
        return urlunsplit(("rediss", netloc, "/0", "ssl_cert_reqs=required", ""))


# Singleton used across the app.
settings = Settings()


def apply_langchain_env():
    """Write LangSmith/LangChain settings to os.environ for automatic tracing.

    Tracing is only activated when both LANGSMITH_TRACING and LANGSMITH_API_KEY
    are set — enabling tracing without a key causes LangChain to emit 401 noise
    on every LangGraph step.
    """
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.LANGSMITH_TRACING)
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGSMITH_API_KEY)
    if settings.LANGSMITH_PROJECT:
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGSMITH_PROJECT)
    if settings.LANGSMITH_ENDPOINT:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.LANGSMITH_ENDPOINT)


apply_langchain_env()
