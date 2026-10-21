"""LLM access layer.

Default: talk directly to any OpenAI-compatible endpoint (Groq, Gemini, OpenRouter,
Ollama, LM Studio, ...) configured via LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.

Legacy: when LLM_API_KEY is empty and PORTKEY_API_KEY + PORTKEY_PRIMARY_CONFIG_ID are
set, requests are routed through the Portkey gateway exactly as before.
"""

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI

from app.config import settings


def _portkey_headers(feature: str = "rag") -> dict:
    """Build Portkey headers that reference the primary saved config by ID."""
    from portkey_ai import createHeaders

    return createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        config_id=settings.PORTKEY_PRIMARY_CONFIG_ID,
        metadata={
            "feature": feature,
            "_user": "rag-system",
            "environment": "production",
        },
    )


def _endpoint(feature: str = "rag") -> tuple[str, str, dict | None]:
    """Return (base_url, api_key, default_headers) for the active LLM backend."""
    if settings.use_portkey:
        from portkey_ai import PORTKEY_GATEWAY_URL

        return PORTKEY_GATEWAY_URL, settings.PORTKEY_API_KEY, _portkey_headers(feature)
    # Local servers (Ollama, LM Studio) ignore the key, but the OpenAI SDK needs a non-empty one.
    return settings.LLM_BASE_URL, settings.LLM_API_KEY or "not-needed", None


def llm_model_name() -> str:
    """Model identifier to send in requests (Portkey needs the `@slug/model` form)."""
    if settings.use_portkey:
        return f"@{settings.PORTKEY_PRIMARY_SLUG}/{settings.LLM_MODEL}"
    return settings.LLM_MODEL


# Synchronous OpenAI-compatible client. The name is kept for backwards compatibility
# (responder + health check + tests import it) even though Portkey is now optional.
_base_url, _api_key, _headers = _endpoint()
portkey_client = OpenAI(api_key=_api_key, base_url=_base_url, default_headers=_headers)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """Returns a ChatOpenAI bound to the configured endpoint - a drop-in for LangChain nodes."""
    base_url, api_key, headers = _endpoint(feature)
    kwargs: dict = {}
    if not settings.use_portkey:
        # Deterministic output for planner/guardrail classification. Skipped in legacy
        # Portkey mode because reasoning models (gpt-5-mini) reject a custom temperature.
        kwargs["temperature"] = 0
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=llm_model_name(),
        default_headers=headers,
        **kwargs,
    )


def get_async_openai_client(feature: str = "rag") -> AsyncOpenAI:
    """
    Returns an async OpenAI-compatible client for the configured endpoint.
    Use this for non-LangChain async LLM calls (e.g. async FastAPI endpoints).
    """
    base_url, api_key, headers = _endpoint(feature)
    return AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=headers)


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the response (Portkey mode only).

    The OpenAI SDK does not expose raw headers on parsed responses, so cache
    hit/miss tracking is best-effort. We inspect common attribute paths and
    fall back to 'MISS'.
    """
    for attr in ("_raw_response", "_response", "_http_response", "headers"):
        raw = getattr(response, attr, None)
        if raw is not None:
            headers = getattr(raw, "headers", None)
            if headers is not None:
                status = headers.get("x-portkey-cache-status", "")
                if status:
                    return status.upper()
    return "MISS"
