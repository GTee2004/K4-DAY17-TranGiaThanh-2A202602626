from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_PROVIDERS = (
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
)


@dataclass
class ProviderConfig:
    """Configuration needed to construct one supported chat model."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return the canonical provider name, accepting documented aliases."""

    aliases = {
        "google": "gemini",
        "anthorpic": "anthropic",
        "claude": "anthropic",
    }
    normalized = value.strip().lower() if isinstance(value, str) else ""
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_PROVIDERS:
        valid = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unsupported provider {value!r}. Valid providers: {valid}."
        )
    return normalized


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the configured chat model using a provider-local import."""

    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url,
        )

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    valid = ", ".join(SUPPORTED_PROVIDERS)
    raise ValueError(f"Unsupported provider {provider!r}. Valid providers: {valid}.")
