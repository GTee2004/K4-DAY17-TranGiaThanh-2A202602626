from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared paths, memory settings, and model configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    memory_confidence_threshold: float
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load the lab configuration without connecting to a model provider."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    load_dotenv(dotenv_path=root / ".env")

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    judge_provider = normalize_provider(
        os.getenv("JUDGE_LLM_PROVIDER", provider)
    )
    judge_model_name = os.getenv("JUDGE_LLM_MODEL", model_name)

    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "custom": os.getenv("CUSTOM_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "ollama": None,
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
    }
    base_urls = {
        "openai": None,
        "custom": os.getenv("CUSTOM_BASE_URL"),
        "gemini": None,
        "anthropic": None,
        "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "openrouter": "https://openrouter.ai/api/v1",
    }

    confidence_threshold = float(os.getenv("MEMORY_CONFIDENCE_THRESHOLD", "0.8"))
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("MEMORY_CONFIDENCE_THRESHOLD must be between 0 and 1")

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(
            os.getenv("COMPACT_THRESHOLD_TOKENS", "500")
        ),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        memory_confidence_threshold=confidence_threshold,
        model=ProviderConfig(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            api_key=api_keys[provider],
            base_url=base_urls[provider],
        ),
        judge_model=ProviderConfig(
            provider=judge_provider,
            model_name=judge_model_name,
            temperature=0.0,
            api_key=api_keys[judge_provider],
            base_url=base_urls[judge_provider],
        ),
    )
