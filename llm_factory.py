from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import anthropic
import openai
from google import genai
from openai.types.chat import ChatCompletionReasoningEffort

from agentdojo import agent_pipeline
from rate_limiter import DailyLimitAction, OpenRouterRateLimiter, RateLimitedClientProxy
from token_usage import TokenUsageLogger, UsageLoggingProxy


ModelProvider = Literal["openai", "ollama", "anthropic", "google", "openrouter"]


class _SleepAfterCallProxy:
    def __init__(self, target, sleep_seconds: float):
        self._target = target
        self._sleep_seconds = sleep_seconds

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if callable(attr):
            def wrapped(*args, **kwargs):
                try:
                    return attr(*args, **kwargs)
                finally:
                    time.sleep(self._sleep_seconds)

            return wrapped
        return _SleepAfterCallProxy(attr, self._sleep_seconds) if hasattr(attr, "__dict__") else attr


def parse_model(model: str) -> tuple[ModelProvider, str]:
    if ":" not in model:
        raise ValueError(
            "model must be provider:model_name, for example: "
            "openai:gpt-4o-mini-2024-07-18 or ollama:qwen2.5:7b"
        )

    provider, model_name = model.split(":", 1)

    if provider not in {"openai", "ollama", "anthropic", "google", "openrouter"}:
        raise ValueError(f"Unsupported provider: {provider}")

    return provider, model_name


def is_openai_reasoning_model(model_name: str) -> bool:
    return "o1" in model_name or "o3" in model_name or "o4" in model_name or "codex" in model_name


def make_llm(
    model: str,
    reasoning_effort: ChatCompletionReasoningEffort = "medium",
    thinking_budget_tokens: int | None = None,
    token_log_dir: Path | None = None,
    rate_limit_state_path: Path | None = None,
    daily_limit_action: DailyLimitAction = "wait",
):
    provider, model_name = parse_model(model)
    token_logger = (
        TokenUsageLogger(token_log_dir, provider=provider, model=model_name)
        if token_log_dir is not None
        else None
    )

    if provider == "openai":
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if token_logger is not None:
            client = UsageLoggingProxy(client, token_logger)
        if is_openai_reasoning_model(model_name):
            llm = agent_pipeline.OpenAILLM(client, model_name, reasoning_effort, None)
        else:
            llm = agent_pipeline.OpenAILLM(client, model_name, None)
    elif provider == "ollama":
        client = openai.OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        )
        if token_logger is not None:
            client = UsageLoggingProxy(client, token_logger)
        llm = agent_pipeline.OpenAILLM(client, model_name, None)
    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        if token_logger is not None:
            client = UsageLoggingProxy(client, token_logger)
        max_tokens = 8192 + thinking_budget_tokens if thinking_budget_tokens else 8192
        llm = agent_pipeline.AnthropicLLM(
            client,
            model_name,
            thinking_budget_tokens=thinking_budget_tokens,
            max_tokens=max_tokens,
        )
    elif provider == "google":
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        if token_logger is not None:
            client = UsageLoggingProxy(client, token_logger)
        client = _SleepAfterCallProxy(client, 60.0)
        llm = agent_pipeline.GoogleLLM(model_name, client, max_tokens=8192)
    elif provider == "openrouter":
        client = openai.OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        if token_logger is not None:
            client = UsageLoggingProxy(client, token_logger)

        state_path = rate_limit_state_path or Path("./logs_my_agentdojo/openrouter_rate_limit_state.json")
        limiter = OpenRouterRateLimiter.from_env(
            state_path=state_path,
            daily_limit_action=daily_limit_action,
        )
        client = RateLimitedClientProxy(client, limiter)
        llm = agent_pipeline.OpenAILLM(client, model_name, None)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    llm.name = model_name
    # llm = _SleepAfterCallProxy(llm, 60.0)
    return llm
