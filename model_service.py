"""Local provider-error detection, independent of task/security decisions."""
from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from wait_tracker import tracked_sleep


_text_only = ContextVar("model_service_text_only", default=False)
_response_schema = ContextVar("model_service_response_schema", default=None)
_task_status = ContextVar("model_service_task_status", default=None)


class QuotaExhaustedError(SystemExit):
    """Stop the run through broad Exception handlers, leaving this task pending."""
    def __init__(self, message: str, wait_seconds: float | None = None):
        super().__init__(2)
        self.message = message
        self.wait_seconds = wait_seconds

    def __str__(self):
        return self.message


def is_quota_exhausted(status, payload) -> bool:
    text = str(payload).lower()
    return status in {402, 429} and (status == 402 or any(marker in text for marker in (
        "insufficient_quota", "insufficient credits", "credit balance", "quota exceeded",
        "quota exhausted", "session usage", "weekly usage", "usage limit",
    )))


@contextmanager
def capture_task_status():
    status = {}
    token = _task_status.set(status)
    try:
        yield status
    finally:
        _task_status.reset(token)


def mark_task_unavailable(reason):
    status = _task_status.get()
    if status is not None:
        status["unavailable"] = reason


class ModelServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, *, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.body = {"error": {"message": message, "code": status_code}}


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def check_response(response: Any) -> None:
    """Inspect only the provider envelope, never search ordinary content for keywords."""
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except (ValueError, TypeError):
            return
    error = _field(response, "error")
    status = _field(response, "status_code")
    if not error and not (isinstance(status, int) and status >= 400):
        return
    code = _field(error, "code", _field(error, "type"))
    if not isinstance(status, int):
        status = int(code) if str(code).isdigit() else None
    if status is None and code in {"rate_limit_exceeded", "rate_limit_error", "insufficient_quota"}:
        status = 429
    message = str(_field(error, "message", error) or f"Provider HTTP {status}")
    exc = ModelServiceError(message, status, retryable=status in {408, 429, 500, 502, 503, 504})
    # Preserve structured quota codes for the outer limiter (including key rotation).
    exc.body = {"error": error}
    raise exc


def service_error(exc: Exception) -> ModelServiceError | None:
    status = getattr(exc, "status_code", None) or _field(getattr(exc, "response", None), "status_code")
    if is_quota_exhausted(status, (str(exc), getattr(exc, "body", None))):
        raise QuotaExhaustedError(str(exc)) from exc
    if isinstance(exc, ModelServiceError):
        return exc
    # Existing local quota exceptions intentionally support pause/wait policies.
    if any(cls.__name__ in {"OpenRouterRateLimitError", "OllamaCloudUsageLimitError"}
           for cls in type(exc).__mro__):
        raise QuotaExhaustedError(str(exc)) from exc
    status = getattr(exc, "status_code", None) or _field(getattr(exc, "response", None), "status_code")
    module = type(exc).__module__
    sdk_error = module.startswith(("openai", "anthropic", "google.genai", "httpx", "httpcore"))
    if isinstance(status, int) and status >= 400:
        return ModelServiceError(str(exc), status, retryable=status in {408, 429, 500, 502, 503, 504})
    if sdk_error:
        return ModelServiceError(str(exc), retryable="connect" in type(exc).__name__.lower()
                                 or "timeout" in type(exc).__name__.lower())
    return None


@contextmanager
def text_only_request():
    token = _text_only.set(True)
    try:
        yield
    finally:
        _text_only.reset(token)


@contextmanager
def structured_output_request(schema):
    token = _response_schema.set(schema)
    try:
        with text_only_request():
            yield
    finally:
        _response_schema.reset(token)


class TextOnlyRuntime:
    """Expose no executable tools to an internal judge, without mutating the agent."""
    def __init__(self, runtime):
        self._runtime = runtime
        self.functions = {}
        self.tools = []

    def __getattr__(self, name):
        return getattr(self._runtime, name)


class ProviderResponseGuard:
    """Check raw SDK responses before AgentDojo parses choices/content."""
    def __init__(self, target, path=()):
        self._target = target
        self._path = path

    @property
    def api_key(self):
        return self._target.api_key

    @api_key.setter
    def api_key(self, value):
        self._target.api_key = value

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        path = (*self._path, name)
        request = any(path[-len(suffix):] == suffix for suffix in (
            ("chat", "completions", "create"), ("responses", "create"),
            ("messages", "create"), ("models", "generate_content"),
        ))
        if callable(attr):
            if not request:
                return attr
            def wrapped(*args, **kwargs):
                if _text_only.get():
                    kwargs = dict(kwargs)
                    if path[-3:] == ("chat", "completions", "create") or path[-2:] == ("responses", "create"):
                        kwargs["tool_choice"] = "none"
                    elif path[-2:] == ("messages", "create"):
                        kwargs.pop("tools", None)
                        kwargs.pop("tool_choice", None)
                schema = _response_schema.get()
                if schema is not None:
                    kwargs = dict(kwargs)
                    format_spec = {"type": "json_schema", "name": "tces_decision", "strict": True, "schema": schema}
                    if path[-3:] == ("chat", "completions", "create"):
                        kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                            key: value for key, value in format_spec.items() if key != "type"
                        }}
                    elif path[-2:] == ("responses", "create"):
                        kwargs["text"] = {**(kwargs.get("text") or {}), "format": format_spec}
                result = attr(*args, **kwargs)
                check_response(result)
                return result
            return wrapped
        if isinstance(attr, (str, int, float, bool, bytes, type(None))):
            return attr
        return ProviderResponseGuard(attr, path)


def query_model(llm, query, runtime, env, messages, extra_args, *, max_attempts=3):
    """Service retries have their own budget; never ask an LLM to diagnose them."""
    for attempt in range(max_attempts):
        try:
            result = llm.query(query, runtime, env, messages, extra_args)
            generated = result[3] or []
            if len(generated) > len(messages):
                last = generated[-1]
                check_response(last)
                content = _field(last, "content")
                if isinstance(content, list):
                    content = "\n".join(str(_field(c, "content", "")) for c in content
                                        if _field(c, "type") == "text")
                # Compatibility for providers returning an error envelope as text.
                if isinstance(content, str):
                    check_response(content)
            return result
        except Exception as exc:
            error = service_error(exc)
            if error is None:
                raise
            extra_args.setdefault("model_service_events", []).append({
                "status_code": error.status_code,
                "event": "retry" if error.retryable and attempt + 1 < max_attempts else "unavailable",
                "attempt": attempt + 1,
            })
            if not error.retryable or attempt + 1 >= max_attempts:
                raise error from exc
            tracked_sleep(2 ** attempt, reason="model_service_retry")
