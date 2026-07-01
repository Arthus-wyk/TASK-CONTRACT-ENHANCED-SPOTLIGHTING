from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from wait_tracker import tracked_sleep


OllamaCloudLimitAction = Literal["pause", "wait"]

OLLAMA_CLOUD_OPERATION_SUFFIXES = (
    ("chat", "completions", "create"),
    ("responses", "create"),
)


class OllamaCloudUsageLimitError(RuntimeError):
    def __init__(self, message: str, wait_seconds: float | None = None):
        super().__init__(message)
        self.wait_seconds = wait_seconds


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float_from_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    value = float(raw_value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value


def _action_from_env(default: OllamaCloudLimitAction = "pause") -> OllamaCloudLimitAction:
    raw_value = os.getenv("OLLAMA_CLOUD_LIMIT_ACTION", default).strip().lower()
    if raw_value not in {"pause", "wait"}:
        raise ValueError("OLLAMA_CLOUD_LIMIT_ACTION must be 'pause' or 'wait'")
    return raw_value  # type: ignore[return-value]


def _should_limit_operation(operation_path: tuple[str, ...]) -> bool:
    return any(
        operation_path[-len(suffix) :] == suffix
        for suffix in OLLAMA_CLOUD_OPERATION_SUFFIXES
    )


def _atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    try:
        temp_path.replace(path)
    except OSError:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        with suppress(OSError):
            temp_path.unlink()


def _response_usage_total(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0

    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and isinstance(usage, dict):
        total_tokens = usage.get("total_tokens")
    return int(total_tokens) if isinstance(total_tokens, (int, float)) and total_tokens > 0 else 0


def _exception_status_code(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    return int(status_code) if isinstance(status_code, int) else None


def _exception_text(exc: BaseException) -> str:
    payload = getattr(exc, "body", None) or getattr(exc, "response", None)
    parts = [str(exc)]
    if payload is not None:
        parts.append(str(payload))
    return "\n".join(parts).lower()


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        return None


def _is_usage_limit_error(exc: BaseException) -> bool:
    text = _exception_text(exc)
    return _exception_status_code(exc) == 429 and any(
        marker in text
        for marker in (
            "session usage",
            "weekly usage",
            "usage limit",
            "rate limit",
            "too many requests",
        )
    )


@dataclass
class OllamaCloudUsageLimiter:
    state_path: Path
    requests_per_minute: int = 10
    session_usage_limit: int = 0
    weekly_usage_limit: int = 0
    request_usage_units: int = 1
    session_reset_seconds: float = 3600.0
    weekly_reset_seconds: float = 7 * 24 * 3600.0
    limit_action: OllamaCloudLimitAction = "pause"

    def __post_init__(self) -> None:
        self.state_path = Path(self.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @classmethod
    def from_env(cls, state_path: Path) -> "OllamaCloudUsageLimiter":
        return cls(
            state_path=state_path,
            requests_per_minute=max(1, _positive_int_from_env("OLLAMA_CLOUD_RPM", 10)),
            session_usage_limit=_positive_int_from_env("OLLAMA_CLOUD_SESSION_USAGE_LIMIT", 0),
            weekly_usage_limit=_positive_int_from_env("OLLAMA_CLOUD_WEEKLY_USAGE_LIMIT", 0),
            request_usage_units=max(1, _positive_int_from_env("OLLAMA_CLOUD_REQUEST_USAGE_UNITS", 1)),
            session_reset_seconds=_positive_float_from_env("OLLAMA_CLOUD_SESSION_RESET_SECONDS", 3600.0),
            weekly_reset_seconds=_positive_float_from_env("OLLAMA_CLOUD_WEEKLY_RESET_SECONDS", 7 * 24 * 3600.0),
            limit_action=_action_from_env(),
        )

    def acquire(self) -> None:
        while True:
            with self._lock:
                state = self._normalized_state(self._load_state())
                self._raise_or_wait_if_usage_exhausted(state)
                now = time.time()
                recent = [
                    timestamp
                    for timestamp in state["recent_request_timestamps"]
                    if now - timestamp < 60.0
                ]
                if len(recent) < self.requests_per_minute:
                    recent.append(now)
                    state["recent_request_timestamps"] = recent
                    state["request_count"] += 1
                    self._save_state(state)
                    return

                oldest = min(recent)
                wait_seconds = max(0.1, oldest + 60.0 - now + 0.25)
                state["recent_request_timestamps"] = recent
                self._save_state(state)
                print(
                    "[ollama cloud rate limit] Local per-minute limit reached "
                    f"({self.requests_per_minute}/minute). Waiting {wait_seconds:.1f}s."
                )

            tracked_sleep(wait_seconds, reason="ollama_cloud_rate_limit_wait")

    def record_success(self, response: Any) -> None:
        usage_units = self.request_usage_units
        token_total = _response_usage_total(response)
        with self._lock:
            state = self._normalized_state(self._load_state())
            state["session_usage"] += usage_units
            state["weekly_usage"] += usage_units
            state["token_total"] += token_total
            self._save_state(state)

    def handle_limit_error(self, exc: BaseException) -> None:
        if not _is_usage_limit_error(exc):
            raise exc

        retry_after = _retry_after_seconds(exc)
        wait_seconds = retry_after if retry_after is not None else self.session_reset_seconds
        message = (
            "Ollama Cloud usage/rate limit reached. "
            f"Retry after approximately {wait_seconds:.0f} seconds."
        )
        if "weekly" in _exception_text(exc):
            wait_seconds = retry_after if retry_after is not None else self.weekly_reset_seconds
            message = (
                "Ollama Cloud weekly usage limit reached. "
                f"Retry after approximately {wait_seconds:.0f} seconds."
            )

        if self.limit_action == "pause":
            raise OllamaCloudUsageLimitError(message, wait_seconds) from exc

        print(f"[ollama cloud limit] {message} Waiting before continuing.")
        tracked_sleep(wait_seconds, reason="ollama_cloud_usage_limit_wait")

    def _raise_or_wait_if_usage_exhausted(self, state: dict[str, Any]) -> None:
        wait_seconds = 0.0
        reason = ""
        if self.session_usage_limit and state["session_usage"] >= self.session_usage_limit:
            wait_seconds = max(0.0, state["session_reset_at"] - time.time())
            reason = "session"
        if self.weekly_usage_limit and state["weekly_usage"] >= self.weekly_usage_limit:
            weekly_wait = max(0.0, state["weekly_reset_at"] - time.time())
            if weekly_wait > wait_seconds:
                wait_seconds = weekly_wait
                reason = "weekly"

        if not reason:
            return

        message = (
            f"Ollama Cloud local {reason} usage guard reached. "
            f"Reset in approximately {wait_seconds:.0f} seconds."
        )
        if self.limit_action == "pause":
            raise OllamaCloudUsageLimitError(message, wait_seconds)
        print(f"[ollama cloud limit] {message} Waiting before continuing.")
        tracked_sleep(wait_seconds, reason=f"ollama_cloud_{reason}_usage_wait")

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            with self.state_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _normalized_state(self, state: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        session_reset_at = float(state.get("session_reset_at") or 0)
        weekly_reset_at = float(state.get("weekly_reset_at") or 0)

        if now >= session_reset_at:
            state["session_usage"] = 0
            state["session_reset_at"] = now + self.session_reset_seconds
        if now >= weekly_reset_at:
            state["weekly_usage"] = 0
            state["weekly_reset_at"] = now + self.weekly_reset_seconds

        recent = state.get("recent_request_timestamps", [])
        if not isinstance(recent, list):
            recent = []
        recent = [float(item) for item in recent if isinstance(item, (int, float))]

        return {
            "updated_at": _utc_now().isoformat(),
            "session_usage": int(state.get("session_usage") or 0),
            "session_usage_limit": self.session_usage_limit,
            "session_reset_at": float(state["session_reset_at"]),
            "weekly_usage": int(state.get("weekly_usage") or 0),
            "weekly_usage_limit": self.weekly_usage_limit,
            "weekly_reset_at": float(state["weekly_reset_at"]),
            "request_count": int(state.get("request_count") or 0),
            "request_usage_units": self.request_usage_units,
            "token_total": int(state.get("token_total") or 0),
            "requests_per_minute": self.requests_per_minute,
            "recent_request_timestamps": recent,
        }

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _utc_now().isoformat()
        _atomic_json_dump(self.state_path, state)


class OllamaCloudClientProxy:
    def __init__(
        self,
        target: Any,
        limiter: OllamaCloudUsageLimiter,
        operation_path: tuple[str, ...] = (),
    ):
        self._target = target
        self._limiter = limiter
        self._operation_path = operation_path

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        operation_path = self._operation_path + (name,)
        if callable(attr):
            if not _should_limit_operation(operation_path):
                return attr

            def wrapped(*args, **kwargs):
                while True:
                    self._limiter.acquire()
                    try:
                        response = attr(*args, **kwargs)
                    except Exception as exc:
                        self._limiter.handle_limit_error(exc)
                        continue
                    self._limiter.record_success(response)
                    return response

            return wrapped

        if isinstance(attr, (str, int, float, bool, bytes, type(None))):
            return attr
        return OllamaCloudClientProxy(attr, self._limiter, operation_path)


def normalize_ollama_cloud_model_name(model_name: str) -> str:
    aliases = {
        "gpt-oss-20b": "gpt-oss:20b",
        "gpt-oss-120b": "gpt-oss:120b",
    }
    if model_name in aliases:
        return aliases[model_name]
    return re.sub(r"^gpt-oss-(\d+b)$", r"gpt-oss:\1", model_name)
