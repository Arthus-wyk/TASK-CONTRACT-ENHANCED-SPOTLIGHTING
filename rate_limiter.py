from __future__ import annotations

import json
import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal


DailyLimitAction = Literal["pause", "wait"]

RATE_LIMITED_OPERATION_SUFFIXES = (
    ("chat", "completions", "create"),
    ("responses", "create"),
)
TEMPORARY_429_RETRY_SECONDS = 60.0


class OpenRouterRateLimitError(RuntimeError):
    """Raised when OpenRouter quota prevents more model requests."""


class OpenRouterDailyLimitReached(OpenRouterRateLimitError):
    def __init__(self, message: str, wait_seconds: float):
        super().__init__(message)
        self.wait_seconds = wait_seconds


class OpenRouterCreditLimitReached(OpenRouterRateLimitError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_date_string(now: datetime | None = None) -> str:
    return (now or _utc_now()).date().isoformat()


def _parse_utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _seconds_until_window_reset(window_start: datetime, buffer_seconds: float = 1.0) -> float:
    reset_at = window_start + timedelta(days=1)
    return max(0.0, (reset_at - _utc_now()).total_seconds() + buffer_seconds)


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float_from_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _daily_action_from_env(default: DailyLimitAction = "wait") -> DailyLimitAction:
    raw_value = os.getenv("OPENROUTER_DAILY_LIMIT_ACTION", default).strip().lower()
    if raw_value not in {"pause", "wait"}:
        raise ValueError("OPENROUTER_DAILY_LIMIT_ACTION must be 'pause' or 'wait'")
    return raw_value  # type: ignore[return-value]


def _openrouter_keys_from_env() -> list[str]:
    raw_keys = os.getenv("OPENROUTER_API_KEYS")
    if raw_keys:
        keys = [key.strip() for key in re.split(r"[,;\n\r]+", raw_keys) if key.strip()]
    else:
        single_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        keys = [single_key] if single_key else []

    deduplicated_keys = []
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        deduplicated_keys.append(key)
    return deduplicated_keys


def _key_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def _key_label(api_key: str) -> str:
    suffix = api_key[-4:] if len(api_key) >= 4 else "????"
    return f"{_key_id(api_key)}...{suffix}"


def _should_rate_limit_operation(operation_path: tuple[str, ...]) -> bool:
    return any(
        operation_path[-len(suffix) :] == suffix
        for suffix in RATE_LIMITED_OPERATION_SUFFIXES
    )


def _response_retry_after_seconds(response: Any) -> float | None:
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = None
    get_header = getattr(headers, "get", None)
    if callable(get_header):
        retry_after = get_header("retry-after") or get_header("Retry-After")
    elif isinstance(headers, dict):
        retry_after = headers.get("retry-after") or headers.get("Retry-After")

    if retry_after is None:
        return None

    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None


def _find_retry_after_seconds(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("retry_after_seconds", "retry_after_seconds_raw"):
            retry_after = value.get(key)
            if isinstance(retry_after, (int, float)) and retry_after > 0:
                return float(retry_after)
            if isinstance(retry_after, str):
                try:
                    parsed = float(retry_after)
                except ValueError:
                    parsed = None
                if parsed and parsed > 0:
                    return parsed

        for item in value.values():
            retry_after = _find_retry_after_seconds(item)
            if retry_after is not None:
                return retry_after

    if isinstance(value, list):
        for item in value:
            retry_after = _find_retry_after_seconds(item)
            if retry_after is not None:
                return retry_after

    return None


def _exception_payload(exc: BaseException) -> Any:
    body = getattr(exc, "body", None)
    if body is not None:
        return body

    response = getattr(exc, "response", None)
    if response is not None:
        json_method = getattr(response, "json", None)
        if callable(json_method):
            try:
                return json_method()
            except Exception:
                return None

    return None


def _exception_status_code(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _retry_delay_for_exception(
    exc: BaseException,
    attempt: int,
    base_seconds: float,
    max_seconds: float,
) -> float | None:
    status_code = _exception_status_code(exc)
    if status_code != 429:
        return None

    return TEMPORARY_429_RETRY_SECONDS


def _walk_json_values(value: Any):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_json_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json_values(item)


def _exception_text(exc: BaseException) -> str:
    payload = _exception_payload(exc)
    parts = [str(exc)]
    if payload is not None:
        for value in _walk_json_values(payload):
            if isinstance(value, (str, int, float, bool)):
                parts.append(str(value))
    return "\n".join(parts).lower()


def _is_daily_key_limit_error(exc: BaseException) -> bool:
    if _exception_status_code(exc) != 429:
        return False

    text = _exception_text(exc)
    daily_markers = (
        "free-models-per-day",
        "free models per day",
        "add 10 credits",
    )
    if any(marker in text for marker in daily_markers):
        return True

    return (
        "rate limit exceeded" in text
        and "x-ratelimit-remaining" in text
        and "\n0\n" in f"\n{text}\n"
    )


def _is_temporary_rate_limit_error(exc: BaseException) -> bool:
    if _exception_status_code(exc) != 429:
        return False
    text = _exception_text(exc)
    temporary_markers = (
        "temporarily rate-limited upstream",
        "temporarily rate limited upstream",
        "retry shortly",
    )
    return any(marker in text for marker in temporary_markers)


def _json_safe(value: Any, max_string_length: int = 2000) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, str):
        if len(value) <= max_string_length:
            return value
        return value[:max_string_length] + "...<truncated>"

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, max_string_length=max_string_length)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item, max_string_length=max_string_length) for item in value]

    return _json_safe(repr(value), max_string_length=max_string_length)


@dataclass
class OpenRouterRateLimiter:
    state_path: Path
    requests_per_minute: int = 20
    daily_request_limit: int = 50
    daily_limit_action: DailyLimitAction = "wait"
    api_keys: list[str] | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    utc_reset_buffer_seconds: float = 1.0
    retry_max_attempts: int = 10
    retry_base_seconds: float = 60.0
    retry_max_seconds: float = 300.0
    retry_daily_limit: int = 50
    error_log_path: Path | None = None

    def __post_init__(self) -> None:
        self.state_path = Path(self.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.error_log_path = Path(self.error_log_path) if self.error_log_path else self.state_path.with_name(
            "openrouter_error_log.jsonl"
        )
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.api_keys = self.api_keys or []
        if not self.api_keys:
            raise ValueError("Set OPENROUTER_API_KEY or OPENROUTER_API_KEYS before using openrouter models.")
        self._lock = threading.RLock()

    @classmethod
    def from_env(
        cls,
        state_path: Path,
        daily_limit_action: DailyLimitAction | None = None,
    ) -> "OpenRouterRateLimiter":
        return cls(
            state_path=state_path,
            requests_per_minute=_positive_int_from_env("OPENROUTER_RPM", 20),
            daily_request_limit=_positive_int_from_env("OPENROUTER_DAILY_LIMIT", 50),
            daily_limit_action=daily_limit_action or _daily_action_from_env(),
            api_keys=_openrouter_keys_from_env(),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            retry_max_attempts=_positive_int_from_env("OPENROUTER_RETRY_MAX_ATTEMPTS", 10),
            retry_base_seconds=_positive_float_from_env("OPENROUTER_RETRY_BASE_SECONDS", 60.0),
            retry_max_seconds=_positive_float_from_env("OPENROUTER_RETRY_MAX_SECONDS", 300.0),
            retry_daily_limit=_positive_int_from_env("OPENROUTER_RETRY_DAILY_LIMIT", 50),
            error_log_path=Path(os.getenv("OPENROUTER_ERROR_LOG_PATH"))
            if os.getenv("OPENROUTER_ERROR_LOG_PATH")
            else None,
        )

    @property
    def first_api_key(self) -> str:
        return self.api_keys[0]

    def acquire(self) -> str:
        while True:
            with self._lock:
                state = self._normalized_state(self._load_state())
                now = time.time()
                active_key_index = int(state.get("active_key_index", 0))
                window_start = _parse_utc_datetime(state.get("daily_window_start_utc")) or _utc_now()

                active_key_index = self._next_available_key_index(state, active_key_index)
                if active_key_index is None:
                    self._save_state(state)
                    wait_seconds = _seconds_until_window_reset(window_start, self.utc_reset_buffer_seconds)
                    message = (
                        "All OpenRouter keys reached today's free-model limit "
                        f"({len(self.api_keys)} keys). "
                        f"The 24h window started at {window_start.isoformat()} and resets "
                        f"in {wait_seconds:.0f} seconds."
                    )
                    if self.daily_limit_action == "pause":
                        raise OpenRouterDailyLimitReached(message, wait_seconds)

                    print(f"[openrouter rate limit] {message} Waiting before continuing.")
                else:
                    api_key = self.api_keys[active_key_index]
                    key_state = state["keys"][_key_id(api_key)]
                    key_state["recent_request_timestamps"] = [
                        timestamp
                        for timestamp in key_state["recent_request_timestamps"]
                        if now - timestamp < 60.0
                    ]

                    if len(key_state["recent_request_timestamps"]) < self.requests_per_minute:
                        key_state["recent_request_timestamps"].append(now)
                        key_state["daily_request_count"] += 1
                        state["active_key_index"] = active_key_index
                        state["current_key"] = _key_label(api_key)
                        self._save_state(state)
                        return api_key

                    oldest = min(key_state["recent_request_timestamps"])
                    wait_seconds = max(0.1, oldest + 60.0 - now + 0.25)
                    self._save_state(state)
                    print(
                        "[openrouter rate limit] Current OpenRouter key is at the local "
                        f"per-minute limit ({self.requests_per_minute}/minute/key). "
                        f"Waiting {wait_seconds:.1f}s before reusing the same key."
                    )

            time.sleep(wait_seconds)

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
        now = _utc_now()
        window_start = _parse_utc_datetime(state.get("daily_window_start_utc"))
        if window_start is None:
            legacy_date = state.get("date_utc")
            if isinstance(legacy_date, str):
                window_start = _parse_utc_datetime(f"{legacy_date}T00:00:00+00:00")
        if window_start is None:
            window_start = now

        if now >= window_start + timedelta(days=1):
            window_start = now
            state = {
                "daily_window_start_utc": window_start.isoformat(),
                "active_key_index": int(state.get("active_key_index", 0) or 0),
                "keys": {},
            }

        window_start_iso = window_start.isoformat()
        keys_state = state.get("keys")
        if not isinstance(keys_state, dict):
            keys_state = {}

        # Migrate the old single-key state shape if present.
        old_daily_count = state.get("daily_request_count", 0)
        old_recent_timestamps = state.get("recent_request_timestamps", [])

        normalized_keys: dict[str, Any] = {}
        for index, api_key in enumerate(self.api_keys):
            key_id = _key_id(api_key)
            key_state = keys_state.get(key_id)
            if not isinstance(key_state, dict):
                key_state = {}
                if index == 0:
                    key_state["daily_request_count"] = old_daily_count
                    key_state["recent_request_timestamps"] = old_recent_timestamps

            daily_request_count = key_state.get("daily_request_count", 0)
            if not isinstance(daily_request_count, int) or daily_request_count < 0:
                daily_request_count = 0

            recent_timestamps = key_state.get("recent_request_timestamps", [])
            if not isinstance(recent_timestamps, list):
                recent_timestamps = []
            recent_timestamps = [
                float(timestamp)
                for timestamp in recent_timestamps
                if isinstance(timestamp, (int, float))
            ]

            normalized_key_state = {
                "label": _key_label(api_key),
                "daily_request_count": daily_request_count,
                "retry_count": key_state.get("retry_count", 0)
                if isinstance(key_state.get("retry_count", 0), int)
                and key_state.get("retry_count", 0) >= 0
                else 0,
                "recent_request_timestamps": recent_timestamps,
            }
            exhausted_window = key_state.get("daily_exhausted_window_start_utc")
            legacy_exhausted_date = key_state.get("credit_exhausted_date_utc")
            if isinstance(exhausted_window, str) and exhausted_window == window_start_iso:
                normalized_key_state["daily_exhausted_window_start_utc"] = exhausted_window
                normalized_key_state["daily_exhausted_at_utc"] = key_state.get("daily_exhausted_at_utc")
            elif isinstance(legacy_exhausted_date, str) and legacy_exhausted_date == _utc_date_string(window_start):
                normalized_key_state["daily_exhausted_window_start_utc"] = window_start_iso
                normalized_key_state["daily_exhausted_at_utc"] = key_state.get("daily_exhausted_at_utc")
            normalized_keys[key_id] = normalized_key_state

        active_key_index = state.get("active_key_index", 0)
        if not isinstance(active_key_index, int) or active_key_index < 0:
            active_key_index = 0
        active_key_index %= len(self.api_keys)

        return {
            "date_utc": _utc_date_string(window_start),
            "daily_window_start_utc": window_start_iso,
            "daily_window_reset_utc": (window_start + timedelta(days=1)).isoformat(),
            "active_key_index": active_key_index,
            "key_count": len(self.api_keys),
            "keys": normalized_keys,
            "requests_per_minute": self.requests_per_minute,
            "daily_limit_action": self.daily_limit_action,
            "updated_at": _utc_now().isoformat(),
        }

    def _key_is_daily_exhausted(self, state: dict[str, Any], api_key: str) -> bool:
        key_state = state["keys"][_key_id(api_key)]
        return key_state.get("daily_exhausted_window_start_utc") == state["daily_window_start_utc"]

    def _next_available_key_index(self, state: dict[str, Any], start_index: int) -> int | None:
        for offset in range(len(self.api_keys)):
            key_index = (start_index + offset) % len(self.api_keys)
            if not self._key_is_daily_exhausted(state, self.api_keys[key_index]):
                return key_index
        return None

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _utc_now().isoformat()
        with self.state_path.open("w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)

    def record_model_error(
        self,
        *,
        event: Literal["retry", "crash"],
        operation: str,
        api_key: str,
        attempt: int,
        max_attempts: int,
        exc: BaseException,
        retry_delay_seconds: float | None = None,
        reason: str | None = None,
    ) -> None:
        entry = {
            "timestamp": _utc_now().isoformat(),
            "event": event,
            "operation": operation,
            "key": _key_label(api_key),
            "attempt": attempt,
            "max_attempts": max_attempts,
            "retry_delay_seconds": retry_delay_seconds,
            "reason": reason,
            "exception_type": exc.__class__.__name__,
            "status_code": _exception_status_code(exc),
            "message": _json_safe(str(exc)),
            "payload": _json_safe(_exception_payload(exc)),
        }
        with self.error_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def record_retry_for_key(self, api_key: str) -> int:
        with self._lock:
            state = self._normalized_state(self._load_state())
            key_id = _key_id(api_key)
            key_state = state["keys"][key_id]
            key_state["retry_count"] += 1
            state["active_key_index"] = self.api_keys.index(api_key)
            self._save_state(state)
            return key_state["retry_count"]

    def mark_key_daily_exhausted(self, api_key: str) -> bool:
        with self._lock:
            state = self._normalized_state(self._load_state())
            key_index = self.api_keys.index(api_key)
            key_state = state["keys"][_key_id(api_key)]
            key_state["daily_exhausted_window_start_utc"] = state["daily_window_start_utc"]
            key_state["daily_exhausted_at_utc"] = _utc_now().isoformat()
            next_index = self._next_available_key_index(state, (key_index + 1) % len(self.api_keys))
            if next_index is not None:
                state["active_key_index"] = next_index
                state["current_key"] = _key_label(self.api_keys[next_index])
                has_next_key = True
            else:
                state["active_key_index"] = key_index
                state["current_key"] = None
                has_next_key = False
            self._save_state(state)
            return has_next_key


class RateLimitedClientProxy:
    def __init__(
        self,
        target: Any,
        limiter: OpenRouterRateLimiter,
        operation_path: tuple[str, ...] = (),
        root_target: Any | None = None,
    ):
        self._target = target
        self._limiter = limiter
        self._operation_path = operation_path
        self._root_target = target if root_target is None else root_target

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        operation_path = self._operation_path + (name,)

        if callable(attr):
            if not _should_rate_limit_operation(operation_path):
                return attr

            def wrapped(*args, **kwargs):
                attempt = 1
                while True:
                    api_key = self._limiter.acquire()
                    self._apply_api_key(api_key)
                    try:
                        return attr(*args, **kwargs)
                    except Exception as exc:
                        retry_delay = _retry_delay_for_exception(
                            exc=exc,
                            attempt=attempt,
                            base_seconds=self._limiter.retry_base_seconds,
                            max_seconds=self._limiter.retry_max_seconds,
                        )
                        if retry_delay is None:
                            self._limiter.record_model_error(
                                event="crash",
                                operation=".".join(operation_path),
                                api_key=api_key,
                                attempt=attempt,
                                max_attempts=self._limiter.retry_max_attempts,
                                exc=exc,
                                retry_delay_seconds=None,
                                reason="non_retryable_error",
                            )
                            raise

                        if _is_daily_key_limit_error(exc):
                            has_next_key = self._limiter.mark_key_daily_exhausted(api_key)
                            self._limiter.record_model_error(
                                event="retry",
                                operation=".".join(operation_path),
                                api_key=api_key,
                                attempt=attempt,
                                max_attempts=self._limiter.retry_max_attempts,
                                exc=exc,
                                retry_delay_seconds=0.0,
                                reason="daily_key_limit_exhausted",
                            )
                            if has_next_key:
                                print(
                                    "[openrouter retry] Current key reached today's free-model limit; "
                                    "switching to the next available key."
                                )
                            else:
                                print(
                                    "[openrouter retry] Current key reached today's free-model limit; "
                                    "all keys are exhausted for this 24h window."
                                )
                        else:
                            retry_count = self._limiter.record_retry_for_key(api_key)
                            reason = (
                                "temporary_upstream_rate_limit"
                                if _is_temporary_rate_limit_error(exc)
                                else "http_429_temporary_rate_limit"
                            )
                            self._limiter.record_model_error(
                                event="retry",
                                operation=".".join(operation_path),
                                api_key=api_key,
                                attempt=attempt,
                                max_attempts=self._limiter.retry_max_attempts,
                                exc=exc,
                                retry_delay_seconds=retry_delay,
                                reason=f"{reason}_retry_count_{retry_count}",
                            )
                            print(
                                "[openrouter retry] Model request returned a temporary 429. "
                                f"Waiting {retry_delay:.1f}s before retrying the same key."
                            )
                            time.sleep(retry_delay)
                        attempt += 1

            return wrapped

        if isinstance(attr, (str, int, float, bool, bytes, type(None))):
            return attr

        return RateLimitedClientProxy(attr, self._limiter, operation_path, self._root_target)

    def _apply_api_key(self, api_key: str) -> None:
        candidates = [
            self._target,
            self._root_target,
            getattr(self._target, "_client", None),
            getattr(self._root_target, "_client", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if hasattr(candidate, "api_key"):
                setattr(candidate, "api_key", api_key)
