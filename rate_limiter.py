from __future__ import annotations

import json
import hashlib
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal


DailyLimitAction = Literal["pause", "wait"]

RATE_LIMITED_OPERATION_SUFFIXES = (
    ("chat", "completions", "create"),
    ("responses", "create"),
)


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


def _seconds_until_next_utc_day(buffer_seconds: float = 60.0) -> float:
    now = _utc_now()
    tomorrow = now.date() + timedelta(days=1)
    midnight = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
    return max(0.0, (midnight - now).total_seconds() + buffer_seconds)


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

    retry_after = _response_retry_after_seconds(getattr(exc, "response", None))
    if retry_after is None:
        retry_after = _find_retry_after_seconds(_exception_payload(exc))
    if retry_after is None:
        retry_after = base_seconds

    return min(max_seconds, max(base_seconds, retry_after + 0.5))


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
    key_check_interval: int = 10
    utc_reset_buffer_seconds: float = 60.0
    retry_max_attempts: int = 10
    retry_base_seconds: float = 60.0
    retry_max_seconds: float = 300.0
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
        self._requests_since_key_check = {
            _key_id(api_key): self.key_check_interval
            for api_key in self.api_keys
        }

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
            key_check_interval=_positive_int_from_env("OPENROUTER_KEY_CHECK_INTERVAL", 10),
            retry_max_attempts=_positive_int_from_env("OPENROUTER_RETRY_MAX_ATTEMPTS", 10),
            retry_base_seconds=_positive_float_from_env("OPENROUTER_RETRY_BASE_SECONDS", 60.0),
            retry_max_seconds=_positive_float_from_env("OPENROUTER_RETRY_MAX_SECONDS", 300.0),
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
                minute_wait_seconds: list[float] = []

                for offset in range(len(self.api_keys)):
                    key_index = (active_key_index + offset) % len(self.api_keys)
                    api_key = self.api_keys[key_index]
                    key_id = _key_id(api_key)
                    key_state = state["keys"][key_id]
                    key_state["recent_request_timestamps"] = [
                        timestamp
                        for timestamp in key_state["recent_request_timestamps"]
                        if now - timestamp < 60.0
                    ]

                    if key_state.get("credit_exhausted_date_utc") == state["date_utc"]:
                        continue

                    if key_state["daily_request_count"] >= self.daily_request_limit:
                        continue

                    if len(key_state["recent_request_timestamps"]) >= self.requests_per_minute:
                        oldest = min(key_state["recent_request_timestamps"])
                        minute_wait_seconds.append(max(0.1, oldest + 60.0 - now + 0.25))
                        continue

                    if not self._check_openrouter_key_if_needed(api_key, state, key_state):
                        continue

                    key_state["recent_request_timestamps"].append(now)
                    key_state["daily_request_count"] += 1
                    self._requests_since_key_check[key_id] = self._requests_since_key_check.get(key_id, 0) + 1
                    state["active_key_index"] = key_index
                    state["current_key"] = _key_label(api_key)
                    self._save_state(state)
                    return api_key

                self._save_state(state)
                all_keys_daily_exhausted = all(
                    state["keys"][_key_id(api_key)]["daily_request_count"] >= self.daily_request_limit
                    or state["keys"][_key_id(api_key)].get("credit_exhausted_date_utc") == state["date_utc"]
                    for api_key in self.api_keys
                )

                if all_keys_daily_exhausted:
                    wait_seconds = _seconds_until_next_utc_day(self.utc_reset_buffer_seconds)
                    message = (
                        "All OpenRouter keys reached their daily request or credit limit "
                        f"({len(self.api_keys)} keys, {self.daily_request_limit} requests/key/day). "
                        f"UTC day resets in {wait_seconds:.0f} seconds."
                    )
                    if self.daily_limit_action == "pause":
                        raise OpenRouterDailyLimitReached(message, wait_seconds)

                    print(f"[openrouter rate limit] {message} Waiting before continuing.")
                else:
                    wait_seconds = min(minute_wait_seconds) if minute_wait_seconds else 1.0
                    print(
                        "[openrouter rate limit] All available OpenRouter keys are at their "
                        f"per-minute limit ({self.requests_per_minute}/minute/key). "
                        f"Waiting {wait_seconds:.1f}s."
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
        today = _utc_date_string()
        state_date = state.get("date_utc")
        if state_date != today:
            state = {
                "date_utc": today,
                "active_key_index": int(state.get("active_key_index", 0) or 0),
                "keys": {},
            }

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
                "recent_request_timestamps": recent_timestamps,
            }
            exhausted_date = key_state.get("credit_exhausted_date_utc")
            if isinstance(exhausted_date, str) and exhausted_date == today:
                normalized_key_state["credit_exhausted_date_utc"] = exhausted_date
            normalized_keys[key_id] = normalized_key_state

        active_key_index = state.get("active_key_index", 0)
        if not isinstance(active_key_index, int) or active_key_index < 0:
            active_key_index = 0
        active_key_index %= len(self.api_keys)

        return {
            "date_utc": today,
            "active_key_index": active_key_index,
            "key_count": len(self.api_keys),
            "keys": normalized_keys,
            "requests_per_minute": self.requests_per_minute,
            "daily_request_limit": self.daily_request_limit,
            "daily_limit_action": self.daily_limit_action,
            "updated_at": _utc_now().isoformat(),
        }

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

    def _check_openrouter_key_if_needed(
        self,
        api_key: str,
        state: dict[str, Any],
        key_state: dict[str, Any],
    ) -> bool:
        key_id = _key_id(api_key)
        if self._requests_since_key_check.get(key_id, self.key_check_interval) < self.key_check_interval:
            return True

        self._requests_since_key_check[key_id] = 0
        try:
            key_data = self._fetch_key_data(api_key)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            print(
                "[openrouter rate limit] Key status check failed; using local limits "
                f"for key {_key_label(api_key)}. {exc!r}"
            )
            return True

        data = key_data.get("data")
        if not isinstance(data, dict):
            return True

        limit_remaining = data.get("limit_remaining")
        if limit_remaining is None:
            return True
        if not isinstance(limit_remaining, (int, float)):
            return True
        if limit_remaining > 0:
            return True

        key_state["credit_exhausted_date_utc"] = state["date_utc"]
        print(
            "[openrouter rate limit] OpenRouter key has no remaining credits; "
            f"switching away from key {_key_label(api_key)}."
        )
        return False

    def _fetch_key_data(self, api_key: str) -> dict[str, Any]:
        key_url = f"{self.base_url.rstrip('/')}/key"
        request = urllib.request.Request(
            key_url,
            headers={"Authorization": f"Bearer {api_key}"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8")
        loaded = json.loads(payload)
        return loaded if isinstance(loaded, dict) else {}


class RateLimitedClientProxy:
    def __init__(
        self,
        target: Any,
        limiter: OpenRouterRateLimiter,
        operation_path: tuple[str, ...] = (),
    ):
        self._target = target
        self._limiter = limiter
        self._operation_path = operation_path

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
                        if retry_delay is None or attempt >= self._limiter.retry_max_attempts:
                            self._limiter.record_model_error(
                                event="crash",
                                operation=".".join(operation_path),
                                api_key=api_key,
                                attempt=attempt,
                                max_attempts=self._limiter.retry_max_attempts,
                                exc=exc,
                                retry_delay_seconds=None,
                                reason="non_retryable_error"
                                if retry_delay is None
                                else "retry_attempts_exhausted",
                            )
                            raise

                        self._limiter.record_model_error(
                            event="retry",
                            operation=".".join(operation_path),
                            api_key=api_key,
                            attempt=attempt,
                            max_attempts=self._limiter.retry_max_attempts,
                            exc=exc,
                            retry_delay_seconds=retry_delay,
                            reason="http_429",
                        )
                        print(
                            "[openrouter retry] Model request returned 429. "
                            f"Waiting {retry_delay:.1f}s before retry "
                            f"{attempt + 1}/{self._limiter.retry_max_attempts}."
                        )
                        time.sleep(retry_delay)
                        attempt += 1

            return wrapped

        if isinstance(attr, (str, int, float, bool, bytes, type(None))):
            return attr

        return RateLimitedClientProxy(attr, self._limiter, operation_path)

    def _apply_api_key(self, api_key: str) -> None:
        candidates = [self._target, getattr(self._target, "_client", None)]
        for candidate in candidates:
            if candidate is None:
                continue
            if hasattr(candidate, "api_key"):
                setattr(candidate, "api_key", api_key)
