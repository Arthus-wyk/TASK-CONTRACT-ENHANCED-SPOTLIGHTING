from __future__ import annotations

import json
import os
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


def _daily_action_from_env(default: DailyLimitAction = "wait") -> DailyLimitAction:
    raw_value = os.getenv("OPENROUTER_DAILY_LIMIT_ACTION", default).strip().lower()
    if raw_value not in {"pause", "wait"}:
        raise ValueError("OPENROUTER_DAILY_LIMIT_ACTION must be 'pause' or 'wait'")
    return raw_value  # type: ignore[return-value]


def _should_rate_limit_operation(operation_path: tuple[str, ...]) -> bool:
    return any(
        operation_path[-len(suffix) :] == suffix
        for suffix in RATE_LIMITED_OPERATION_SUFFIXES
    )


@dataclass
class OpenRouterRateLimiter:
    state_path: Path
    requests_per_minute: int = 20
    daily_request_limit: int = 50
    daily_limit_action: DailyLimitAction = "wait"
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    key_check_interval: int = 10
    utc_reset_buffer_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.state_path = Path(self.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._requests_since_key_check = self.key_check_interval

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
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            key_check_interval=_positive_int_from_env("OPENROUTER_KEY_CHECK_INTERVAL", 10),
        )

    def acquire(self) -> None:
        while True:
            self._check_openrouter_key_if_needed()

            with self._lock:
                state = self._normalized_state(self._load_state())
                now = time.time()
                state["recent_request_timestamps"] = [
                    timestamp
                    for timestamp in state["recent_request_timestamps"]
                    if now - timestamp < 60.0
                ]

                if state["daily_request_count"] >= self.daily_request_limit:
                    wait_seconds = _seconds_until_next_utc_day(self.utc_reset_buffer_seconds)
                    self._save_state(state)
                    message = (
                        "OpenRouter daily request limit reached "
                        f"({self.daily_request_limit}/{self.daily_request_limit}). "
                        f"UTC day resets in {wait_seconds:.0f} seconds."
                    )
                    if self.daily_limit_action == "pause":
                        raise OpenRouterDailyLimitReached(message, wait_seconds)

                    print(f"[openrouter rate limit] {message} Waiting before continuing.")
                elif len(state["recent_request_timestamps"]) < self.requests_per_minute:
                    state["recent_request_timestamps"].append(now)
                    state["daily_request_count"] += 1
                    self._requests_since_key_check += 1
                    self._save_state(state)
                    return
                else:
                    oldest = min(state["recent_request_timestamps"])
                    wait_seconds = max(0.1, oldest + 60.0 - now + 0.25)
                    self._save_state(state)
                    print(
                        "[openrouter rate limit] Per-minute request limit reached "
                        f"({self.requests_per_minute}/minute). Waiting {wait_seconds:.1f}s."
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
                "daily_request_count": 0,
                "recent_request_timestamps": state.get("recent_request_timestamps", []),
            }

        daily_request_count = state.get("daily_request_count", 0)
        if not isinstance(daily_request_count, int) or daily_request_count < 0:
            daily_request_count = 0

        recent_timestamps = state.get("recent_request_timestamps", [])
        if not isinstance(recent_timestamps, list):
            recent_timestamps = []
        recent_timestamps = [
            float(timestamp)
            for timestamp in recent_timestamps
            if isinstance(timestamp, (int, float))
        ]

        return {
            "date_utc": today,
            "daily_request_count": daily_request_count,
            "recent_request_timestamps": recent_timestamps,
            "requests_per_minute": self.requests_per_minute,
            "daily_request_limit": self.daily_request_limit,
            "daily_limit_action": self.daily_limit_action,
            "updated_at": _utc_now().isoformat(),
        }

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _utc_now().isoformat()
        with self.state_path.open("w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)

    def _check_openrouter_key_if_needed(self) -> None:
        if not self.api_key:
            return
        if self._requests_since_key_check < self.key_check_interval:
            return

        self._requests_since_key_check = 0
        try:
            key_data = self._fetch_key_data()
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"[openrouter rate limit] Key status check failed; using local limits. {exc!r}")
            return

        data = key_data.get("data")
        if not isinstance(data, dict):
            return

        limit_remaining = data.get("limit_remaining")
        if limit_remaining is None:
            return
        if not isinstance(limit_remaining, (int, float)):
            return
        if limit_remaining > 0:
            return

        reset_kind = data.get("limit_reset")
        if reset_kind == "daily" and self.daily_limit_action == "wait":
            wait_seconds = _seconds_until_next_utc_day(self.utc_reset_buffer_seconds)
            print(
                "[openrouter rate limit] OpenRouter credit limit is exhausted for the "
                f"current UTC day. Waiting {wait_seconds:.0f}s before continuing."
            )
            time.sleep(wait_seconds)
            return

        raise OpenRouterCreditLimitReached(
            "OpenRouter key has no remaining credits according to /api/v1/key. "
            "Add credits, switch keys, or wait for the configured credit reset."
        )

    def _fetch_key_data(self) -> dict[str, Any]:
        key_url = f"{self.base_url.rstrip('/')}/key"
        request = urllib.request.Request(
            key_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
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
                self._limiter.acquire()
                return attr(*args, **kwargs)

            return wrapped

        if isinstance(attr, (str, int, float, bool, bytes, type(None))):
            return attr

        return RateLimitedClientProxy(attr, self._limiter, operation_path)
