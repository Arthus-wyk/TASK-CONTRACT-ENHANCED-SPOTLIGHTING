from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wait_tracker import result_wall_interval, thread_wait_total_seconds


LOGGED_OPERATION_SUFFIXES = (
    ("chat", "completions", "create"),
    ("responses", "create"),
    ("messages", "create"),
    ("models", "generate_content"),
)


def _to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_plain(model_dump())

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_plain(to_dict())

    if hasattr(value, "__dict__"):
        return {
            key: _to_plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return str(value)


def _extract_usage(response: Any) -> dict[str, Any]:
    usage = None
    if isinstance(response, dict):
        usage = response.get("usage") or response.get("usage_metadata")
    else:
        usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)

    if usage is None:
        return {}

    plain_usage = _to_plain(usage)
    return plain_usage if isinstance(plain_usage, dict) else {"raw_usage": plain_usage}


def _numeric_usage_values(usage: dict[str, Any]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            totals[key] = value
    return totals


@dataclass(frozen=True)
class TokenUsageEvent:
    start_wall: datetime
    end_wall: datetime
    provider: str
    model: str
    operation: str
    duration_seconds: float | None
    wait_duration_excluded_seconds: float | None
    usage: dict[str, Any]
    error: str | None


_events: list[TokenUsageEvent] = []
_events_lock = threading.Lock()


def reset_token_usage_events() -> None:
    with _events_lock:
        _events.clear()


def _event_overlaps_interval(event: TokenUsageEvent, start: datetime, end: datetime) -> bool:
    return event.start_wall <= end and event.end_wall >= start


def _add_numeric_usage_totals(target: dict[str, int | float], usage: dict[str, Any]) -> None:
    for key, value in _numeric_usage_values(usage).items():
        target[key] = target.get(key, 0) + value


def _token_usage_summary_for_interval(start: datetime, end: datetime) -> dict[str, Any]:
    with _events_lock:
        events = [event for event in _events if _event_overlaps_interval(event, start, end)]

    usage_totals: dict[str, int | float] = {}
    duration_total = 0.0
    wait_duration_total = 0.0
    provider_models: list[dict[str, str]] = []
    seen_provider_models: set[tuple[str, str]] = set()
    error_count = 0

    for event in events:
        _add_numeric_usage_totals(usage_totals, event.usage)
        if isinstance(event.duration_seconds, (int, float)):
            duration_total += float(event.duration_seconds)
        if isinstance(event.wait_duration_excluded_seconds, (int, float)):
            wait_duration_total += float(event.wait_duration_excluded_seconds)
        if event.error is not None:
            error_count += 1
        provider_model = (event.provider, event.model)
        if provider_model not in seen_provider_models:
            seen_provider_models.add(provider_model)
            provider_models.append({"provider": event.provider, "model": event.model})

    return {
        "call_count": len(events),
        "error_count": error_count,
        "duration_seconds": duration_total,
        "wait_duration_excluded_seconds": wait_duration_total,
        "usage_totals": usage_totals,
        "provider_models": provider_models,
    }


def apply_token_usage_to_result(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as file:
            result: dict[str, Any] = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    interval = result_wall_interval(path, result)
    if interval is None:
        return False

    start, end, _duration_seconds = interval
    token_usage = _token_usage_summary_for_interval(start, end)
    if token_usage["call_count"] == 0:
        return False

    try:
        original_stat = path.stat()
    except OSError:
        original_stat = None

    result["token_usage"] = token_usage
    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    if original_stat is not None:
        os.utime(path, (original_stat.st_atime, original_stat.st_mtime))

    return True


def _should_log_operation(operation_path: tuple[str, ...]) -> bool:
    return any(operation_path[-len(suffix) :] == suffix for suffix in LOGGED_OPERATION_SUFFIXES)


class TokenUsageLogger:
    def __init__(self, log_dir: Path, provider: str, model: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.model = model
        self.run_id = str(uuid.uuid4())
        self.records_path = self.log_dir / "token_usage.jsonl"
        self.summary_path = self.log_dir / "token_usage_summary.json"
        self._lock = threading.Lock()
        self._summary: dict[str, Any] = {
            "run_id": self.run_id,
            "provider": provider,
            "model": model,
            "call_count": 0,
            "error_count": 0,
            "usage_totals": {},
            "scope": "current_run",
            "records_path": str(self.records_path),
        }
        self.records_path.touch(exist_ok=True)
        self._write_summary()

    def _write_summary(self) -> None:
        with self.summary_path.open("w", encoding="utf-8") as file:
            json.dump(self._summary, file, ensure_ascii=False, indent=2)

    def record(
        self,
        operation: str,
        response: Any = None,
        error: BaseException | None = None,
        duration_seconds: float | None = None,
        wait_duration_excluded_seconds: float | None = None,
        start_wall: datetime | None = None,
        end_wall: datetime | None = None,
    ) -> None:
        usage = _extract_usage(response)
        error_text = repr(error) if error else None
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "operation": operation,
            "duration_seconds": duration_seconds,
            "wait_duration_excluded_seconds": wait_duration_excluded_seconds,
            "usage": usage,
            "error": error_text,
        }

        with self._lock:
            with self.records_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")

            self._summary["call_count"] += 1
            if error:
                self._summary["error_count"] += 1

            usage_totals = self._summary["usage_totals"]
            for key, value in _numeric_usage_values(usage).items():
                usage_totals[key] = usage_totals.get(key, 0) + value

            self._write_summary()

        if start_wall is not None and end_wall is not None:
            with _events_lock:
                _events.append(
                    TokenUsageEvent(
                        start_wall=start_wall,
                        end_wall=end_wall,
                        provider=self.provider,
                        model=self.model,
                        operation=operation,
                        duration_seconds=duration_seconds,
                        wait_duration_excluded_seconds=wait_duration_excluded_seconds,
                        usage=usage,
                        error=error_text,
                    )
                )


class UsageLoggingProxy:
    def __init__(
        self,
        target: Any,
        logger: TokenUsageLogger,
        operation_path: tuple[str, ...] = (),
    ):
        self._target = target
        self._logger = logger
        self._operation_path = operation_path

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        operation_path = self._operation_path + (name,)

        if callable(attr):
            if not _should_log_operation(operation_path):
                return attr

            def wrapped(*args, **kwargs):
                started_at = time.perf_counter()
                start_wall = datetime.now()
                started_wait_seconds = thread_wait_total_seconds()
                try:
                    response = attr(*args, **kwargs)
                except Exception as exc:
                    end_wall = datetime.now()
                    elapsed_seconds = time.perf_counter() - started_at
                    wait_seconds = max(0.0, thread_wait_total_seconds() - started_wait_seconds)
                    self._logger.record(
                        operation=".".join(operation_path),
                        error=exc,
                        duration_seconds=max(0.0, elapsed_seconds - wait_seconds),
                        wait_duration_excluded_seconds=wait_seconds,
                        start_wall=start_wall,
                        end_wall=end_wall,
                    )
                    raise

                end_wall = datetime.now()
                elapsed_seconds = time.perf_counter() - started_at
                wait_seconds = max(0.0, thread_wait_total_seconds() - started_wait_seconds)
                self._logger.record(
                    operation=".".join(operation_path),
                    response=response,
                    duration_seconds=max(0.0, elapsed_seconds - wait_seconds),
                    wait_duration_excluded_seconds=wait_seconds,
                    start_wall=start_wall,
                    end_wall=end_wall,
                )
                return response

            return wrapped

        if isinstance(attr, (str, int, float, bool, bytes, type(None))):
            return attr

        return UsageLoggingProxy(attr, self._logger, operation_path)
