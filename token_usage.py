from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    ) -> None:
        usage = _extract_usage(response)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "operation": operation,
            "duration_seconds": duration_seconds,
            "usage": usage,
            "error": repr(error) if error else None,
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
                try:
                    response = attr(*args, **kwargs)
                except Exception as exc:
                    self._logger.record(
                        operation=".".join(operation_path),
                        error=exc,
                        duration_seconds=time.perf_counter() - started_at,
                    )
                    raise

                self._logger.record(
                    operation=".".join(operation_path),
                    response=response,
                    duration_seconds=time.perf_counter() - started_at,
                )
                return response

            return wrapped

        if isinstance(attr, (str, int, float, bool, bytes, type(None))):
            return attr

        return UsageLoggingProxy(attr, self._logger, operation_path)
