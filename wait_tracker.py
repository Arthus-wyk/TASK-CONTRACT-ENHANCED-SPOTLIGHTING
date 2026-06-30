from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WaitEvent:
    start_wall: datetime
    end_wall: datetime
    duration_seconds: float
    reason: str


_events: list[WaitEvent] = []
_events_lock = threading.Lock()
_thread_state = threading.local()


def _thread_total() -> float:
    return float(getattr(_thread_state, "wait_seconds", 0.0))


def thread_wait_total_seconds() -> float:
    return _thread_total()


def reset_wait_events() -> None:
    with _events_lock:
        _events.clear()


def tracked_sleep(seconds: float, reason: str) -> None:
    if seconds <= 0:
        return

    start_wall = datetime.now()
    started_at = time.perf_counter()
    try:
        time.sleep(seconds)
    finally:
        duration_seconds = max(0.0, time.perf_counter() - started_at)
        end_wall = datetime.now()
        _thread_state.wait_seconds = _thread_total() + duration_seconds
        with _events_lock:
            _events.append(
                WaitEvent(
                    start_wall=start_wall,
                    end_wall=end_wall,
                    duration_seconds=duration_seconds,
                    reason=reason,
                )
            )


def _parse_evaluation_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None

    raw_value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw_value[:19], fmt)
        except ValueError:
            pass

    try:
        parsed = datetime.fromisoformat(raw_value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=None)


def _overlap_seconds(start: datetime, end: datetime, event: WaitEvent) -> float:
    overlap_start = max(start, event.start_wall)
    overlap_end = min(end, event.end_wall)
    return max(0.0, (overlap_end - overlap_start).total_seconds())


def _wait_seconds_in_interval(start: datetime, duration_seconds: float) -> float:
    end = start + timedelta(seconds=duration_seconds)
    with _events_lock:
        wait_seconds = sum(_overlap_seconds(start, end, event) for event in _events)
    return min(duration_seconds, wait_seconds)


def result_wall_interval(path: Path, result: dict[str, Any]) -> tuple[datetime, datetime, float] | None:
    duration = result.get("duration_including_wait_seconds", result.get("duration"))
    if not isinstance(duration, (int, float)):
        return None

    duration_seconds = float(duration)
    try:
        end = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        end = None

    if end is not None:
        return end - timedelta(seconds=duration_seconds), end, duration_seconds

    start = _parse_evaluation_timestamp(result.get("evaluation_timestamp"))
    if start is None:
        return None
    return start, start + timedelta(seconds=duration_seconds), duration_seconds


def apply_duration_wait_correction(path: Path) -> tuple[float, float] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            result: dict[str, Any] = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    if result.get("duration_excludes_wait") is True:
        return None

    interval = result_wall_interval(path, result)
    if interval is None:
        return None

    start, _end, duration_seconds = interval
    wait_seconds = _wait_seconds_in_interval(start, duration_seconds)
    if wait_seconds <= 1e-6:
        wait_seconds = 0.0

    try:
        original_stat = path.stat()
    except OSError:
        original_stat = None

    adjusted_duration = max(0.0, duration_seconds - wait_seconds)
    result["duration_including_wait_seconds"] = duration_seconds
    result["wait_duration_excluded_seconds"] = wait_seconds
    result["duration"] = adjusted_duration
    result["duration_excludes_wait"] = True

    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    if original_stat is not None:
        os.utime(path, (original_stat.st_atime, original_stat.st_mtime))

    return duration_seconds, wait_seconds
