from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Bucket:
    utility_total: int = 0
    utility_success: int = 0
    security_total: int = 0
    security_success: int = 0
    duration_total: float = 0.0

    def add(self, result: dict[str, Any]) -> None:
        utility = result.get("utility")
        if isinstance(utility, bool):
            self.utility_total += 1
            self.utility_success += int(utility)

        security = result.get("security")
        if isinstance(security, bool):
            self.security_total += 1
            self.security_success += int(security)

        duration = result.get("duration")
        if isinstance(duration, (int, float)):
            self.duration_total += float(duration)

    @property
    def attack_success_rate(self) -> float | None:
        security_rate = self.rate("security")
        return None if security_rate is None else 1.0 - security_rate

    def rate(self, metric: str) -> float | None:
        if metric == "utility":
            if self.utility_total == 0:
                return None
            return self.utility_success / self.utility_total
        if metric == "security":
            if self.security_total == 0:
                return None
            return self.security_success / self.security_total
        raise ValueError(f"unknown metric: {metric}")


def iter_result_files(logdir: Path):
    for path in logdir.rglob("*.json"):
        if path.name == "token_usage_summary.json":
            continue

        try:
            with path.open("r", encoding="utf-8") as file:
                result = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        if "suite_name" not in result or "pipeline_name" not in result:
            continue
        if "utility" not in result and "security" not in result:
            continue

        yield path, result


def is_attack_result(result: dict[str, Any]) -> bool:
    return result.get("injection_task_id") is not None or result.get("attack_type") is not None


def format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize existing AgentDojo JSON logs without rerunning the model."
    )
    parser.add_argument("--logdir", type=Path, default=Path("logs_my_agentdojo"))
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also include no-attack baseline logs under */none/*.json.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        help="Only include this suite. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pipeline",
        help="Only include this pipeline_name.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    args = parser.parse_args()

    suites = set(args.suite or [])
    by_suite: dict[str, Bucket] = defaultdict(Bucket)
    overall = Bucket()

    for _path, result in iter_result_files(args.logdir):
        if suites and result.get("suite_name") not in suites:
            continue
        if args.pipeline and result.get("pipeline_name") != args.pipeline:
            continue
        if not args.include_baseline and not is_attack_result(result):
            continue

        suite_name = str(result["suite_name"])
        by_suite[suite_name].add(result)
        overall.add(result)

    if args.json:
        payload = {
            "suites": {
                suite: {
                    "utility_rate": bucket.rate("utility"),
                    "utility_cases": bucket.utility_total,
                    "security_rate": bucket.rate("security"),
                    "attack_success_rate": bucket.attack_success_rate,
                    "security_cases": bucket.security_total,
                    "duration_seconds": bucket.duration_total,
                }
                for suite, bucket in sorted(by_suite.items())
            },
            "overall": {
                "utility_rate": overall.rate("utility"),
                "utility_cases": overall.utility_total,
                "security_rate": overall.rate("security"),
                "attack_success_rate": overall.attack_success_rate,
                "security_cases": overall.security_total,
                "duration_seconds": overall.duration_total,
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    for suite, bucket in sorted(by_suite.items()):
        print(f"{suite} - utility: {format_rate(bucket.rate('utility'))} ({bucket.utility_success}/{bucket.utility_total})")
        print(f"{suite} - security: {format_rate(bucket.rate('security'))} ({bucket.security_success}/{bucket.security_total})")
        print(f"{suite} - attack_success_rate: {format_rate(bucket.attack_success_rate)}")

    print("=" * 80)
    print("Overall")
    print("=" * 80)
    print(f"overall - utility: {format_rate(overall.rate('utility'))} ({overall.utility_success}/{overall.utility_total})")
    print(f"overall - security: {format_rate(overall.rate('security'))} ({overall.security_success}/{overall.security_total})")
    print(f"overall - attack_success_rate: {format_rate(overall.attack_success_rate)}")


if __name__ == "__main__":
    main()
