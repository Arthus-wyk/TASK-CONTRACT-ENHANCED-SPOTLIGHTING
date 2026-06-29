from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agentdojo import attacks, benchmark, logging
from agentdojo.task_suite import get_suite

from pipeline_builder import DefenseMode, make_my_agent_pipeline
from rate_limiter import DailyLimitAction, OpenRouterRateLimitError


def patch_agentdojo_pydantic_models() -> None:
    """
    AgentDojo 0.1.x can leave TaskResults with an unresolved FunctionCall
    forward reference under newer Pydantic versions. Resume mode loads existing
    JSON results through TaskResults, so rebuild the model before any loading.
    """
    from agentdojo.functions_runtime import FunctionCall

    FunctionCall.model_rebuild()
    benchmark.TaskResults.model_rebuild(_types_namespace={"FunctionCall": FunctionCall})


def task_result_path(
    logdir: Path,
    pipeline_name: str,
    suite_name: str,
    user_task_id: str,
    attack_name: str,
    injection_task_id: str,
) -> Path:
    pipeline_name = pipeline_name.replace("/", "_")
    return logdir / pipeline_name / suite_name / user_task_id / attack_name / f"{injection_task_id}.json"


def is_complete_task_result(
    path: Path,
    suite_name: str,
    pipeline_name: str,
    user_task_id: str,
    attack_name: str,
    injection_task_id: str,
) -> bool:
    if not path.exists():
        return False

    try:
        with path.open("r", encoding="utf-8") as file:
            result: dict[str, Any] = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    expected_attack_type = None if attack_name == "none" else attack_name
    expected_injection_task_id = None if injection_task_id == "none" else injection_task_id
    if result.get("suite_name") != suite_name:
        return False
    if result.get("pipeline_name") != pipeline_name.replace("/", "_"):
        return False
    if result.get("user_task_id") != user_task_id:
        return False
    if result.get("attack_type") != expected_attack_type:
        return False
    if result.get("injection_task_id") != expected_injection_task_id:
        return False
    if not isinstance(result.get("utility"), bool):
        return False
    if not isinstance(result.get("security"), bool):
        return False
    if not isinstance(result.get("duration"), (int, float)):
        return False

    messages = result.get("messages")
    return isinstance(messages, list) and len(messages) >= 2


def quarantine_partial_result(path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    destination = path.with_name(f"{path.stem}.partial-{timestamp}{path.suffix}.bak")
    counter = 1
    while destination.exists():
        destination = path.with_name(f"{path.stem}.partial-{timestamp}-{counter}{path.suffix}.bak")
        counter += 1
    path.replace(destination)
    return destination


def prepare_resume_files(
    logdir: Path,
    pipeline_name: str,
    suite_name: str,
    expected_results: list[tuple[str, str, str]],
) -> None:
    completed = 0
    pending = 0
    quarantined: list[tuple[Path, Path]] = []

    for user_task_id, attack_name, injection_task_id in expected_results:
        path = task_result_path(logdir, pipeline_name, suite_name, user_task_id, attack_name, injection_task_id)
        if is_complete_task_result(path, suite_name, pipeline_name, user_task_id, attack_name, injection_task_id):
            completed += 1
            continue

        pending += 1
        if path.exists():
            quarantined.append((path, quarantine_partial_result(path)))

    print(
        "Resume scan: "
        f"{completed} completed result(s), {pending} pending result(s), "
        f"{len(quarantined)} partial/corrupt result file(s) quarantined."
    )
    for original, archived in quarantined:
        print(f"Quarantined partial result: {original} -> {archived}")


def selected_task_ids(all_task_ids, selected_task_ids: list[str] | None) -> list[str]:
    if selected_task_ids is None:
        return list(all_task_ids)
    return list(selected_task_ids)


def expected_results_with_injections(suite, attack, user_tasks: list[str] | None, injection_tasks: list[str] | None):
    expected: list[tuple[str, str, str]] = []
    selected_user_tasks = selected_task_ids(suite.user_tasks.keys(), user_tasks)

    if attack.is_dos_attack:
        selected_injection_tasks = [next(iter(suite.injection_tasks.keys()))]
    else:
        selected_injection_tasks = selected_task_ids(suite.injection_tasks.keys(), injection_tasks)
        for injection_task_id in selected_injection_tasks:
            expected.append((injection_task_id, "none", "none"))

    for user_task_id in selected_user_tasks:
        for injection_task_id in selected_injection_tasks:
            expected.append((user_task_id, attack.name, injection_task_id))

    return expected


def expected_results_without_injections(suite, user_tasks: list[str] | None):
    return [(user_task_id, "none", "none") for user_task_id in selected_task_ids(suite.user_tasks.keys(), user_tasks)]


def print_suite_tools(suite_name: str, suite) -> None:
    """
    Best-effort dump of tools exposed by an AgentDojo suite.

    Different AgentDojo versions expose tool metadata on slightly different
    attributes, so we try a few common shapes and fall back to a light
    introspection dump if needed.
    """
    print(f"\n[{suite_name}] available tools")

    candidate_attrs = (
        "tools",
        "functions",
        "tool_names",
        "available_tools",
    )
    for attr in candidate_attrs:
        value = getattr(suite, attr, None)
        if not value:
            continue

        print(f"- found suite.{attr}:")
        if isinstance(value, dict):
            for name in sorted(value.keys()):
                print(f"  - {name}")
            return

        if isinstance(value, (list, tuple, set)):
            for item in value:
                name = getattr(item, "name", None) or getattr(item, "__name__", None) or str(item)
                print(f"  - {name}")
            return

        name = getattr(value, "name", None) or str(value)
        print(f"  - {name}")
        return

    attrs = [a for a in dir(suite) if not a.startswith("_")]
    print("- no obvious tool container found; suite attributes:")
    for attr in attrs:
        if "tool" in attr.lower() or "function" in attr.lower() or "env" in attr.lower():
            print(f"  - {attr}")


def summarize_results(suite_name: str, results: dict, run_attack: bool):
    utility_values = list(results["utility_results"].values())
    utility_rate = sum(utility_values) / len(utility_values)
    print(f"{suite_name} - utility: {utility_rate:.4f}")

    summary = {
        "suite": suite_name,
        "utility_rate": utility_rate,
        "num_utility_cases": len(utility_values),
    }

    if run_attack:
        security_values = list(results["security_results"].values())
        security_rate = sum(security_values) / len(security_values)
        attack_success_rate = 1.0 - security_rate
        print(f"{suite_name} - security: {security_rate:.4f}")
        print(f"{suite_name} - attack_success_rate: {attack_success_rate:.4f}")
        summary.update(
            {
                "security_rate": security_rate,
                "attack_success_rate": attack_success_rate,
                "num_security_cases": len(security_values),
            }
        )

    return summary


def main(
    model: str = "ollama:qwen2.5:7b",
    suites: list[str] | None = None,
    benchmark_version: str = "v1.2",
    attack_name: str = "important_instructions",
    run_attack: bool = True,
    user_tasks: list[str] | None = None,
    injection_tasks: list[str] | None = None,
    max_injection_tasks: int | None = None,
    reasoning_effort="medium",
    thinking_budget_tokens: int | None = None,
    logdir: Path = Path("./logs_my_agentdojo"),
    record_token_usage: bool = True,
    force_rerun: bool = False,
    daily_limit_action: DailyLimitAction = "wait",
    defense: DefenseMode = "spotlighting",
):
    suites = suites or ["workspace", "banking", "travel", "slack"]
    total_utility_results: list[bool] = []
    total_security_results: list[bool] = []
    logdir.mkdir(parents=True, exist_ok=True)
    patch_agentdojo_pydantic_models()

    for suite_name in suites:
        print("=" * 80)
        print(f"Running suite: {suite_name}")
        print(f"Model: {model}")
        print(f"Attack: {attack_name if run_attack else 'none'}")
        print(f"Defense: {defense}")
        print("=" * 80)

        suite = get_suite(benchmark_version, suite_name)
        print_suite_tools(suite_name, suite)
        selected_injection_tasks = injection_tasks
        if max_injection_tasks is not None:
            selected_injection_tasks = list(suite.injection_tasks.keys())[:max_injection_tasks]

        suite_logdir = logdir / suite_name
        tools_pipeline = make_my_agent_pipeline(
            model=model,
            suite=suite_name,
            reasoning_effort=reasoning_effort,
            thinking_budget_tokens=thinking_budget_tokens,
            token_log_dir=suite_logdir if record_token_usage else None,
            rate_limit_state_path=logdir / "openrouter_rate_limit_state.json",
            daily_limit_action=daily_limit_action,
            defense=defense,
        )

        try:
            attack = None
            if run_attack:
                attack = attacks.load_attack(attack_name, suite, tools_pipeline)
                expected_results = expected_results_with_injections(
                    suite,
                    attack,
                    user_tasks=user_tasks,
                    injection_tasks=selected_injection_tasks,
                )
            else:
                expected_results = expected_results_without_injections(suite, user_tasks=user_tasks)

            prepare_resume_files(
                suite_logdir,
                tools_pipeline.name,
                suite.name,
                expected_results,
            )

            with logging.OutputLogger(str(suite_logdir)):
                if run_attack:
                    results = benchmark.benchmark_suite_with_injections(
                        tools_pipeline,
                        suite,
                        attack,
                        suite_logdir,
                        force_rerun=force_rerun,
                        user_tasks=user_tasks,
                        injection_tasks=selected_injection_tasks,
                    )
                else:
                    results = benchmark.benchmark_suite_without_injections(
                        tools_pipeline,
                        suite,
                        suite_logdir,
                        force_rerun=force_rerun,
                        user_tasks=user_tasks,
                    )
        except OpenRouterRateLimitError as exc:
            print("=" * 80)
            print("OpenRouter run paused")
            print("=" * 80)
            print(str(exc))
            print("Re-run the same command with force_rerun=False to continue from completed result files.")
            return

        total_utility_results.extend(results["utility_results"].values())
        if run_attack:
            total_security_results.extend(results["security_results"].values())

        summarize_results(suite_name=suite_name, results=results, run_attack=run_attack)

    print("=" * 80)
    print("Overall")
    print("=" * 80)
    if not total_utility_results:
        print("overall - utility: n/a (0 completed cases)")
        return
    print(f"overall - utility: {sum(total_utility_results) / len(total_utility_results):.4f}")
    if run_attack:
        overall_security = sum(total_security_results) / len(total_security_results)
        print(f"overall - security: {overall_security:.4f}")
        print(f"overall - attack_success_rate: {1.0 - overall_security:.4f}")

