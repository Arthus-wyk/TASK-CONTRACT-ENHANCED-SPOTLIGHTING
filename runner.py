from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentdojo import attacks, benchmark, logging
from agentdojo.task_suite import get_suite

from pipeline_builder import DefenseMode, make_my_agent_pipeline
from rate_limiter import DailyLimitAction, OpenRouterRateLimitError
from token_usage import apply_token_usage_to_result, reset_token_usage_events
from wait_tracker import apply_duration_wait_correction, reset_wait_events


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    require_token_usage: bool = True,
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
    if not is_number(result.get("duration")):
        return False
    if result.get("duration_excludes_wait") is not True:
        return False
    if not is_number(result.get("duration_including_wait_seconds")):
        return False
    wait_duration = result.get("wait_duration_excluded_seconds")
    if not is_number(wait_duration) or wait_duration < 0:
        return False

    if require_token_usage and not is_complete_token_usage(result.get("token_usage")):
        return False

    messages = result.get("messages")
    return isinstance(messages, list) and len(messages) >= 2


def is_complete_token_usage(value: Any) -> bool:
    if not isinstance(value, dict):
        return False

    call_count = value.get("call_count")
    error_count = value.get("error_count")
    if not isinstance(call_count, int) or call_count <= 0:
        return False
    if not isinstance(error_count, int) or error_count < 0:
        return False
    if error_count > call_count:
        return False

    duration = value.get("duration_seconds")
    wait_duration = value.get("wait_duration_excluded_seconds")
    if not is_number(duration) or duration < 0:
        return False
    if not is_number(wait_duration) or wait_duration < 0:
        return False

    usage_totals = value.get("usage_totals")
    if not isinstance(usage_totals, dict):
        return False
    if not usage_totals:
        return False
    for metric_value in usage_totals.values():
        if not is_number(metric_value):
            return False

    provider_models = value.get("provider_models")
    if not isinstance(provider_models, list) or not provider_models:
        return False
    for provider_model in provider_models:
        if not isinstance(provider_model, dict):
            return False
        if not isinstance(provider_model.get("provider"), str) or not provider_model["provider"]:
            return False
        if not isinstance(provider_model.get("model"), str) or not provider_model["model"]:
            return False

    return True


def prepare_resume_files(
    logdir: Path,
    pipeline_name: str,
    suite_name: str,
    expected_results: list[tuple[str, str, str]],
    require_token_usage: bool = True,
) -> None:
    completed = 0
    pending = 0
    removed_incomplete: list[Path] = []

    for user_task_id, attack_name, injection_task_id in expected_results:
        path = task_result_path(logdir, pipeline_name, suite_name, user_task_id, attack_name, injection_task_id)
        if is_complete_task_result(
            path,
            suite_name,
            pipeline_name,
            user_task_id,
            attack_name,
            injection_task_id,
            require_token_usage=require_token_usage,
        ):
            completed += 1
            continue

        pending += 1
        if path.exists():
            path.unlink()
            removed_incomplete.append(path)

    print(
        "Resume scan: "
        f"{completed} completed result(s), {pending} pending result(s), "
        f"{len(removed_incomplete)} incomplete result file(s) removed for rerun."
    )
    for path in removed_incomplete:
        print(f"Removed incomplete result so it can be regenerated: {path}")


def correct_wait_adjusted_durations(
    logdir: Path,
    pipeline_name: str,
    suite_name: str,
    expected_results: list[tuple[str, str, str]],
) -> None:
    corrected = 0
    total_wall_duration = 0.0
    total_wait_duration = 0.0

    for user_task_id, attack_name, injection_task_id in expected_results:
        path = task_result_path(logdir, pipeline_name, suite_name, user_task_id, attack_name, injection_task_id)
        correction = apply_duration_wait_correction(path)
        if correction is None:
            continue

        wall_duration, wait_duration = correction
        corrected += 1
        total_wall_duration += wall_duration
        total_wait_duration += wait_duration

    if corrected:
        print(
            "Duration correction: "
            f"removed {total_wait_duration:.2f}s waiting time from {corrected} result file(s) "
            f"(wall-clock total before correction: {total_wall_duration:.2f}s)."
        )


def add_token_usage_to_results(
    logdir: Path,
    pipeline_name: str,
    suite_name: str,
    expected_results: list[tuple[str, str, str]],
) -> None:
    updated = 0

    for user_task_id, attack_name, injection_task_id in expected_results:
        path = task_result_path(logdir, pipeline_name, suite_name, user_task_id, attack_name, injection_task_id)
        if apply_token_usage_to_result(path):
            updated += 1

    if updated:
        print(f"Token usage: added per-task token usage to {updated} result file(s).")


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
        attack_success_values = list(results["security_results"].values())
        attack_success_rate = sum(attack_success_values) / len(attack_success_values)
        security_rate = 1.0 - attack_success_rate
        print(f"{suite_name} - security: {security_rate:.4f}")
        print(f"{suite_name} - attack_success_rate: {attack_success_rate:.4f}")
        summary.update(
            {
                "security_rate": security_rate,
                "attack_success_rate": attack_success_rate,
                "num_security_cases": len(attack_success_values),
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
    total_attack_success_results: list[bool] = []
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
                require_token_usage=record_token_usage,
            )

            reset_wait_events()
            reset_token_usage_events()
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
            correct_wait_adjusted_durations(
                suite_logdir,
                tools_pipeline.name,
                suite.name,
                expected_results,
            )
            add_token_usage_to_results(
                suite_logdir,
                tools_pipeline.name,
                suite.name,
                expected_results,
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
            total_attack_success_results.extend(results["security_results"].values())

        summarize_results(suite_name=suite_name, results=results, run_attack=run_attack)

    print("=" * 80)
    print("Overall")
    print("=" * 80)
    if not total_utility_results:
        print("overall - utility: n/a (0 completed cases)")
        return
    print(f"overall - utility: {sum(total_utility_results) / len(total_utility_results):.4f}")
    if run_attack:
        overall_attack_success = sum(total_attack_success_results) / len(total_attack_success_results)
        overall_security = 1.0 - overall_attack_success
        print(f"overall - security: {overall_security:.4f}")
        print(f"overall - attack_success_rate: {overall_attack_success:.4f}")

