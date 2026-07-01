from __future__ import annotations

from pathlib import Path
from typing import Literal

from openai.types.chat import ChatCompletionReasoningEffort

from agentdojo import agent_pipeline
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from defenses import MyAgentPreProcessor
from llm_factory import make_llm, parse_model
from pipeline_hooks import ToolCallHook, ToolResultHook
from rate_limiter import DailyLimitAction
from task_shield import TaskShield


DefenseMode = Literal[
    "no_defense",
    "spotlighting",
    "task_shield",
    "task-shield",
    "spotlighting_task_shield",
    "spotlighting-task-shield",
]

SUPPORTED_DEFENSES = {
    "no_defense",
    "spotlighting",
    "task_shield",
    "spotlighting_task_shield",
}


def canonical_defense_name(defense: str) -> str:
    return defense.replace("-", "_")


def make_my_agent_pipeline(
    model: str,
    suite: str,
    reasoning_effort: ChatCompletionReasoningEffort = "medium",
    thinking_budget_tokens: int | None = None,
    token_log_dir: Path | None = None,
    rate_limit_state_path: Path | None = None,
    daily_limit_action: DailyLimitAction = "wait",
    defense: DefenseMode = "spotlighting",
) -> agent_pipeline.AgentPipeline:
    defense_name = canonical_defense_name(defense)
    if defense_name not in SUPPORTED_DEFENSES:
        raise ValueError(
            "defense must be one of: "
            + ", ".join(sorted(SUPPORTED_DEFENSES | {"task-shield", "spotlighting-task-shield"}))
        )

    llm = make_llm(
        model=model,
        reasoning_effort=reasoning_effort,
        thinking_budget_tokens=thinking_budget_tokens,
        token_log_dir=token_log_dir,
        rate_limit_state_path=rate_limit_state_path,
        daily_limit_action=daily_limit_action,
    )
    enable_spotlighting = defense_name in {"spotlighting", "spotlighting_task_shield"}
    enable_task_shield = defense_name in {"task_shield", "spotlighting_task_shield"}
    task_shield = TaskShield(llm) if enable_task_shield else None
    llm_with_tool_call_hook = ToolCallHook(
        llm,
        enable_spotlighting=enable_spotlighting,
        task_shield=task_shield,
    )
    tools_executor_with_result_hook = ToolResultHook(
        agent_pipeline.ToolsExecutor(),
        enable_spotlighting=enable_spotlighting,
        task_shield=task_shield,
    )

    tools_loop = agent_pipeline.ToolsExecutionLoop(
        [
            tools_executor_with_result_hook,
            llm_with_tool_call_hook,
        ]
    )

    pipeline = agent_pipeline.AgentPipeline(
        [
            agent_pipeline.SystemMessage(load_system_message(None)),
            agent_pipeline.InitQuery(),
            MyAgentPreProcessor(task_shield=task_shield),
            llm_with_tool_call_hook,
            tools_loop,
        ]
    )

    provider, model_name = parse_model(model)
    import re

    def safe_filename_name(name: str) -> str:
        """
        Make pipeline name safe for Windows/macOS/Linux paths.
        AgentDojo uses pipeline.name as part of log/result path.
        """
        return re.sub(r'[<>:"/\\|?*\s]+', "-", name)

    agentdojo_model_key = "local" if provider == "openrouter" else provider
    pipeline.name = safe_filename_name(f"{agentdojo_model_key}-{provider}-{model_name}+my_agent+{defense_name}")

    return pipeline
