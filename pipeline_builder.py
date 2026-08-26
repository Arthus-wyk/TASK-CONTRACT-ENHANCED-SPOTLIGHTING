from __future__ import annotations

from pathlib import Path
from typing import Literal

from openai.types.chat import ChatCompletionReasoningEffort

from agentdojo import agent_pipeline
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from defenses import MyAgentPreProcessor
from llm_factory import make_llm, parse_model
from pipeline_hooks import SpotlightingMode, ToolCallHook, ToolResultHook
from rate_limiter import DailyLimitAction
from task_contract import TaskContractDefense
from task_shield import TaskShield


DefenseMode = Literal[
    "no_defense",
    "spotlighting",
    "delimiting",
    "task_shield",
    "task-shield",
    "spotlighting_task_shield",
    "spotlighting-task-shield",
    "delimiting_task_shield",
    "delimiting-task-shield",
    "TCES",
    "tces",
]

SUPPORTED_DEFENSES = {
    "no_defense",
    "spotlighting",
    "delimiting",
    "task_shield",
    "spotlighting_task_shield",
    "delimiting_task_shield",
    "tces",
}

AGENTDOJO_MODEL_NAME_KEYS = {
    "ollama_cloud": "local",
    "openrouter": "local",
}


def canonical_defense_name(defense: str) -> str:
    return defense.replace("-", "_").lower()


def spotlighting_mode_for_defense(defense_name: str) -> SpotlightingMode:
    if defense_name in {"delimiting", "delimiting_task_shield", "tces"}:
        return "delimiting"
    return "datamarking"


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
            + ", ".join(
                sorted(
                    SUPPORTED_DEFENSES
                    | {"TCES", "task-shield", "spotlighting-task-shield", "delimiting-task-shield"}
                )
            )
        )

    llm = make_llm(
        model=model,
        reasoning_effort=reasoning_effort,
        thinking_budget_tokens=thinking_budget_tokens,
        token_log_dir=token_log_dir,
        defense=defense_name,
        rate_limit_state_path=rate_limit_state_path,
        daily_limit_action=daily_limit_action,
    )
    enable_spotlighting = defense_name in {
        "spotlighting",
        "spotlighting_task_shield",
        "delimiting",
        "delimiting_task_shield",
        "tces",
    }
    spotlighting_mode = spotlighting_mode_for_defense(defense_name)
    enable_task_shield = defense_name in {"task_shield", "spotlighting_task_shield", "delimiting_task_shield"}
    enable_task_contract = defense_name == "tces"
    task_shield = TaskShield(llm) if enable_task_shield else None
    task_contract = TaskContractDefense(llm) if enable_task_contract else None
    llm_with_tool_call_hook = ToolCallHook(
        llm,
        enable_spotlighting=enable_spotlighting,
        spotlighting_mode=spotlighting_mode,
        task_shield=task_shield,
        task_contract=task_contract,
    )
    tools_executor_with_result_hook = ToolResultHook(
        agent_pipeline.ToolsExecutor(),
        enable_spotlighting=enable_spotlighting,
        spotlighting_mode=spotlighting_mode,
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
            MyAgentPreProcessor(task_shield=task_shield, task_contract=task_contract),
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

    # AgentDojo attacks infer a human-readable model label from pipeline.name
    # by searching for a known model key (for example "local" or a GPT/Gemini
    # model ID). Ollama Cloud model IDs are not in its registry, so expose a
    # compatible key in the pipeline name.
    agentdojo_model_key = AGENTDOJO_MODEL_NAME_KEYS.get(provider, provider)
    pipeline.name = safe_filename_name(f"{agentdojo_model_key}-{provider}-{model_name}+my_agent+{defense_name}")

    return pipeline
