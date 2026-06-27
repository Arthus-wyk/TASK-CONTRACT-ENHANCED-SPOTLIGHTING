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


DefenseMode = Literal["no_defense", "spotlighting"]


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
    if defense not in {"no_defense", "spotlighting"}:
        raise ValueError("defense must be 'no_defense' or 'spotlighting'")

    llm = make_llm(
        model=model,
        reasoning_effort=reasoning_effort,
        thinking_budget_tokens=thinking_budget_tokens,
        token_log_dir=token_log_dir,
        rate_limit_state_path=rate_limit_state_path,
        daily_limit_action=daily_limit_action,
    )
    enable_spotlighting = defense == "spotlighting"
    llm_with_tool_call_hook = ToolCallHook(llm, enable_spotlighting=enable_spotlighting)
    tools_executor_with_result_hook = ToolResultHook(
        agent_pipeline.ToolsExecutor(),
        enable_spotlighting=enable_spotlighting,
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
            MyAgentPreProcessor(),
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

    agentdojo_model_key = "local" if provider == "ollama" else model_name
    pipeline.name = safe_filename_name(f"{agentdojo_model_key}-{provider}-{model_name}+my_agent+{defense}")

    return pipeline
