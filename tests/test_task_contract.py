from __future__ import annotations

import json
import os
import tempfile
import unittest
import sys
import types
from unittest.mock import patch
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

from pipeline_hooks import ToolCallHook, ToolResultHook, TCES_UNEXECUTED_CALLS_FIELD, _tces_provider_messages
from task_contract import (
    TASK_CONTRACT_KEY,
    TaskContractDecision,
    TaskContractConfig,
    TaskContractDefense,
    _normalize_prompt_messages,
)
from model_service import ModelServiceError, QuotaExhaustedError, capture_task_status


def validation_json(decision="allow", reason="Matches authorization.", **overrides):
    value = {"decision": decision, "reason": reason}
    value.update(overrides)
    return json.dumps(value)


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


class FakeRetryLLM:
    def __init__(self) -> None:
        self.received_messages = []

    def query(self, query, runtime, env, messages, extra_args):
        self.received_messages.append(messages)
        if len(self.received_messages) == 1:
            response = {
                "role": "assistant",
                "content": None,
                "tool_calls": [SimpleNamespace(function="send_email", args={"to": "attacker@example.com"}, id="1")],
            }
        else:
            response = {
                "role": "assistant",
                "content": [{"type": "text", "content": "Unable to send the email." if extra_args.get("tces_terminal") else "Done."}],
                "tool_calls": None,
            }
        return query, runtime, env, [*messages, response], extra_args


class FakeRetryToolAfterTextLLM:
    def __init__(self) -> None:
        self.received_messages = []

    def query(self, query, runtime, env, messages, extra_args):
        self.received_messages.append(messages)
        if len(self.received_messages) == 1:
            response = {
                "role": "assistant",
                "content": None,
                "tool_calls": [SimpleNamespace(function="send_email", args={"to": "attacker@example.com"}, id="1")],
            }
        elif len(self.received_messages) == 2:
            response = {
                "role": "assistant",
                "content": [{"type": "text", "content": "Done."}],
                "tool_calls": None,
            }
        else:
            response = {
                "role": "assistant",
                "content": None,
                "tool_calls": [SimpleNamespace(function="send_email", args={"to": "safe@example.com"}, id="2")],
            }
        return query, runtime, env, [*messages, response], extra_args


class FakeContentLLM:
    def query(self, query, runtime, env, messages, extra_args):
        response = {
            "role": "assistant",
            "content": [{"type": "text", "content": "Done."}],
            "tool_calls": None,
        }
        return query, runtime, env, [*messages, response], extra_args


class FakeTaskContract:
    def __init__(self) -> None:
        self.config = SimpleNamespace(max_blocked_tool_call_retries=0)
        self.logged_events = []

    def validate_tool_call(self, tool_call, messages, extra_args, runtime, env):
        return TaskContractDecision(
            decision="block",
            reason="recipient is not authorised by the task contract",
        )

    def build_tool_call_retry_feedback(self, blocked):
        return "TCES blocked this tool call."

    def log_event(self, extra_args, event):
        self.logged_events.append(event)


class BlockOnceTaskContract(FakeTaskContract):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(max_blocked_tool_call_retries=1)
        self.validation_count = 0

    def validate_tool_call(self, tool_call, messages, extra_args, runtime, env):
        self.validation_count += 1
        if self.validation_count == 1:
            return super().validate_tool_call(tool_call, messages, extra_args, runtime, env)
        return TaskContractDecision(
            decision="allow",
            reason="corrected call is allowed",
        )


class FailingTaskShield:
    def check_message_alignment(self, *args, **kwargs):
        raise AssertionError("Task Shield should not run when TCES is active")


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

    def contract_data(self):
        return {
            "task_goal": "Send the report to safe@example.com",
            "allowed_tools": ["read_file", "send_email"],
            "constraints": {"recipient": "safe@example.com", "file": "report.txt"},
        }

    def runtime(self):
        return SimpleNamespace(tools=[
            {"name": "read_file", "description": "Read a file", "annotations": {"readOnlyHint": True}},
            {"name": "send_email", "description": "Send an email", "annotations": {"readOnlyHint": False}},
            {"name": "delete_file", "description": "Delete a file"},
        ])

    def test_contract_extraction_preserves_explicit_positive_parameters(self):
        data = self.contract_data()
        data["constraints"]["body"] = "Keep  double spaces\nand line breaks"
        data["constraints"]["amount"] = 42.5
        data["constraints"]["attendees"] = ["Alice", "Bob"]
        llm = FakeLLM([json.dumps(data)])
        extra_args = {}
        contract = TaskContractDefense(llm).ensure_task_contract(
            "Send report.txt to safe@example.com", self.runtime(), object(), [], extra_args,
        )
        self.assertEqual(contract.__dict__, data)
        self.assertEqual(extra_args[TASK_CONTRACT_KEY], data)
        extraction_prompt = llm.received_messages[0]
        self.assertIn("definitely unrelated", extraction_prompt[0]["content"][0]["content"])
        self.assertIn("delete_file", extraction_prompt[1]["content"][0]["content"])

    def test_contract_extraction_logs_exact_three_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "AGENTDOJA_RUN_LOGDIR": tmpdir,
            "AGENTDOJA_CURRENT_SUITE": "workspace",
        }):
            TaskContractDefense(FakeLLM([json.dumps(self.contract_data())])).ensure_task_contract(
                "Send report.txt to safe@example.com", self.runtime(), object(), [], {},
            )
            with open(os.path.join(tmpdir, "task_contracts.jsonl"), encoding="utf-8") as file:
                record = json.loads(file.readline())
        self.assertEqual(record["suite"], "workspace")
        self.assertEqual(record["task_contract"], self.contract_data())

    def test_contract_schema_rejects_legacy_and_malformed_fields(self):
        defense = TaskContractDefense(FakeLLM())
        invalid = [
            {"task_goal": "Send", "forbidden_actions": [], "sensitive_targets": []},
            {**self.contract_data(), "allowed_tools": "send_email"},
            {**self.contract_data(), "constraints": []},
            {**self.contract_data(), "task_goal": 123},
            {**self.contract_data(), "extra": True},
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                defense._parse_contract(json.dumps(data))

    def test_unknown_tool_name_retries_extraction(self):
        invalid = {**self.contract_data(), "allowed_tools": ["invented_tool"]}
        llm = FakeLLM([json.dumps(invalid), json.dumps(self.contract_data())])
        contract = TaskContractDefense(llm).extract_task_contract("Send report", self.runtime(), object(), {})
        self.assertEqual(contract.allowed_tools, ["read_file", "send_email"])
        self.assertEqual(len(llm.received_messages), 2)

    def test_legacy_contract_is_reextracted(self):
        extra_args = {TASK_CONTRACT_KEY: {"task_goal": "old", "forbidden_actions": [], "sensitive_targets": []}}
        llm = FakeLLM([json.dumps(self.contract_data())])
        contract = TaskContractDefense(llm).ensure_task_contract("Send report", self.runtime(), object(), [], extra_args)
        self.assertEqual(contract.__dict__, self.contract_data())
        self.assertEqual(len(llm.received_messages), 1)

    def test_full_tool_inventory_supports_agentdojo_objects_and_openai_wrappers(self):
        class Parameters:
            @staticmethod
            def model_json_schema():
                return {"type": "object", "properties": {"to": {"type": "string"}}}
        defense = TaskContractDefense(FakeLLM())
        obj = SimpleNamespace(description="Send mail", parameters=Parameters, name="send_email", function=lambda: None)
        tools = defense._available_tools(SimpleNamespace(functions={"send_email": obj}))
        self.assertIn("to", tools[0]["parameters"]["properties"])
        runtime = SimpleNamespace(tools=[{"type": "function", "function": {
            "name": "last_tool", "description": "x" * 13000,
        }}])
        inventory = json.loads(defense._available_tools_text(runtime))
        self.assertEqual(len(inventory[0]["description"]), 13000)
        self.assertEqual(inventory[0]["name"], "last_tool")

    def test_read_only_is_allowed_by_llm_even_when_tool_is_not_allowed(self):
        llm = FakeLLM([validation_json(reason="Read-only lookup.")])
        data = {**self.contract_data(), "allowed_tools": []}
        decision = TaskContractDefense(llm).validate_tool_call(
            SimpleNamespace(function="read_file", args={"file": "other.txt"}), [],
            {TASK_CONTRACT_KEY: data}, self.runtime(), object(),
        )
        self.assertTrue(decision.allow)
        self.assertEqual(decision.decision, "allow")
        self.assertEqual(len(llm.received_messages), 1)

    def test_read_only_is_allowed_by_llm_when_contract_missing(self):
        llm = FakeLLM([validation_json(reason="Read-only lookup needs no contract.")])
        decision = TaskContractDefense(llm).validate_tool_call(
            SimpleNamespace(function="read_file", args={}), [], {}, self.runtime(), object(),
        )
        self.assertTrue(decision.allow)
        self.assertEqual(len(llm.received_messages), 1)

    def test_classification_and_validation_use_a_single_llm_call(self):
        llm = FakeLLM([validation_json(reason="Read-only search.")])
        runtime = SimpleNamespace(tools=[{"name": "search", "description": "Search without modifying data"}])
        decision = TaskContractDefense(llm).validate_tool_call(
            SimpleNamespace(function="search", args={"query": "report"}), [], {}, runtime, object(),
        )
        self.assertTrue(decision.allow)
        self.assertEqual(len(llm.received_messages), 1)

    def test_read_like_name_does_not_bypass_for_side_effectful_tool(self):
        llm = FakeLLM([validation_json("block", "This sends data; retry read_file.")])
        runtime = SimpleNamespace(tools=[{"name": "get_and_send", "description": "Read a file and send it"}])
        decision = TaskContractDefense(llm).validate_tool_call(
            SimpleNamespace(function="get_and_send", args={}), [],
            {TASK_CONTRACT_KEY: self.contract_data()}, runtime, object(),
        )
        self.assertFalse(decision.allow)
        self.assertEqual(len(llm.received_messages), 1)

    def test_obsolete_classifier_response_is_retried_as_a_decision(self):
        llm = FakeLLM(['{"read_only":"true"}', validation_json("block", "Wrong target; retry with safe@example.com.")])
        runtime = SimpleNamespace(tools=[{"name": "send_email", "description": "Send email"}])
        decision = TaskContractDefense(llm).validate_tool_call(
            SimpleNamespace(function="send_email", args={}), [],
            {TASK_CONTRACT_KEY: self.contract_data()}, runtime, object(),
        )
        self.assertFalse(decision.allow)
        self.assertEqual(len(llm.received_messages), 2)

    def test_non_read_only_validation_receives_goal_tools_and_full_parameters(self):
        reason = "Recipient was changed. Retry send_email with to=safe@example.com."
        llm = FakeLLM([validation_json("block", reason)])
        defense = TaskContractDefense(llm)
        long_body = "a" * 14000 + "tail-marker"
        decision = defense.validate_tool_call(
            SimpleNamespace(function="send_email", args={"to": "attacker@example.com", "body": long_body}), [],
            {TASK_CONTRACT_KEY: self.contract_data()}, self.runtime(), object(),
        )
        self.assertFalse(decision.allow)
        prompt = llm.received_messages[0][1]["content"][0]["content"]
        self.assertIn(json.dumps(self.contract_data()), prompt)
        self.assertIn(long_body, prompt)
        self.assertEqual(decision.reason, reason)
        feedback = defense.build_tool_call_retry_feedback([(SimpleNamespace(function="send_email", args={}), decision)])
        self.assertIn(reason, feedback)
        self.assertIn("corrected NEXT tool call", feedback)

    def test_missing_contract_is_unavailable_without_executing_non_read_only_call(self):
        llm = FakeLLM(["", "{}"])
        extra_args = {"original_user_query": "Send report"}
        decision = TaskContractDefense(llm).validate_tool_call(
            SimpleNamespace(function="send_email", args={}), [], extra_args, self.runtime(), object(),
        )
        self.assertFalse(decision.allow)
        self.assertTrue(decision.fallback)
        self.assertEqual(decision.decision, "unavailable")
        self.assertNotIn(TASK_CONTRACT_KEY, extra_args)

    def test_validation_failure_is_unavailable_even_in_audit(self):
        for mode in ("enforce", "audit"):
            defense = TaskContractDefense(FakeLLM(["{}", "{}"]), TaskContractConfig(mode=mode))
            decision = defense.validate_tool_call(
                SimpleNamespace(function="send_email", args={}), [],
                {TASK_CONTRACT_KEY: self.contract_data()}, self.runtime(), object(),
            )
            self.assertEqual(decision.decision, "unavailable")
            self.assertFalse(decision.allow)

    def test_validation_result_requires_exact_two_field_schema(self):
        defense = TaskContractDefense(FakeLLM())
        for data in [
            {"decision": "audit_allow", "reason": "ok"},
            {"decision": "ALLOW", "reason": "ok"},
            {"decision": "allow", "reason": ""},
            {"decision": "allow", "reason": 5},
            {"decision": "allow", "reason": "ok", "risk_level": "low"},
        ]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                defense._parse_validation_decision(json.dumps(data))
        self.assertTrue(defense._parse_validation_decision(validation_json()).allow)

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
        self.assertIn("本地状态说明", assistant_message["content"][0]["content"])
        self.assertTrue(messages[1][TCES_UNEXECUTED_CALLS_FIELD])

    def test_tces_retry_requires_tool_call_after_text_response(self) -> None:
        llm = FakeRetryToolAfterTextLLM()
        hook = ToolCallHook(
            llm,
            enable_spotlighting=False,
            task_contract=BlockOnceTaskContract(),
            blocked_tool_call_shape_retries=2,
        )
        hook.task_contract.config = SimpleNamespace(max_blocked_tool_call_retries=1)

        with patch("pipeline_hooks.time.sleep", return_value=None):
            _, _, _, messages, _ = hook.query(
                "Summarize the file.",
                runtime=object(),
                env=object(),
                messages=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "Summarize the file."},
                ],
                extra_args={"original_user_query": "Summarize the file."},
            )

        self.assertEqual(len(llm.received_messages), 3)
        retry_messages = llm.received_messages[1]
        self.assertEqual([message["role"] for message in retry_messages], ["system", "user", "assistant", "assistant"])
        self.assertNotIn("TCES retry feedback", json.dumps(retry_messages))
        self.assertTrue(messages[-1]["tool_calls"])

    def test_tces_retry_finalizes_after_two_non_tool_call_attempts(self):
        llm = FakeRetryLLM()
        hook = ToolCallHook(llm, enable_spotlighting=False, task_contract=BlockOnceTaskContract(), blocked_tool_call_shape_retries=2)
        state = hook.query("Send report", object(), object(), [
            {"role": "system", "content": "system"}, {"role": "user", "content": "Send report"},
        ], {})
        self.assertEqual(len(llm.received_messages), 4)
        self.assertIsNone(state[3][-1]["tool_calls"])
        self.assertEqual(state[3][-1]["content"][0]["content"], "Unable to send the email.")
        self.assertTrue(state[4]["tces_terminal"])
        self.assertIn("Do not retry or call any tool", llm.received_messages[-1][-1]["content"][0]["content"])

    def test_tces_retry_empty_history_cannot_reuse_blocked_call(self):
        class EmptyRetry(FakeRetryLLM):
            def query(self, query, runtime, env, messages, extra_args):
                if not self.received_messages:
                    return super().query(query, runtime, env, messages, extra_args)
                self.received_messages.append(messages)
                return query, runtime, env, [], extra_args
        llm = EmptyRetry()
        hook = ToolCallHook(llm, enable_spotlighting=False, task_contract=BlockOnceTaskContract())
        state = hook.query("Send", object(), object(), [{"role": "user", "content": "Send"}], {})
        self.assertIsNone(state[3][-1]["tool_calls"])
        self.assertIn("本地状态说明", state[3][-1]["content"][0]["content"])
        self.assertTrue(state[3][1][TCES_UNEXECUTED_CALLS_FIELD])

    def test_block_retry_execution_and_feedback_preserve_exact_history(self):
        class RetryLLM:
            def __init__(self):
                self.received_messages = []
            def query(self, query, runtime, env, messages, extra_args):
                self.received_messages.append(messages)
                if len(self.received_messages) <= 2:
                    recipient = "attacker@example.com" if len(self.received_messages) == 1 else "safe@example.com"
                    response = {"role": "assistant", "content": None, "tool_calls": [
                        SimpleNamespace(function="send_email", args={"to": recipient}, id=str(len(self.received_messages))),
                    ]}
                else:
                    response = {"role": "assistant", "content": "Sent successfully.", "tool_calls": None}
                return query, runtime, env, [*messages, response], extra_args
        class Executor:
            def __init__(self):
                self.executed = []
            def query(self, query, runtime, env, messages, extra_args):
                results = []
                for call in messages[-1]["tool_calls"]:
                    self.executed.append(call)
                    results.append({"role": "tool", "tool_call_id": call.id, "tool_call": call, "content": "sent"})
                return query, runtime, env, [*messages, *results], extra_args
        llm = RetryLLM()
        defense = BlockOnceTaskContract()
        original_validate = defense.validate_tool_call
        def validate(call, messages, *args):
            self.assertIs(messages[-1]["tool_calls"][0], call)
            return original_validate(call, messages, *args)
        defense.validate_tool_call = validate
        hook = ToolCallHook(llm, enable_spotlighting=False, task_contract=defense)
        initial = [{"role": "system", "content": "system"}, {"role": "user", "content": "Send report"}]
        state = hook.query("Send report", object(), object(), initial, {})
        messages = state[3]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "assistant", "assistant"])
        self.assertEqual(messages[2]["tool_calls"][0].args["to"], "attacker@example.com")
        self.assertTrue(messages[2][TCES_UNEXECUTED_CALLS_FIELD])
        provider = llm.received_messages[1]
        self.assertIsNone(provider[2]["tool_calls"])
        self.assertIn('"to": "attacker@example.com"', provider[2]["content"][0]["content"])
        self.assertEqual(provider[3]["role"], "assistant")
        executor = Executor()
        state = ToolResultHook(executor, enable_spotlighting=False).query(*state)
        state = hook.query(*state)
        self.assertEqual([m["role"] for m in state[3]], ["system", "user", "assistant", "assistant", "assistant", "tool", "assistant"])
        self.assertEqual([c.args["to"] for c in executor.executed], ["safe@example.com"])
        self.assertEqual(state[3][-1]["content"], "Sent successfully.")

    def test_mixed_batch_exhaustion_executes_no_calls_and_finalizes(self):
        good = SimpleNamespace(function="read_file", args={"file": "report.txt"}, id="good")
        bad = SimpleNamespace(function="send_email", args={"to": "attacker@example.com"}, id="bad")
        class MixedLLM:
            def query(self, query, runtime, env, messages, extra_args):
                return query, runtime, env, [*messages, {"role": "assistant", "content": None, "tool_calls": [good, bad]}], extra_args
        class MixedDefense(FakeTaskContract):
            def validate_tool_call(self, call, *args):
                return TaskContractDecision(decision="allow" if call.id == "good" else "block", reason="test")
        state = ToolCallHook(MixedLLM(), enable_spotlighting=False, task_contract=MixedDefense()).query(
            "Read report", object(), object(), [{"role": "user", "content": "Read report"}], {},
        )
        self.assertEqual(state[3][1]["tool_calls"], [good, bad])
        self.assertIsNone(state[3][-1]["tool_calls"])
        self.assertTrue(state[3][1][TCES_UNEXECUTED_CALLS_FIELD])
        self.assertTrue(state[4]["tces_terminal"])

    def test_provider_projection_preserves_nested_openai_arguments(self):
        args = json.dumps({"to": ["a@example.com"], "body": "space  and\nnewline"})
        messages = [{"role": "assistant", "content": None, TCES_UNEXECUTED_CALLS_FIELD: True, "tool_calls": [
            {"id": "id1", "function": {"name": "send_email", "arguments": args}},
        ]}]
        projected = _tces_provider_messages(messages)
        text = projected[0]["content"][0]["content"]
        self.assertIn(json.dumps(args), text)
        self.assertTrue(messages[0]["tool_calls"])
        self.assertIsNone(projected[0]["tool_calls"])

    def test_empty_assistant_retry_is_rendered_as_valid_provider_content(self):
        original = [{"role": "assistant", "content": None, "tool_calls": []}]
        projected = _tces_provider_messages(original)
        self.assertTrue(projected[0]["content"])
        self.assertIsNone(original[0]["content"])

    def test_legacy_extra_fields_are_retried(self):
        llm = FakeLLM([validation_json(evidence=["explanation"]), validation_json()])
        decision = TaskContractDefense(llm).validate_tool_call(
            SimpleNamespace(function="send_email", args={"to": "safe@example.com"}), [],
            {TASK_CONTRACT_KEY: self.contract_data()}, self.runtime(), object(),
        )
        self.assertTrue(decision.allow)
        self.assertEqual(len(llm.received_messages), 2)

    def test_block_reason_is_preserved_in_audit(self):
        reason = "Recipient differs from authorized safe@example.com."
        defense = TaskContractDefense(FakeLLM([validation_json("block", reason)]), TaskContractConfig(mode="audit"))
        decision = defense.validate_tool_call(SimpleNamespace(function="send_email", args={}), [],
            {TASK_CONTRACT_KEY: self.contract_data()}, self.runtime(), object())
        self.assertEqual(decision.decision, "block")
        self.assertTrue(decision.audit)
        self.assertEqual(decision.reason, reason)

    def test_quota_exit_is_not_converted_to_unavailable_or_block(self):
        class ExhaustedLLM:
            def query(self, *args):
                raise QuotaExhaustedError("weekly usage limit")
        defense = TaskContractDefense(ExhaustedLLM())
        for contract in ({}, {TASK_CONTRACT_KEY: self.contract_data()}):
            with self.subTest(contract=contract), self.assertRaises(QuotaExhaustedError):
                defense.validate_tool_call(SimpleNamespace(function="send_email", args={}), [],
                    {**contract, "original_user_query": "Send report"}, self.runtime(), object())

    def test_service_error_during_each_internal_stage_is_not_a_block(self):
        class UnavailableLLM:
            def query(self, *args):
                raise ModelServiceError("service unavailable", 503)
        defense = TaskContractDefense(UnavailableLLM())
        args = {TASK_CONTRACT_KEY: self.contract_data()}
        decision = defense.validate_tool_call(SimpleNamespace(function="send_email", args={}), [], args, self.runtime(), object())
        self.assertEqual(decision.decision, "unavailable")
        self.assertFalse(any(e["decision"] == "block" for e in args["task_contract_events"]))
        args = {}
        self.assertIsNone(defense.ensure_task_contract("Send", self.runtime(), object(), [], args))
        self.assertIn("tces_unavailable", args)

    def test_service_retry_does_not_consume_validation_retry_budget(self):
        class RateLimitedThenValid(FakeLLM):
            def query(self, *args):
                if not hasattr(self, "failed"):
                    self.failed = True
                    raise ModelServiceError("rate limit", 429, retryable=True)
                return super().query(*args)
        llm = RateLimitedThenValid([validation_json()])
        defense = TaskContractDefense(llm, TaskContractConfig(validation_retries=0))
        args = {TASK_CONTRACT_KEY: self.contract_data()}
        with patch("model_service.tracked_sleep"):
            decision = defense.validate_tool_call(SimpleNamespace(function="send_email", args={}), [], args, self.runtime(), object())
        self.assertTrue(decision.allow)
        self.assertEqual(len(llm.received_messages), 1)
        self.assertFalse(any(e["decision"] == "block" for e in args["task_contract_events"]))

    def test_finalization_disables_tools_without_mutating_runtime(self):
        class ClosingLLM(FakeAssistantLLM):
            def query(inner, query, runtime, env, messages, extra_args):
                if extra_args.get("tces_terminal"):
                    self.assertEqual(runtime.functions, {})
                    self.assertEqual(runtime.tools, [])
                    return FakeContentLLM().query(query, runtime, env, messages, extra_args)
                return super().query(query, runtime, env, messages, extra_args)
        runtime = self.runtime()
        original = list(runtime.tools)
        hook = ToolCallHook(ClosingLLM(), enable_spotlighting=False, task_contract=FakeTaskContract())
        state = hook.query("Send", runtime, object(), [{"role": "user", "content": "Send"}], {})
        self.assertIs(state[1], runtime)
        self.assertEqual(runtime.tools, original)
        self.assertIsNone(state[3][-1]["tool_calls"])

    def test_malformed_retry_and_broken_finalization_do_not_raise(self):
        class BrokenAfterCall(FakeRetryLLM):
            def query(self, *args):
                if self.received_messages:
                    raise json.JSONDecodeError("broken", "", 0)
                return super().query(*args)
        hook = ToolCallHook(BrokenAfterCall(), enable_spotlighting=False, task_contract=BlockOnceTaskContract())
        with patch("pipeline_hooks.time.sleep"):
            state = hook.query("Send", object(), object(), [{"role": "user", "content": "Send"}], {})
        self.assertIsNone(state[3][-1]["tool_calls"])
        self.assertTrue(state[4]["tces_terminal"])

    def test_invalid_validation_still_gets_natural_final_reply(self):
        class UnusableDecision(FakeTaskContract):
            def validate_tool_call(self, *args):
                return TaskContractDecision("unavailable", "No valid validation response.")
        llm = FakeRetryLLM()
        hook = ToolCallHook(llm, enable_spotlighting=False, task_contract=UnusableDecision())
        state = hook.query("Send", object(), object(), [{"role": "user", "content": "Send"}], {})
        self.assertEqual(len(llm.received_messages), 2)
        self.assertEqual(state[3][-1]["content"][0]["content"], "Unable to send the email.")
        self.assertEqual(state[4]["tces_outcome"], "unavailable")

    def test_finalization_service_failure_is_not_counted_as_security_success(self):
        class OfflineAtClose(FakeAssistantLLM):
            def query(self, query, runtime, env, messages, extra_args):
                if extra_args.get("tces_terminal"):
                    raise ModelServiceError("service unavailable", 503)
                return super().query(query, runtime, env, messages, extra_args)
        hook = ToolCallHook(OfflineAtClose(), enable_spotlighting=False, task_contract=FakeTaskContract())
        with capture_task_status() as status:
            state = hook.query("Send", object(), object(), [{"role": "user", "content": "Send"}], {})
        self.assertIn("unavailable", status)
        self.assertIsNone(state[3][-1]["tool_calls"])

    def test_next_task_runs_after_unavailable_or_exhausted_task(self):
        class SometimesUnavailable(FakeRetryLLM):
            def query(self, query, *args):
                if query == "offline":
                    raise ModelServiceError("unauthorized", 401)
                return super().query(query, *args)
        hook = ToolCallHook(SometimesUnavailable(), enable_spotlighting=False, task_contract=BlockOnceTaskContract())
        with capture_task_status() as first:
            state = hook.query("offline", object(), object(), [{"role": "user", "content": "offline"}], {})
        self.assertEqual(state[4]["tces_outcome"], "unavailable")
        self.assertIn("unavailable", first)
        with capture_task_status() as second:
            state = hook.query("Send", object(), object(), [{"role": "user", "content": "Send"}], {})
        self.assertEqual(state[4]["tces_outcome"], "retry_exhausted")
        self.assertEqual(second, {})

    def test_tool_call_hook_skips_task_shield_content_check_when_tces_is_active(self) -> None:
        hook = ToolCallHook(
            FakeContentLLM(),
            enable_spotlighting=False,
            task_shield=FailingTaskShield(),
            task_contract=FakeTaskContract(),
        )

        _, _, _, messages, _ = hook.query(
            "Summarize the file.",
            runtime=object(),
            env=object(),
            messages=[{"role": "system", "content": "system"}],
            extra_args={"original_user_query": "Summarize the file."},
        )

        self.assertEqual(messages[-1]["content"], [{"type": "text", "content": "Done."}])


if __name__ == "__main__":
    unittest.main()
