from __future__ import annotations

import unittest
from types import SimpleNamespace

from task_shield import TaskShield, _normalize_prompt_messages


class FakeLLM:
    def __init__(self) -> None:
        self.received_messages = []

    def query(self, query, runtime, env, messages, extra_args):
        self.received_messages = messages
        response = {
            "role": "assistant",
            "content": [{"type": "text", "content": '["normalized"]'}],
            "tool_calls": None,
        }
        return query, runtime, env, [*messages, response], extra_args


class TaskShieldMessageFormatTests(unittest.TestCase):
    def test_internal_prompts_use_agentdojo_content_blocks(self) -> None:
        messages = _normalize_prompt_messages(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ]
        )

        self.assertEqual(messages[0]["content"], [{"type": "text", "content": "system prompt"}])
        self.assertEqual(messages[1]["content"], [{"type": "text", "content": "user prompt"}])

    def test_query_llm_text_normalizes_messages_before_adapter_call(self) -> None:
        llm = FakeLLM()
        shield = TaskShield(llm)

        text = shield._query_llm_text(
            [{"role": "user", "content": "check this"}],
            runtime=object(),
            env=object(),
            extra_args={},
        )

        self.assertEqual(text, '["normalized"]')
        self.assertEqual(
            llm.received_messages[0]["content"],
            [{"type": "text", "content": "check this"}],
        )

    def test_agentdojo_function_call_args_are_included_in_instruction(self) -> None:
        shield = TaskShield(FakeLLM())
        tool_call = SimpleNamespace(
            function="search_emails",
            args={"query": "TODOs for the week"},
            id="call-1",
        )

        instruction = shield.tool_call_to_instruction(tool_call)

        self.assertIn('"search_emails"', instruction)
        self.assertIn('"query": "TODOs for the week"', instruction)


if __name__ == "__main__":
    unittest.main()
