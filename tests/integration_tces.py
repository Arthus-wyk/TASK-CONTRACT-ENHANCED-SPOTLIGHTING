"""Run separately with the project's AgentDojo environment (no network requests)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agentdojo import agent_pipeline, benchmark, logging
from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

from aggregate_results import Bucket
from model_service import ModelServiceError, ProviderResponseGuard, QuotaExhaustedError
from pipeline_hooks import ToolCallHook
from runner import completed_metric_values, log_current_tasks, summarize_results
from task_contract import TaskContractConfig, TaskContractDecision, TaskContractDefense, TASK_CONTRACT_KEY


class BlockAll:
    config = TaskContractConfig(max_blocked_tool_call_retries=0)

    def validate_tool_call(self, *args):
        return TaskContractDecision("block", "Sending to an unauthorized recipient.")

    def build_tool_call_retry_feedback(self, *args):
        return "The pending batch was not executed."

    def log_event(self, *args):
        pass


class IntegrationTests(unittest.TestCase):
    def test_real_adapter_single_structured_judge_call(self):
        runtime = FunctionsRuntime()

        @runtime.register_function
        def read_file(file_id: str) -> str:
            """Read a file without changing it.

            :param file_id: The file to read.
            """
            raise AssertionError("Internal judgment must not execute tools")

        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"decision":"allow","reason":"This is a read-only lookup."}', tool_calls=None))]))
        client = ProviderResponseGuard(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
        defense = TaskContractDefense(OpenAILLM(client, "test-model", None))
        decision = defense.validate_tool_call(SimpleNamespace(function="read_file", args={"file_id": "13"}), [],
            {TASK_CONTRACT_KEY: {"task_goal": "Find report", "allowed_tools": [], "constraints": {}}}, runtime, EmptyEnv())
        self.assertTrue(decision.allow)
        create.assert_called_once()
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], "none")
        self.assertFalse(isinstance(kwargs.get("tools"), list) and kwargs["tools"])
        self.assertEqual(kwargs["response_format"]["json_schema"]["schema"]["required"], ["decision", "reason"])
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])

    def test_runner_exits_on_quota_then_resumes_interrupted_task(self):
        import runner
        attempted = []
        interrupted = [True]

        class Suite:
            name = "test-suite"
            user_tasks = {name: SimpleNamespace(ID=name, PROMPT=name) for name in ("first", "second", "third")}

            def run_task_with_pipeline(self, pipeline, task, injection_task, injections):
                attempted.append(task.ID)
                logging.Logger.get().log([
                    {"role": "user", "content": [{"type": "text", "content": task.PROMPT}]},
                    {"role": "assistant", "content": [{"type": "text", "content": "Working"}], "tool_calls": None},
                ])
                if task.ID == "second" and interrupted[0]:
                    raise QuotaExhaustedError("weekly usage limit reached")
                return True, False

        suite = Suite()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner, "get_suite", return_value=suite), \
             patch.object(runner, "print_suite_tools"), \
             patch.object(runner, "make_my_agent_pipeline", return_value=SimpleNamespace(name="local-test")):
            root = Path(directory)
            with self.assertRaises(SystemExit) as exit_info:
                runner.main(suites=[suite.name], run_attack=False, logdir=root, record_token_usage=False)
            self.assertEqual(exit_info.exception.code, 2)
            self.assertEqual(attempted, ["first", "second"])
            results_dir = root / "test-suite/local-test/test-suite"
            first_path = results_dir / "first/none/none.json"
            first_result = json.loads(first_path.read_text())
            partial = json.loads((results_dir / "second/none/none.json").read_text())
            self.assertEqual(partial["evaluation_status"], "interrupted")
            self.assertIsNone(partial["utility"])
            self.assertFalse((results_dir / "third/none/none.json").exists())
            interrupted[0] = False
            runner.main(suites=[suite.name], run_attack=False, logdir=root, record_token_usage=False)
            self.assertEqual(attempted, ["first", "second", "second", "third"])
            # Resume may add duration/token metadata; the completed evaluation is retained.
            resumed_first = json.loads(first_path.read_text())
            self.assertEqual({key: resumed_first[key] for key in first_result}, first_result)
            self.assertTrue(json.loads((results_dir / "second/none/none.json").read_text())["utility"])
            self.assertEqual(len(list((results_dir / "second/none").glob("*.bak"))), 1)

    def test_real_openai_adapter_and_executor_text_only_close(self):
        executed = []
        runtime = FunctionsRuntime()

        @runtime.register_function
        def send_email(to: str) -> str:
            """Send an email.

            :param to: Recipient address.
            """
            executed.append(to)
            return "sent"

        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                    id="call1", function=SimpleNamespace(name="send_email", arguments='{"to":"wrong@example.com"}'))])
            else:
                self.assertEqual(kwargs["tool_choice"], "none")
                self.assertFalse(isinstance(kwargs.get("tools"), list) and kwargs["tools"])
                msg = SimpleNamespace(content="The email could not be sent.", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        client = ProviderResponseGuard(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
        llm = OpenAILLM(client, "test-model", None)
        hook = ToolCallHook(llm, enable_spotlighting=False, task_contract=BlockAll())
        pipeline = agent_pipeline.AgentPipeline([
            agent_pipeline.InitQuery(), hook,
            agent_pipeline.ToolsExecutionLoop([agent_pipeline.ToolsExecutor(), hook]),
        ])
        state = pipeline.query("Send mail", runtime, EmptyEnv())
        self.assertEqual(executed, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(state[3][-1]["content"][0]["content"], "The email could not be sent.")
        self.assertIn("send_email", runtime.functions)

    def test_actual_benchmark_continues_and_excludes_unavailable_results(self):
        class LLM:
            def query(self, query, runtime, env, messages, extra_args):
                if query == "offline":
                    raise ModelServiceError("service unavailable", 503)
                return query, runtime, env, [*messages, {"role": "assistant", "content": [
                    {"type": "text", "content": "Finished."}], "tool_calls": None}], extra_args

        hook = ToolCallHook(LLM(), enable_spotlighting=False, task_contract=BlockAll())
        hook.name = "local-test"
        class Suite:
            name = "test-suite"
            user_tasks = {name: SimpleNamespace(ID=name, PROMPT=name) for name in ("offline", "online")}
            def run_task_with_pipeline(self, pipeline, task, injection_task, injections):
                state = pipeline.query(task.ID, FunctionsRuntime(), EmptyEnv(), [
                    {"role": "user", "content": [{"type": "text", "content": task.ID}]}], {})
                logging.Logger.get().log(state[3])
                return task.ID == "online", False
        suite = Suite()
        with tempfile.TemporaryDirectory() as directory:
            with logging.OutputLogger(directory), log_current_tasks(suite.name, suite, None):
                results = benchmark.benchmark_suite_without_injections(hook, suite, Path(directory), True)
            self.assertIsNone(results["utility_results"][("offline", "")])
            self.assertTrue(results["utility_results"][("online", "")])
            summary = summarize_results(suite.name, results, False)
            self.assertEqual(summary["num_utility_cases"], 1)
            self.assertEqual(summary["utility_rate"], 1.0)
            saved = json.loads((Path(directory) / "local-test/test-suite/offline/none/none.json").read_text())
            self.assertEqual(saved["evaluation_status"], "unavailable")
            self.assertIsNone(saved["security"])

    def test_unknown_dos_result_is_excluded_and_empty_summary_is_valid(self):
        results = {"utility_results": {("task", "injection"): None},
                   "security_results": {("task", "injection"): True}}
        self.assertEqual(completed_metric_values(results, "security_results"), [])
        summary = summarize_results("empty", results, True)
        self.assertIsNone(summary["security_rate"])
        bucket = Bucket()
        bucket.add({"evaluation_status": "unavailable", "utility": None, "security": True, "injection_task_id": "i"})
        self.assertEqual(bucket.security_total, 0)


if __name__ == "__main__":
    unittest.main()
