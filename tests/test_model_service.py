import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from model_service import (ModelServiceError, ProviderResponseGuard, check_response,
                           query_model, service_error, text_only_request,
                           structured_output_request, QuotaExhaustedError)


class ModelServiceTests(unittest.TestCase):
    def test_structured_validation_request_and_context_cleanup(self):
        from task_contract import CONTRACT_VALIDATION_SCHEMA
        create = Mock(return_value={"choices": []})
        guard = ProviderResponseGuard(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
        with structured_output_request(CONTRACT_VALIDATION_SCHEMA):
            guard.chat.completions.create(tool_choice="auto", tools=[{"name": "send"}])
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], "none")
        spec = kwargs["response_format"]["json_schema"]
        self.assertTrue(spec["strict"])
        self.assertEqual(set(spec["schema"]["properties"]), {"decision", "reason"})
        self.assertFalse(spec["schema"]["additionalProperties"])
        guard.chat.completions.create(tool_choice="auto")
        self.assertNotIn("response_format", create.call_args.kwargs)
        self.assertEqual(create.call_args.kwargs["tool_choice"], "auto")

    def test_structured_quota_code_stops_without_retry_or_finalization(self):
        def query(*args):
            check_response({"error": {"type": "insufficient_quota", "message": "Balance depleted"}})
        llm = SimpleNamespace(query=Mock(side_effect=query))
        with patch("model_service.tracked_sleep") as sleep, self.assertRaises(QuotaExhaustedError):
            query_model(llm, "", object(), object(), [], {})
        self.assertEqual(llm.query.call_count, 1)
        sleep.assert_not_called()

    def test_structured_errors_are_detected_before_choice_parsing(self):
        for payload in ({"error": {"code": 429, "message": "too many requests"}},
                        SimpleNamespace(error={"type": "rate_limit_error", "message": "limited"}),
                        '{"error":{"code":503,"message":"unavailable"}}'):
            with self.subTest(payload=payload), self.assertRaises(ModelServiceError):
                check_response(payload)

    def test_normal_text_and_quoted_errors_are_not_provider_errors(self):
        for content in ('Explain HTTP 429 and quota limits.',
                        'Example: {"error":{"code":429}}',
                        {"choices": [{"message": {"content": 'HTTP 429 rate limit'}}]}):
            check_response(content)

    def test_sdk_status_detected_without_llm(self):
        exc = RuntimeError("Request rejected")
        exc.status_code = 429
        result = service_error(exc)
        self.assertTrue(result.retryable)
        self.assertIsNone(service_error(ValueError("quota is a word in a normal validation message")))

    def test_guard_disables_tools_and_preserves_api_key_rotation(self):
        create = Mock(return_value={"choices": []})
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), api_key="old")
        guard = ProviderResponseGuard(client)
        guard.api_key = "new"
        with text_only_request():
            guard.chat.completions.create(tools=[{"name": "send"}], tool_choice="auto")
        self.assertEqual(create.call_args.kwargs["tool_choice"], "none")
        guard.chat.completions.create(tool_choice="auto")
        self.assertEqual(create.call_args.kwargs["tool_choice"], "auto")
        self.assertEqual(client.api_key, "new")

    def test_guard_checks_http_success_error_body(self):
        create = Mock(return_value={"error": {"code": 429, "message": "limited"}})
        guard = ProviderResponseGuard(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
        with self.assertRaises(ModelServiceError):
            guard.chat.completions.create()

    def test_retry_budget_is_finite_and_not_a_task_decision(self):
        llm = SimpleNamespace(query=Mock(side_effect=ModelServiceError("limited", 429, retryable=True)))
        args = {}
        with patch("model_service.tracked_sleep"), self.assertRaises(ModelServiceError):
            query_model(llm, "", object(), object(), [], args)
        self.assertEqual(llm.query.call_count, 3)
        self.assertEqual([e["event"] for e in args["model_service_events"]], ["retry", "retry", "unavailable"])
        self.assertNotIn("task_contract_events", args)


if __name__ == "__main__":
    unittest.main()
