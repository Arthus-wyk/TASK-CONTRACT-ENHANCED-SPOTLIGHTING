import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from model_service import QuotaExhaustedError
from ollama_cloud_limiter import OllamaCloudClientProxy, OllamaCloudUsageLimiter
from rate_limiter import OpenRouterRateLimiter, RateLimitedClientProxy


def provider_error(message, status=429, retry_after="7"):
    error = RuntimeError(message)
    error.status_code = status
    error.response = SimpleNamespace(headers={"Retry-After": retry_after})
    return error


def client(create):
    return SimpleNamespace(api_key="test", chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class QuotaLimitTests(unittest.TestCase):
    def test_ollama_rate_limit_waits_then_retries_same_request(self):
        with tempfile.TemporaryDirectory() as directory:
            limiter = OllamaCloudUsageLimiter(Path(directory) / "state.json", limit_action="pause")
            create = Mock(side_effect=[provider_error("Too many requests"), {"choices": []}])
            proxy = OllamaCloudClientProxy(client(create), limiter)
            with patch("ollama_cloud_limiter.tracked_sleep") as sleep:
                result = proxy.chat.completions.create(model="test", messages=[])
            self.assertEqual(result, {"choices": []})
            sleep.assert_called_once_with(7.0, reason="ollama_cloud_rate_limit_wait")
            self.assertEqual(create.call_args_list[0], create.call_args_list[1])

    def test_ollama_quota_stops_even_with_legacy_wait_setting(self):
        for message in ("session usage limit reached", "weekly usage limit reached", "insufficient_quota"):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                limiter = OllamaCloudUsageLimiter(Path(directory) / "state.json", limit_action="wait")
                create = Mock(side_effect=provider_error(message))
                with patch("ollama_cloud_limiter.tracked_sleep") as sleep, self.assertRaises(QuotaExhaustedError) as error:
                    OllamaCloudClientProxy(client(create), limiter).chat.completions.create()
                self.assertEqual(error.exception.code, 2)
                create.assert_called_once()
                sleep.assert_not_called()

    def test_local_usage_guard_survives_restart_and_does_not_make_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            limiter = OllamaCloudUsageLimiter(path, session_usage_limit=1)
            limiter.record_success({})
            restarted = OllamaCloudUsageLimiter(path, session_usage_limit=1, limit_action="wait")
            create = Mock()
            with self.assertRaises(QuotaExhaustedError):
                OllamaCloudClientProxy(client(create), restarted).chat.completions.create()
            create.assert_not_called()

    def test_openrouter_all_keys_exhausted_stops_but_preserves_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            limiter = OpenRouterRateLimiter(Path(directory) / "state.json", api_keys=["test-a", "test-b"],
                                           daily_request_limit=1, daily_limit_action="wait")
            self.assertEqual(limiter.acquire(), "test-a")
            self.assertTrue(limiter.mark_key_daily_exhausted("test-a"))
            self.assertEqual(limiter.acquire(), "test-b")
            self.assertFalse(limiter.mark_key_daily_exhausted("test-b"))
            with self.assertRaises(QuotaExhaustedError):
                limiter.acquire()

    def test_openrouter_insufficient_credit_stops_without_rate_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            limiter = OpenRouterRateLimiter(Path(directory) / "state.json", api_keys=["test-a"])
            create = Mock(side_effect=provider_error("Insufficient credits", 402))
            with patch("rate_limiter.tracked_sleep") as sleep, self.assertRaises(QuotaExhaustedError):
                RateLimitedClientProxy(client(create), limiter).chat.completions.create()
            create.assert_called_once()
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
