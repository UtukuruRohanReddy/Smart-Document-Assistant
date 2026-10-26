import json
import unittest
from types import SimpleNamespace

from smart_doc_assistant.document_summarizer import (
    ai_fallback_warning,
    extractive_summary,
    is_ai_quota_error,
    summarize_chunks,
)


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.content)


class DocumentSummarizerTests(unittest.TestCase):
    def setUp(self):
        self.text = (
            "The service reduced processing time by 40 percent. "
            "Customers can track requests from one dashboard. "
            "The team plans multilingual support next year."
        )

    def test_local_summary_contains_overview_and_points(self):
        summary = extractive_summary(self.text)

        self.assertTrue(summary["overview"])
        self.assertTrue(summary["key_points"])
        self.assertEqual(summary["method"], "Local extractive summary")
        self.assertTrue(all(point in self.text for point in summary["key_points"]))

    def test_ai_summary_parses_structured_response(self):
        payload = json.dumps(
            {
                "overview": "The service speeds up work and improves tracking.",
                "key_points": ["Processing time fell by 40 percent."],
            }
        )
        completions = FakeCompletions(payload)
        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=completions.generate_content),
        )

        summary = summarize_chunks([self.text], client=client)

        self.assertEqual(summary["method"], "AI summary")
        self.assertEqual(len(summary["key_points"]), 1)
        self.assertEqual(
            completions.calls[0]["config"].response_mime_type,
            "application/json",
        )

    def test_invalid_ai_summary_raises_clear_error(self):
        completions = FakeCompletions('{"overview": "", "key_points": []}')
        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=completions.generate_content),
        )

        with self.assertRaisesRegex(RuntimeError, "overview"):
            summarize_chunks([self.text], client=client)

    def test_empty_document_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no indexed text"):
            summarize_chunks([])

    def test_quota_error_is_identified_for_local_fallback(self):
        self.assertTrue(
            is_ai_quota_error(
                RuntimeError("Gemini request failed: RESOURCE_EXHAUSTED quota")
            )
        )
        self.assertFalse(is_ai_quota_error(RuntimeError("service unavailable")))

    def test_invalid_api_key_uses_safe_auth_warning(self):
        error = RuntimeError(
            "Gemini error: API key not valid (API_KEY_INVALID)"
        )

        warning = ai_fallback_warning(error)

        self.assertIsNotNone(warning)
        self.assertIn("rejected the configured API key", warning)

    def test_timeout_error_uses_connection_fallback_warning(self):
        error = RuntimeError("AI summarization failed: Request timed out.")

        warning = ai_fallback_warning(error)

        self.assertIsNotNone(warning)
        self.assertIn("could not be reached", warning)

    def test_server_disconnected_error_uses_connection_fallback_warning(self):
        error = RuntimeError("Server disconnected without sending a response.")

        warning = ai_fallback_warning(error)

        self.assertIsNotNone(warning)
        self.assertIn("could not be reached", warning)

    def test_gemini_503_unavailable_uses_busy_fallback_warning(self):
        error = RuntimeError(
            "Gemini summarization failed: 503 UNAVAILABLE. Model experiencing high demand."
        )

        warning = ai_fallback_warning(error)

        self.assertIsNotNone(warning)
        self.assertIn("temporarily unavailable or busy", warning)

    def test_gemini_gateway_deadline_uses_connection_fallback_warning(self):
        error = RuntimeError("504 DEADLINE_EXCEEDED")

        warning = ai_fallback_warning(error)

        self.assertIsNotNone(warning)
        self.assertIn("could not be reached", warning)

    def test_unrelated_ai_errors_are_not_replaced_by_local_results(self):
        self.assertIsNone(ai_fallback_warning(RuntimeError("invalid request")))

    def test_overlapping_chunks_are_deduplicated_before_summary(self):
        summary = summarize_chunks(
            [
                "The first important sentence. Shared transition sentence.",
                "Shared transition sentence. The second important sentence.",
            ]
        )

        self.assertEqual(summary["key_points"].count("Shared transition sentence."), 1)


if __name__ == "__main__":
    unittest.main()
