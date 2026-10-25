import json
import unittest
from types import SimpleNamespace

from smart_doc_assistant.document_summarizer import (
    extractive_summary,
    summarize_chunks,
)


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


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
            chat=SimpleNamespace(completions=completions),
        )

        summary = summarize_chunks([self.text], client=client)

        self.assertEqual(summary["method"], "AI summary")
        self.assertEqual(len(summary["key_points"]), 1)
        self.assertEqual(
            completions.calls[0]["response_format"],
            {"type": "json_object"},
        )

    def test_invalid_ai_summary_raises_clear_error(self):
        completions = FakeCompletions('{"overview": "", "key_points": []}')
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        with self.assertRaisesRegex(RuntimeError, "overview"):
            summarize_chunks([self.text], client=client)

    def test_empty_document_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no indexed text"):
            summarize_chunks([])

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
