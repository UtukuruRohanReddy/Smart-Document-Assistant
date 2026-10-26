from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from smart_doc_assistant.rag_engine import SmartDocumentAssistant


class AnswerFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assistant = SmartDocumentAssistant(storage_dir=Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fallback_returns_the_sentence_matching_the_question(self):
        matches = [
            {
                "text": (
                    "The product supports document search and local summaries. "
                    "The project uses Python and Streamlit for its interface."
                ),
                "source": "overview.txt#chunk-1",
                "score": 0.7,
            }
        ]

        answer = self.assistant._fallback_answer("Which interface technologies are used?", matches)

        self.assertIn("Python and Streamlit", answer)
        self.assertIn("[overview.txt#chunk-1]", answer)
        self.assertNotIn("same", answer.lower())

    def test_fallback_changes_answer_for_different_questions(self):
        matches = [
            {
                "text": (
                    "The project uses Python and Streamlit for its interface. "
                    "The search feature indexes documents with FAISS."
                ),
                "source": "overview.txt#chunk-1",
                "score": 0.7,
            }
        ]

        platform_answer = self.assistant._fallback_answer(
            "Which technologies build the interface?", matches
        )
        search_answer = self.assistant._fallback_answer(
            "What indexes documents for search?", matches
        )

        self.assertIn("Python and Streamlit", platform_answer)
        self.assertIn("FAISS", search_answer)
        self.assertNotEqual(platform_answer, search_answer)

    def test_fallback_handles_passages_without_readable_sentences(self):
        answer = self.assistant._fallback_answer(
            "What is this about?",
            [{"text": "", "source": "empty.txt#chunk-1", "score": 0.5}],
        )

        self.assertEqual(
            answer,
            "I couldn't find a readable sentence in the matching passages.",
        )

    def test_question_answer_includes_a_summary_of_the_selected_document(self):
        file_name = "objectives.txt"
        self.assistant.store.add_document(
            file_name,
            ["The project portal tracks submissions."],
            np.asarray([[1.0, 0.0, 0.0]], dtype="float32"),
        )
        self.assistant.embedding_provider.embed_query = lambda *_args, **_kwargs: np.asarray(
            [1.0, 0.0, 0.0], dtype="float32"
        )

        result = self.assistant.answer_question(
            "What is the objective?",
            file_names=[file_name],
            summary_context=[
                {
                    "file_name": file_name,
                    "overview": "The objective is to automate examination evaluation.",
                    "key_points": ["Reduce manual grading."],
                }
            ],
        )

        self.assertIn("automate examination evaluation", result["context"])
        self.assertIn(f"{file_name}#summary", [source["source"] for source in result["sources"]])
        self.assertIn("automate examination evaluation", result["answer"])

    def test_summary_can_answer_when_question_has_no_matching_chunk_terms(self):
        file_name = "objectives.txt"
        self.assistant.store.add_document(
            file_name,
            ["A student portal lets students submit their work."],
            np.ones((1, 384), dtype="float32"),
        )

        result = self.assistant.answer_question(
            "Give me a short overview.",
            file_names=[file_name],
            summary_context=[
                {
                    "file_name": file_name,
                    "overview": "The project automates examination evaluation.",
                    "key_points": ["It reduces manual grading."],
                }
            ],
        )

        self.assertIn("automates examination evaluation", result["answer"])
        self.assertIn(f"{file_name}#summary", [source["source"] for source in result["sources"]])

    def test_local_index_chat_does_not_load_embedding_model_for_questions(self):
        file_name = "objectives.txt"
        self.assistant.store.add_document(
            file_name,
            ["The project objectives are to automate exams and grading."],
            np.ones((1, 384), dtype="float32"),
        )
        self.assistant.embedding_provider.embed_query = lambda *_args, **_kwargs: (
            self.fail("Local question search should not load the embedding model.")
        )

        result = self.assistant.answer_question(
            "What are the project objectives?",
            file_names=[file_name],
        )

        self.assertIn("automate exams and grading", result["answer"])

    def test_answer_context_is_not_limited_by_source_display_count(self):
        file_name = "objectives.txt"
        self.assistant.store.add_document(
            file_name,
            [
                "The project objectives include automated exams.",
                "The project objectives include instant feedback.",
                "The project objectives include secure remote access.",
            ],
            np.ones((3, 384), dtype="float32"),
        )

        result = self.assistant.answer_question(
            "What are the project objectives?",
            top_k=1,
            file_names=[file_name],
        )

        chunk_sources = [
            source for source in result["sources"]
            if "#chunk-" in source["source"]
        ]
        self.assertGreater(len(chunk_sources), 1)
        self.assertIn("secure remote access", result["context"])

    def test_gemini_quota_exhaustion_falls_back_to_local_answer(self):
        file_name = "objectives.txt"
        self.assistant.store.add_document(
            file_name,
            ["The project objectives are to automate exams and grading."],
            np.ones((1, 384), dtype="float32"),
        )

        with patch.object(
            self.assistant,
            "_generate_answer",
            side_effect=RuntimeError("Gemini request failed: RESOURCE_EXHAUSTED quota"),
        ):
            result = self.assistant.answer_question(
                "What are the project objectives?",
                file_names=[file_name],
            )

        self.assertIn("automate exams and grading", result["answer"])
        self.assertIn("reached its API quota", result["ai_warning"])

    @patch("google.genai.Client")
    def test_invalid_gemini_key_falls_back_to_local_summary(self, client_factory):
        assistant = SmartDocumentAssistant(
            storage_dir=Path(self.temp_dir.name) / "invalid-key-summary",
            gemini_api_key="test-key",
        )
        assistant.store.add_document(
            "objectives.txt",
            ["The project objectives are to automate exams and grading."],
            np.ones((1, 384), dtype="float32"),
        )
        assistant.embedding_provider.client.models.generate_content.side_effect = (
            RuntimeError("Google Gemini error: API key not valid (API_KEY_INVALID)")
        )

        summary = assistant.summarize_document("objectives.txt")

        self.assertEqual(summary["method"], "Local extractive summary")
        self.assertIn("rejected the configured API key", summary["ai_warning"])
        self.assertNotIn("secret-fragment", summary["ai_warning"])
        self.assertIn("automate exams and grading", summary["overview"])

    @patch("google.genai.Client")
    def test_gemini_quota_exhaustion_falls_back_to_local_summary(self, client_factory):
        assistant = SmartDocumentAssistant(
            storage_dir=Path(self.temp_dir.name) / "summary",
            gemini_api_key="test-key",
        )
        assistant.store.add_document(
            "objectives.txt",
            ["The project objectives are to automate exams and grading."],
            np.ones((1, 384), dtype="float32"),
        )
        assistant.embedding_provider.client.models.generate_content.side_effect = (
            RuntimeError("Google Gemini request failed: RESOURCE_EXHAUSTED quota")
        )

        summary = assistant.summarize_document("objectives.txt")

        self.assertEqual(summary["method"], "Local extractive summary")
        self.assertIn("reached its API quota", summary["ai_warning"])
        self.assertIn("automate exams and grading", summary["overview"])

    @patch("google.genai.Client")
    def test_gemini_summary_timeout_falls_back_to_local_summary(self, client_factory):
        assistant = SmartDocumentAssistant(
            storage_dir=Path(self.temp_dir.name) / "timeout-summary",
            gemini_api_key="test-key",
        )
        assistant.store.add_document(
            "objectives.txt",
            ["The project objectives are to automate exams and grading."],
            np.ones((1, 384), dtype="float32"),
        )
        assistant.embedding_provider.client.models.generate_content.side_effect = (
            RuntimeError("Request timed out.")
        )

        summary = assistant.summarize_document("objectives.txt")

        self.assertEqual(summary["method"], "Local extractive summary")
        self.assertIn("could not be reached", summary["ai_warning"])
        self.assertIn("automate exams and grading", summary["overview"])

    @patch("google.genai.Client")
    def test_gemini_server_disconnect_falls_back_to_local_summary(self, client_factory):
        assistant = SmartDocumentAssistant(
            storage_dir=Path(self.temp_dir.name) / "disconnected-summary",
            gemini_api_key="test-key",
        )
        assistant.store.add_document(
            "objectives.txt",
            ["The project objectives are to automate exams and grading."],
            np.ones((1, 384), dtype="float32"),
        )
        assistant.embedding_provider.client.models.generate_content.side_effect = (
            RuntimeError("Server disconnected without sending a response.")
        )

        summary = assistant.summarize_document("objectives.txt")

        self.assertEqual(summary["method"], "Local extractive summary")
        self.assertIn("could not be reached", summary["ai_warning"])
        self.assertIn("automate exams and grading", summary["overview"])

    @patch("google.genai.Client")
    def test_gemini_503_falls_back_to_local_summary(self, client_factory):
        assistant = SmartDocumentAssistant(
            storage_dir=Path(self.temp_dir.name) / "unavailable-summary",
            gemini_api_key="test-key",
        )
        assistant.store.add_document(
            "objectives.txt",
            ["The project objectives are to automate exams and grading."],
            np.ones((1, 384), dtype="float32"),
        )
        assistant.embedding_provider.client.models.generate_content.side_effect = (
            RuntimeError(
                "503 UNAVAILABLE. This model is currently experiencing high demand."
            )
        )

        summary = assistant.summarize_document("objectives.txt")

        self.assertEqual(summary["method"], "Local extractive summary")
        self.assertIn("temporarily unavailable or busy", summary["ai_warning"])
        self.assertIn("automate exams and grading", summary["overview"])

    @patch("google.genai.Client")
    def test_recent_gemini_failure_skips_api_retry_during_cooldown(self, client_factory):
        assistant = SmartDocumentAssistant(
            storage_dir=Path(self.temp_dir.name) / "cooldown-summary",
            gemini_api_key="test-key",
        )
        assistant.store.add_document(
            "objectives.txt",
            ["The project objectives are to automate exams and grading."],
            np.ones((1, 384), dtype="float32"),
        )
        assistant.embedding_provider.client.models.generate_content.side_effect = (
            RuntimeError("503 UNAVAILABLE. Model is experiencing high demand.")
        )

        first_summary = assistant.summarize_document("objectives.txt")
        second_summary = assistant.summarize_document("objectives.txt")

        self.assertEqual(first_summary["method"], "Local extractive summary")
        self.assertEqual(second_summary["method"], "Local extractive summary")
        self.assertIn("temporarily unavailable or busy", second_summary["ai_warning"])
        assistant.embedding_provider.client.models.generate_content.assert_called_once()

    @patch("google.genai.Client")
    def test_gemini_chat_includes_recent_history_and_summary_context(self, client_factory):
        client = client_factory.return_value
        client.models.generate_content.return_value = SimpleNamespace(
            text="It automates grading."
        )
        assistant = SmartDocumentAssistant(
            storage_dir=Path(self.temp_dir.name) / "gemini",
            gemini_api_key="test-key",
        )
        matches = [
            {
                "source": "exam.txt#summary",
                "text": "The objective is to automate grading.",
            }
        ]

        answer = assistant._generate_answer(
            "What is the objective?",
            matches,
            [{"role": "user", "content": "Summarize the project."}],
        )

        self.assertEqual(answer, "It automates grading.")
        prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("User: Summarize the project.", prompt)
        self.assertIn("exam.txt#summary", prompt)


if __name__ == "__main__":
    unittest.main()
