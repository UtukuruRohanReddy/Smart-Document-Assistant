import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from smart_doc_assistant.vector_store import (
    GEMINI_EMBEDDING_BATCH_SIZE,
    GEMINI_EMBEDDING_DIMENSION,
    GEMINI_REQUEST_TIMEOUT_MS,
    EmbeddingProvider,
)


class EmbeddingProviderTests(unittest.TestCase):
    @patch("google.genai.Client")
    def test_gemini_embedding_batches_large_inputs(self, client_factory):
        client = client_factory.return_value

        def embed_content(*, model, contents, config):
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[1.0, float(index + 1)])
                    for index, _ in enumerate(contents)
                ]
            )

        client.models.embed_content.side_effect = embed_content
        provider = EmbeddingProvider(gemini_api_key="test-key")
        texts = [f"chunk {index}" for index in range(GEMINI_EMBEDDING_BATCH_SIZE + 1)]

        result = provider.embed_texts(texts)

        self.assertEqual(result.shape, (len(texts), 2))
        self.assertEqual(client.models.embed_content.call_count, 2)
        self.assertEqual(
            client.models.embed_content.call_args_list[0].kwargs["model"],
            "gemini-embedding-001",
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(result, axis=1)[0]),
            1.0,
            places=5,
        )

    @patch("google.genai.Client")
    def test_existing_local_index_uses_local_embedding_with_gemini_key(self, client_factory):
        provider = EmbeddingProvider(gemini_api_key="test-key")
        expected = np.ones((1, 384), dtype="float32")
        provider._embed_locally = lambda texts: expected

        result = provider.embed_texts(["question"], target_dimension=384)

        self.assertEqual(result.shape, (1, 384))
        provider.client.models.embed_content.assert_not_called()

    @patch("google.genai.Client")
    def test_gemini_query_embedding_uses_retrieval_query_task(self, client_factory):
        provider = EmbeddingProvider(gemini_api_key="test-key")
        provider.client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1.0] * GEMINI_EMBEDDING_DIMENSION)]
        )

        result = provider.embed_query("query")

        self.assertEqual(result.shape, (GEMINI_EMBEDDING_DIMENSION,))
        config = provider.client.models.embed_content.call_args.kwargs["config"]
        self.assertEqual(config.task_type, "RETRIEVAL_QUERY")
        self.assertEqual(config.output_dimensionality, GEMINI_EMBEDDING_DIMENSION)

    @patch("google.genai.Client")
    def test_gemini_client_uses_bounded_timeout_without_retries(self, client_factory):
        EmbeddingProvider(gemini_api_key="test-key")

        options = client_factory.call_args.kwargs["http_options"]
        self.assertGreaterEqual(options.timeout, 10_000)
        self.assertEqual(options.timeout, GEMINI_REQUEST_TIMEOUT_MS)
        self.assertEqual(options.retry_options.attempts, 1)


if __name__ == "__main__":
    unittest.main()
