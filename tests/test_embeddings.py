import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from smart_doc_assistant.vector_store import (
    OPENAI_EMBEDDING_BATCH_SIZE,
    EmbeddingProvider,
)


class EmbeddingProviderTests(unittest.TestCase):
    @patch("openai.OpenAI")
    def test_openai_embedding_batches_large_inputs(self, openai_factory):
        provider = EmbeddingProvider(openai_api_key="test-key")
        create = provider.client.embeddings.create

        def create_embeddings(*, model, input):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[1.0, float(index + 1)])
                    for index, _ in enumerate(input)
                ]
            )

        create.side_effect = create_embeddings
        texts = [f"chunk {index}" for index in range(OPENAI_EMBEDDING_BATCH_SIZE + 1)]

        result = provider.embed_texts(texts)

        self.assertEqual(result.shape, (len(texts), 2))
        self.assertEqual(create.call_count, 2)
        self.assertAlmostEqual(
            float(np.linalg.norm(result, axis=1)[0]),
            1.0,
            places=5,
        )

    @patch("openai.OpenAI")
    def test_existing_local_index_uses_local_embedding_with_openai_key(self, openai_factory):
        provider = EmbeddingProvider(openai_api_key="test-key")
        expected = np.ones((1, 384), dtype="float32")
        provider._embed_locally = lambda texts: expected

        result = provider.embed_texts(
            ["question"],
            target_dimension=384,
        )

        self.assertEqual(result.shape, (1, 384))
        provider.client.embeddings.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
