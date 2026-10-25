from pathlib import Path
import tempfile
import unittest

import numpy as np

from smart_doc_assistant.vector_store import FAISSDocumentStore


class VectorStoreTests(unittest.TestCase):
    def test_search_filter_remove_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FAISSDocumentStore(directory)
            store.add_document(
                "first.txt",
                ["first passage", "second passage"],
                np.asarray([[1, 0], [0, 1]], dtype="float32"),
            )
            store.add_document(
                "second.txt",
                ["third passage"],
                np.asarray([[0.9, 0.1]], dtype="float32"),
            )

            results = store.search(
                np.asarray([1, 0], dtype="float32"),
                k=2,
                file_names=["second.txt"],
            )
            self.assertEqual([item["text"] for item in results], ["third passage"])

            self.assertTrue(store.remove_document("first.txt"))
            reloaded = FAISSDocumentStore(directory)
            self.assertEqual(
                reloaded.document_summary(),
                [{"file_name": "second.txt", "chunks": 1}],
            )
            self.assertEqual(
                reloaded.search(np.asarray([1, 0], dtype="float32"))[0]["text"],
                "third passage",
            )

    def test_clear_preserves_active_embedding_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FAISSDocumentStore(directory)
            vectors = np.zeros((1, 1536), dtype="float32")
            vectors[0, 0] = 1
            store.add_document("openai.txt", ["passage"], vectors)

            store.clear()

            self.assertEqual(store._index.d, 1536)
            self.assertEqual(store._index.ntotal, 0)
            reloaded = FAISSDocumentStore(directory)
            self.assertEqual(reloaded._index.d, 1536)

    def test_index_metadata_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FAISSDocumentStore(directory)
            store.add_document(
                "notes.txt",
                ["passage"],
                np.asarray([[1, 0]], dtype="float32"),
            )
            store.metadata_path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "do not match"):
                FAISSDocumentStore(directory)

    def test_malformed_metadata_is_reported_without_resetting(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "chunks_metadata.json"
            malformed = "{"
            metadata_path.write_text(malformed, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "corrupted"):
                FAISSDocumentStore(directory)
            self.assertEqual(metadata_path.read_text(encoding="utf-8"), malformed)


if __name__ == "__main__":
    unittest.main()
