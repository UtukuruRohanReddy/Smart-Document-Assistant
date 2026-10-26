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
            store.add_document("large-dimension.txt", ["passage"], vectors)

            store.clear()

            self.assertEqual(store._index.d, 1536)
            self.assertEqual(store._index.ntotal, 0)
            reloaded = FAISSDocumentStore(directory)
            self.assertEqual(reloaded._index.d, 1536)

    def test_text_search_ranks_query_terms_and_respects_document_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FAISSDocumentStore(directory)
            store.add_document(
                "objectives.txt",
                [
                    "The project objectives are to automate exams and grading.",
                    "The interface uses a clean layout.",
                ],
                np.asarray([[1, 0], [0, 1]], dtype="float32"),
            )
            store.add_document(
                "other.txt",
                ["A separate document describes deployment."],
                np.asarray([[0.5, 0.5]], dtype="float32"),
            )

            results = store.search_text(
                "What are the project objectives?",
                k=2,
                file_names=["objectives.txt"],
            )

            self.assertEqual(
                [item["text"] for item in results],
                ["The project objectives are to automate exams and grading."],
            )
            self.assertGreater(results[0]["score"], 0)

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

    def test_blocked_faiss_legacy_index_uses_persistent_text_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "faiss_index.index").write_bytes(b"blocked legacy index")
            (root / "chunks_metadata.json").write_text(
                '[{"id":0,"file_name":"notes.txt","text":"Important project objectives",'
                '"source":"notes.txt#chunk-1"}]',
                encoding="utf-8",
            )

            store = FAISSDocumentStore(root)
            self.assertFalse(store.vector_search_available)
            self.assertEqual(
                store.search_text("project objectives")[0]["file_name"],
                "notes.txt",
            )
            store.add_document("new.txt", ["New searchable material."], None)
            reloaded = FAISSDocumentStore(root)

            self.assertEqual(
                [item["file_name"] for item in reloaded.document_summary()],
                ["new.txt", "notes.txt"],
            )
            self.assertEqual(
                reloaded.search_text("searchable material")[0]["file_name"],
                "new.txt",
            )


if __name__ == "__main__":
    unittest.main()
