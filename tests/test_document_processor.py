from pathlib import Path
import tempfile
import unittest

from docx import Document

from smart_doc_assistant.document_processor import (
    chunk_text,
    extract_text_from_file,
)


class DocumentProcessorTests(unittest.TestCase):
    def test_extracts_txt_and_docx(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_file = root / "notes.txt"
            text_file.write_text("A plain text document.", encoding="utf-8")
            self.assertEqual(
                extract_text_from_file(text_file),
                "A plain text document.",
            )

            docx_file = root / "notes.docx"
            document = Document()
            document.add_paragraph("A word processing document.")
            document.save(docx_file)
            self.assertEqual(
                extract_text_from_file(docx_file),
                "A word processing document.",
            )

    def test_rejects_unsupported_files(self):
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            extract_text_from_file("notes.csv")

    def test_chunking_preserves_all_words_and_respects_size(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        chunks = chunk_text(text, chunk_size=24, overlap=5)

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 24 for chunk in chunks))
        reconstructed = chunks[0].split()
        for chunk in chunks[1:]:
            words = chunk.split()
            shared = min(len(reconstructed), len(words))
            while shared and reconstructed[-shared:] != words[:shared]:
                shared -= 1
            reconstructed.extend(words[shared:])
        self.assertEqual(" ".join(reconstructed), text)

    def test_chunking_validates_parameters(self):
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            chunk_text("text", chunk_size=0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_text("text", chunk_size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()
