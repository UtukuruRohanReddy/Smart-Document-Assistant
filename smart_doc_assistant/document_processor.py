from __future__ import annotations

import re
from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from pypdf import PdfReader


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def extract_text_from_pdf(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)

    if not pages:
        raise ValueError("No text could be extracted from the PDF.")

    return clean_text("\n\n".join(pages))


def extract_text_from_docx(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {path}")

    doc = DocxDocument(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    if not paragraphs:
        raise ValueError("No text could be extracted from the DOCX file.")
    return clean_text("\n\n".join(paragraphs))


def extract_text_from_txt(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise ValueError("The TXT file is empty.")
    return clean_text(text)


def extract_text_from_file(file_path: str | Path) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    if suffix == ".txt":
        return extract_text_from_txt(path)
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    if not text.strip():
        return []

    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1.")

    text = clean_text(text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        words = paragraph.split()
        window: List[str] = []
        window_length = 0
        for word in words:
            added_length = len(word) + (1 if window else 0)
            if window and window_length + added_length > chunk_size:
                chunks.append(" ".join(window))
                overlap_words: List[str] = []
                overlap_length = 0
                for previous_word in reversed(window):
                    next_length = overlap_length + len(previous_word) + (1 if overlap_words else 0)
                    if next_length > overlap:
                        break
                    overlap_words.insert(0, previous_word)
                    overlap_length = next_length
                window = overlap_words
                window_length = overlap_length

            word_length = len(word) + (1 if window else 0)
            if word_length > chunk_size:
                if window:
                    chunks.append(" ".join(window))
                    window = []
                    window_length = 0
                chunks.append(word)
            else:
                window.append(word)
                window_length += word_length

        if window:
            current = " ".join(window)

    if current:
        chunks.append(current)

    return [clean_text(chunk) for chunk in chunks if chunk.strip()]
