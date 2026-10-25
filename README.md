# Smart Document Assistant

[![Tests](https://github.com/UtukuruRohanReddy/Smart-Document-Assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/UtukuruRohanReddy/Smart-Document-Assistant/actions/workflows/tests.yml)

A Streamlit app for searching and summarizing PDF, DOCX, and TXT documents. It extracts text, creates overlapping chunks, stores embeddings locally in FAISS, and displays source passages for answers.

## What you can do

- Get a plain-language document overview and key takeaways.
- Ask questions about one document or search across a selected set.
- Inspect the exact retrieved passages and source labels behind each answer.
- Add or replace documents, remove individual documents, or clear the local index.

## Features

- OpenAI-powered answers and summaries, with a local extractive summary when no key is configured.
- Local sentence-transformer embeddings when no key is configured; OpenAI embeddings for newly indexed documents when configured.
- FAISS similarity search with bounded embedding request batches.
- Persistent local index and metadata, kept out of Git.

## Requirements

- Python 3.10 or newer.
- OpenAI API key for AI-generated summaries and answers. OpenAI embeddings are used for newly indexed documents when a key is configured.
- Without a key, local embeddings and extractive summaries are used.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

Enter an OpenAI API key in the app sidebar to enable AI summaries and answers. You can also set `OPENAI_API_KEY` in your environment or in `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-key-here"
```

Never commit real API keys or secrets. The key is used to send document text to OpenAI for the requested AI features.

## Run tests

The test suite uses Python's standard library `unittest`:

```powershell
python -m unittest discover -s tests -v
```

The same compile and unit-test checks run in GitHub Actions on pushes and pull requests to `main`.

## Data and privacy

Uploaded files and local FAISS data are stored in the ignored `data/` directory. That directory is intentionally excluded from Git. Do not commit uploaded documents, generated indexes, metadata, API keys, or other private data.

Documents are processed locally for text extraction and FAISS storage. When OpenAI features are used, relevant document text is sent to OpenAI to generate embeddings, answers, or summaries.

## Notes

- The first local embedding run downloads the `sentence-transformers/all-MiniLM-L6-v2` model.
- An existing index retains its embedding model. Clear the index before intentionally changing the embedding provider for the full library.
