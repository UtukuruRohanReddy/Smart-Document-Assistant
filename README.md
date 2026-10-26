# Smart Document Assistant

[![Tests](https://github.com/UtukuruRohanReddy/Smart-Document-Assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/UtukuruRohanReddy/Smart-Document-Assistant/actions/workflows/tests.yml)

Live demo: https://smart-document-assistant-8opf.onrender.com/

A Streamlit app for searching and summarizing PDF, DOCX, and TXT documents. It extracts text, creates overlapping chunks, stores embeddings locally in a NumPy index, and displays source passages for answers.

## What you can do

- Get a plain-language document overview and key takeaways.
- Ask questions about one document or search across a selected set.
- Inspect the exact retrieved passages and source labels behind each answer.
- Add or replace documents, remove individual documents, or clear the local index.

## Features

- Google Gemini-powered answers and summaries, with a local extractive summary when no key is configured.
- Local extractive summaries and grounded answer fallback when Gemini rejects the configured key, reaches quota, or cannot be reached.
- Gemini requests use the minimum supported 10-second timeout with retries disabled, so slow service responses fall back to local results promptly.
- After a Gemini API failure, the app skips repeat Gemini calls for 60 seconds and serves local results during that cooldown.
- Local sentence-transformer embeddings when no key is configured; Gemini embeddings for newly indexed documents when configured.
- NumPy-backed similarity search with bounded embedding request batches; the app does not load native FAISS DLLs.
- Text-search fallback for legacy FAISS indexes, which are preserved and never loaded by the app.
- Persistent local index and metadata, kept out of Git.

## Requirements

- Python 3.10 or newer.
- Google Gemini API key for AI-generated summaries, answers, and new document embeddings.
- Without a key, local embeddings and extractive summaries are used.

## Deployment

This app is deployed on Render:

- Live demo: https://smart-document-assistant-8opf.onrender.com/
- GitHub repository: https://github.com/UtukuruRohanReddy/Smart-Document-Assistant

For production hosting on Render, set these environment variables in the Render dashboard:

```text
GEMINI_API_KEY=your-google-gemini-api-key
SMART_DOC_DATA_DIR=/var/data/smart-doc
```

Use a Render persistent disk mounted at `/var/data` so uploaded files and the local index survive deploys and restarts.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

Create a Gemini API key at [Google AI Studio](https://aistudio.google.com/apikey). Configure `GEMINI_API_KEY` in your environment or `.streamlit/secrets.toml`; the sidebar reports whether a key is configured:

```toml
GEMINI_API_KEY = "your-key-here"
```

To use the project `.env` file, copy `.env.example` to `.env`, or open it directly with PowerShell:

```powershell
Set-Location "C:\Users\rohan\OneDrive\Desktop\Smart Document Assistant"
notepad .env
```

Add your Google AI Studio key in the project `.env` file and save it:

```dotenv
GEMINI_API_KEY=your-google-gemini-api-key
```

The app reads `.env` on startup. Restart it after changing the key:

```powershell
Set-Location "C:\Users\rohan\OneDrive\Desktop\Smart Document Assistant"
python -m streamlit run app.py
```

Never commit real API keys or secrets. Relevant document text is sent to Google Gemini when its AI features are used.

## Run tests

The test suite uses Python's standard library `unittest`:

```powershell
python -m unittest discover -s tests -v
```

The same compile and unit-test checks run in GitHub Actions on pushes and pull requests to `main`.

## Data and privacy

Uploaded files and local index data are stored in the ignored `data/` directory. That directory is intentionally excluded from Git. Do not commit uploaded documents, generated indexes, metadata, API keys, or other private data.

Documents are processed locally for text extraction and NumPy index storage. When Gemini features are used, relevant document text is sent to Google to generate embeddings, answers, or summaries.

### Deployment storage

The default `data/` directory is local to the app process. Many hosted app platforms use temporary filesystems, so uploaded documents and the NumPy index may disappear after a restart or redeploy. Configure `SMART_DOC_DATA_DIR` to point to a writable persistent volume mounted by your host. For example, if the host provides a persistent volume at `/mnt/smart-doc-data`, set this environment variable in the host's app settings:

```text
SMART_DOC_DATA_DIR=/mnt/smart-doc-data
```

The app creates the directory if needed. A persistent volume must be configured by the hosting provider; setting a path alone cannot make an ephemeral deployment persistent. Keep the volume private because it contains uploaded document text and indexes.

## Notes

- The first local embedding run downloads the `sentence-transformers/all-MiniLM-L6-v2` model.
- An existing index retains its embedding model. New indexes use Gemini embeddings when a key is configured; changing the embedding provider for an existing library requires reprocessing documents.
