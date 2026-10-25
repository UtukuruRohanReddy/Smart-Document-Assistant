# Smart Document Assistant

A Streamlit app for searching and summarizing PDF, DOCX, and TXT documents. It extracts text, creates overlapping chunks, stores embeddings locally in FAISS, and displays source passages for answers.

## Features

- Upload and index PDF, DOCX, and TXT files.
- Search one or more indexed documents with natural-language questions.
- Summarize a document in plain language with an overview and key points.
- Use OpenAI for generated answers, summaries, and embeddings, or use local sentence-transformer embeddings for search.
- View exact retrieved source snippets, and manage individual indexed documents.

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

## Data and privacy

Uploaded files and local FAISS data are stored in the ignored `data/` directory. That directory is intentionally excluded from Git. Do not commit uploaded documents, generated indexes, metadata, API keys, or other private data.

## Notes

- The first local embedding run downloads the `sentence-transformers/all-MiniLM-L6-v2` model.
- An existing index retains its embedding dimension. To switch an existing library between local and OpenAI embeddings, clear the index and reprocess the documents.
