from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from smart_doc_assistant.rag_engine import SmartDocumentAssistant


APP_TITLE = "Smart Document Assistant"
load_dotenv(Path(__file__).resolve().parent / ".env")
os.environ.pop("GOOGLE_API_KEY", None)
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("SMART_DOC_DATA_DIR", str(APP_DIR / "data"))).expanduser().resolve()
CACHE_VERSION = "gemini-document-summaries-v6-gemini-deadline"


def unique_document_name(file_name: str, existing_names: set[str] | None = None) -> str:
    """Return a sanitized, collision-safe document name."""
    candidate = Path(file_name).name.strip()
    if not candidate:
        raise ValueError("The uploaded file has no valid filename.")

    root = Path(candidate)
    stem = root.stem or root.name
    suffix = root.suffix
    safe_stem = re.sub(r"[^A-Za-z0-9_. -]+", "_", stem).strip()
    if not safe_stem:
        safe_stem = "document"
    safe_name = f"{safe_stem}{suffix}"

    existing = existing_names or set()
    counter = 2
    while safe_name in existing:
        safe_name = f"{safe_stem} ({counter}){suffix}"
        counter += 1
    return safe_name


@st.cache_resource
def get_assistant(
    gemini_api_key: str | None = None,
    storage_dir: str = str(DATA_DIR),
    cache_version: str = CACHE_VERSION,
) -> SmartDocumentAssistant:
    del cache_version
    return SmartDocumentAssistant(
        storage_dir=storage_dir,
        gemini_api_key=gemini_api_key or None,
    )


def configured_gemini_key() -> str:
    try:
        secrets_key = st.secrets.get("GEMINI_API_KEY", "")
    except (StreamlitSecretNotFoundError, KeyError):
        secrets_key = ""
    return (
        os.getenv("GEMINI_API_KEY", "").strip()
        or str(secrets_key).strip()
    )


def render_source_context(source: dict) -> None:
    st.markdown(
        f"**{source.get('source', source.get('file_name', 'Unknown file'))}**"
        f" · relevance {source.get('score', 0.0):.2f}"
    )
    st.code(source.get("text", "No context available."), language="text")


def apply_app_styles(theme: str = "Light") -> None:
    palette = {
        "Light": {
            "bg": "#f1f5f9",
            "panel": "#ffffff",
            "panel_alt": "#f8fafc",
            "border": "#dbe3ef",
            "text": "#172033",
            "muted": "#596579",
            "accent": "#4f46e5",
            "accent_soft": "#eef2ff",
        },
        "Dark": {
            "bg": "#0f172a",
            "panel": "#111827",
            "panel_alt": "#172033",
            "border": "#2b3a4f",
            "text": "#edf2ff",
            "muted": "#b9c7e4",
            "accent": "#9aa8ff",
            "accent_soft": "#1d2541",
        },
        "Midnight": {
            "bg": "#050816",
            "panel": "#0b1220",
            "panel_alt": "#121c2c",
            "border": "#2a3550",
            "text": "#edf4ff",
            "muted": "#aebbd7",
            "accent": "#6ed6ff",
            "accent_soft": "#101d2a",
        },
    }
    colors = palette.get(theme, palette["Light"])
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {colors['bg']};
            --panel-bg: {colors['panel']};
            --panel-alt: {colors['panel_alt']};
            --border-color: {colors['border']};
            --text-color: {colors['text']};
            --muted-color: {colors['muted']};
            --accent-color: {colors['accent']};
            --accent-soft: {colors['accent_soft']};
        }}
        .block-container {{max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}}
        [data-testid="stAppViewContainer"] {{background: var(--app-bg); color: var(--text-color);}}
        [data-testid="stHeader"] {{background: rgba(0,0,0,0.12);}}
        [data-testid="stSidebar"] {{background: var(--panel-bg); border-right: 1px solid var(--border-color);}}
        [data-testid="stFileUploader"] section {{border: 1px dashed var(--accent-color); background: var(--panel-alt);}}
        div[data-testid="stMetric"] {{background: var(--panel-bg); border: 1px solid var(--border-color); padding: 1rem 1.1rem; border-radius: 14px;}}
        div.stButton > button[kind="primary"] {{border-radius: 10px; font-weight: 650; background: var(--accent-color); border-color: var(--accent-color); color: #ffffff;}}
        div.stButton > button {{border-radius: 10px;}}
        .stChatMessage {{border: 1px solid var(--border-color); border-radius: 14px; background: var(--panel-bg); color: var(--text-color);}}
        .stExpander {{border: 1px solid var(--border-color); border-radius: 12px; background: var(--panel-bg);}}
        .stCode, pre {{background: var(--panel-alt); color: var(--text-color);}}
        .stAlert, .stInfo, .stSuccess, .stWarning, .stError {{border-radius: 12px;}}
        .stMarkdown {{color: var(--text-color);}}
        .stCaption {{color: var(--muted-color);}}
        h1, h2, h3, h4, h5, h6 {{color: var(--text-color);}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("document_summaries", {})

    st.title(APP_TITLE)
    st.caption(
        "Upload a document for a clear, easy-to-read summary or ask questions grounded in its content."
    )

    with st.sidebar:
        st.header("AI settings")
        st.caption(
            "Google Gemini generates summaries and answers. New indexes use Gemini embeddings when a key is configured; "
            "existing indexes keep their current embedding model."
        )
        st.subheader("Appearance")
        theme_options = ["Light", "Dark", "Midnight"]
        st.session_state.setdefault("app_theme_v2", "Dark")
        if st.session_state.app_theme_v2 not in theme_options:
            st.session_state.app_theme_v2 = "Dark"
        theme = st.selectbox(
            "Theme",
            options=theme_options,
            key="app_theme_v2",
            help="Choose a visual theme for the app.",
        )

        api_key = configured_gemini_key()
        if api_key:
            st.success("Google Gemini API key is available.")
        else:
            st.warning("Google Gemini API key is unavailable.")
        st.caption("Availability checks configuration only; Google validates the key when an AI feature is used.")

        assistant = get_assistant(gemini_api_key=api_key)

        top_k = st.slider("Source passages per answer", min_value=1, max_value=8, value=4)
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Clearing the index removes all indexed documents.")
        clear_button = st.button("Clear document index", use_container_width=True)

    apply_app_styles(theme)

    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Select one or more documents, then process them to make their contents searchable.",
    )
    upload_col, process_col = st.columns([3, 1])
    with upload_col:
        if uploaded_files:
            names = ", ".join(Path(item.name).name for item in uploaded_files)
            st.caption(f"Selected: {names}")
    with process_col:
        process_button = st.button(
            "Process documents",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )

    if clear_button:
        try:
            assistant = get_assistant(gemini_api_key=api_key)
            assistant.clear_documents()
            st.session_state.messages = []
            st.session_state.document_summaries = {}
            st.success("Document index cleared.")
            st.rerun()
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to clear the document index: {exc}")

    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one PDF, DOCX, or TXT file.")
            return

        progress = st.progress(0)
        status = st.empty()

        existing_names = {item["file_name"] for item in assistant.document_summary()}
        processed_count = 0
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                safe_name = unique_document_name(uploaded_file.name, existing_names)
                saved_path = DATA_DIR / safe_name
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                with saved_path.open("wb") as handle:
                    handle.write(uploaded_file.getvalue())

                result = assistant.process_document(saved_path, file_name=safe_name)
                st.session_state.document_summaries.pop(safe_name, None)
                existing_names.add(safe_name)
                processed_count += 1
                status.info(
                    f"Processed {index}/{len(uploaded_files)}: {safe_name} "
                    f"({result['chunks']} passages)"
                )
            except Exception as exc:
                st.error(f"Could not process {uploaded_file.name}: {exc}")
            progress.progress(index / len(uploaded_files))

        failed_count = len(uploaded_files) - processed_count
        if processed_count and not failed_count:
            st.success(f"Ready to search {processed_count} document(s).")
        elif processed_count:
            st.warning(
                f"Indexed {processed_count} document(s); {failed_count} could not be processed. "
                "Review the errors above."
            )

    documents = assistant.document_summary()
    if documents:
        st.subheader("Your document library")
        total_chunks = sum(item["chunks"] for item in documents)
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Documents", len(documents))
        metric_col2.metric("Searchable passages", total_chunks)

        with st.expander("Manage indexed documents", expanded=True):
            for item in documents:
                row_col, summary_col, remove_col = st.columns([5, 1.5, 1])
                row_col.markdown(
                    f"**{item['file_name']}**  \n{item['chunks']} searchable passages"
                )
                if summary_col.button(
                    "Summarize",
                    key=f"summarize-{item['file_name']}",
                    help=f"Create an easy-to-read summary of {item['file_name']}",
                    use_container_width=True,
                ):
                    try:
                        with st.spinner(f"Summarizing {item['file_name']}..."):
                            st.session_state.document_summaries[item["file_name"]] = (
                                assistant.summarize_document(item["file_name"])
                            )
                    except Exception as exc:
                        st.error(f"Could not summarize {item['file_name']}: {exc}")
                if remove_col.button(
                    "Remove",
                    key=f"remove-{item['file_name']}",
                    help=f"Remove {item['file_name']} from the search index",
                ):
                    try:
                        assistant.remove_document(item["file_name"])
                        st.session_state.document_summaries.pop(item["file_name"], None)
                        st.session_state.messages = []
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not remove {item['file_name']}: {exc}")
                summary = st.session_state.document_summaries.get(item["file_name"])
                if summary:
                    with st.container(border=True):
                        st.markdown(f"#### Summary: {item['file_name']}")
                        st.caption(
                            f"{summary['method']} · Based on the indexed document text"
                        )
                        st.markdown(summary["overview"])
                        st.markdown("**Key points**")
                        for point in summary["key_points"]:
                            st.markdown(f"- {point}")

    st.subheader("Ask a question")
    document_names = [item["file_name"] for item in documents]
    selected_documents = st.multiselect(
        "Search in",
        options=document_names,
        default=document_names,
        help="Choose one or more indexed documents. Leave all selected to search the whole library.",
        disabled=not document_names,
    )
    if not document_names:
        st.info("Upload and process at least one document to start asking questions.")
    else:
        st.caption("Ask about document details or its summary. Recent messages are used for follow-up questions.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])
            if message.get("sources"):
                visible_sources = message["sources"][:top_k]
                with st.expander(f"View {len(visible_sources)} exact source passage(s)"):
                    for source in visible_sources:
                        render_source_context(source)

    question_input = st.chat_input(
        "Ask about the selected documents...",
        disabled=not document_names or not selected_documents,
    )

    if question_input and question_input.strip():
        user_question = question_input.strip()
        st.session_state.messages.append({"role": "user", "content": user_question})

        try:
            selected_summaries = [
                {
                    "file_name": file_name,
                    **st.session_state.document_summaries[file_name],
                }
                for file_name in selected_documents
                if file_name in st.session_state.document_summaries
            ]
            conversation_history = [
                message
                for message in st.session_state.messages[:-1]
                if not message.get("error")
            ][-6:]
            with st.spinner("Searching your documents and preparing an answer..."):
                result = assistant.answer_question(
                    user_question,
                    top_k=top_k,
                    file_names=selected_documents,
                    summary_context=selected_summaries,
                    conversation_history=conversation_history,
                )
            answer = result["answer"]
            sources = result["sources"]
            ai_warning = result.get("ai_warning")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "ai_warning": ai_warning,
                }
            )
        except Exception as exc:
            error_message = f"I couldn't answer that question from the indexed document: {exc}"
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "sources": [], "error": True}
            )
        st.rerun()


if __name__ == "__main__":
    main()
