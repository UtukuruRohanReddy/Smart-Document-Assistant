from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from smart_doc_assistant.rag_engine import SmartDocumentAssistant


APP_TITLE = "Smart Document Assistant"
DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_VERSION = "document-summaries-v3"


@st.cache_resource
def get_assistant(
    openai_api_key: str | None = None,
    cache_version: str = CACHE_VERSION,
) -> SmartDocumentAssistant:
    del cache_version
    return SmartDocumentAssistant(storage_dir=DATA_DIR, openai_api_key=openai_api_key or None)


def configured_openai_key() -> str:
    try:
        secrets_key = st.secrets.get("OPENAI_API_KEY", "")
    except (StreamlitSecretNotFoundError, KeyError):
        secrets_key = ""
    return (
        st.session_state.get("openai_api_key", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or str(secrets_key).strip()
    )


def render_source_context(source: dict) -> None:
    st.markdown(
        f"**{source.get('source', source.get('file_name', 'Unknown file'))}**"
        f" · relevance {source.get('score', 0.0):.2f}"
    )
    st.code(source.get("text", "No context available."), language="text")


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stAppViewContainer"] {background: #f7f8fc;}
        [data-testid="stHeader"] {background: rgba(247, 248, 252, 0.85);}
        [data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e8eaf2;}
        [data-testid="stFileUploader"] section {border: 1px dashed #7782ed; background: #ffffff;}
        div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e8eaf2;
          padding: 1rem 1.1rem; border-radius: 14px;}
        div.stButton > button[kind="primary"] {border-radius: 10px; font-weight: 650;}
        .stChatMessage {border: 1px solid #e8eaf2; border-radius: 14px; background: #ffffff;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    apply_app_styles()
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("document_summaries", {})

    st.title(APP_TITLE)
    st.caption(
        "Upload a document for a clear, easy-to-read summary or ask questions grounded in its content."
    )

    with st.sidebar:
        st.header("AI settings")
        st.caption(
            "OpenAI generates summaries and answers. New indexes use OpenAI embeddings when a key is configured; "
            "existing indexes keep their current embedding model."
        )
        api_key_input = st.text_input(
            "OpenAI API key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            placeholder="sk-...",
            help="The app sends document text to OpenAI for AI features. Or configure OPENAI_API_KEY in your environment or Streamlit secrets.",
        )
        if api_key_input.strip():
            st.session_state.openai_api_key = api_key_input.strip()
        elif "openai_api_key" in st.session_state:
            del st.session_state.openai_api_key
        api_key = configured_openai_key()
        if api_key:
            st.success("OpenAI is enabled for this session.")
        else:
            st.info("Add an OpenAI API key to generate AI summaries and answers. Local search will still work.")

        top_k = st.slider("Source passages per answer", min_value=1, max_value=8, value=4)
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Clearing the index removes all indexed documents.")
        clear_button = st.button("Clear document index", use_container_width=True)

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
            assistant = get_assistant(openai_api_key=api_key)
            assistant.clear_documents()
            st.session_state.messages = []
            st.session_state.document_summaries = {}
            st.success("Document index cleared.")
            st.rerun()
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to clear the document index: {exc}")

    assistant = get_assistant(openai_api_key=api_key)

    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one PDF, DOCX, or TXT file.")
            return

        progress = st.progress(0)
        status = st.empty()

        processed_count = 0
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                safe_name = Path(uploaded_file.name).name
                if not safe_name:
                    raise ValueError("The uploaded file has no valid filename.")
                saved_path = DATA_DIR / safe_name
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                with saved_path.open("wb") as handle:
                    handle.write(uploaded_file.getvalue())

                result = assistant.process_document(saved_path, file_name=safe_name)
                st.session_state.document_summaries.pop(safe_name, None)
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"View {len(message['sources'])} exact source passage(s)"):
                    for source in message["sources"]:
                        render_source_context(source)

    user_question = st.chat_input(
        "Ask about the selected documents...",
        disabled=not document_names or not selected_documents,
    )
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        try:
            result = assistant.answer_question(
                user_question,
                top_k=top_k,
                file_names=selected_documents,
            )
            answer = result["answer"]
            sources = result["sources"]
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander(f"View {len(sources)} exact source passage(s)"):
                    for source in sources:
                        render_source_context(source)
        except Exception as exc:
            error_message = f"I couldn't answer that question from the indexed document: {exc}"
            st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})
            with st.chat_message("assistant"):
                st.markdown(error_message)


if __name__ == "__main__":
    main()
