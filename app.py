"""
Healthcare Symptom Information Assistant — Main Streamlit Application
Powered by a configurable LLM endpoint · RAG · LangChain · FAISS
"""
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Healthcare Symptom Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ────────────────────────────────────────────────────────────────────
from config import cfg
from llm.model_loader import get_llm
from rag.vector_store import build_or_load_vector_store
from rag.retriever import build_rag_chain, retrieve_sources
from ui.sidebar import render_sidebar
from ui.chat_interface import init_chat_state, render_chat_history, add_message
from utils.safety_filter import sanitize_response
from logger_config import get_logger, log_section, log_step, log_success, log_error

logger = get_logger(__name__)


# ── Cached resource loaders ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Connecting to configured LLM endpoint…")
def load_llm():
    log_step(logger, 1, "Initialising LLM client")
    return get_llm()


@st.cache_resource(show_spinner="🔄 Building knowledge index…")
def load_vector_store():
    log_step(logger, 2, "Building / loading FAISS vector store")
    return build_or_load_vector_store()


@st.cache_resource(show_spinner="🔗 Assembling RAG pipeline…")
def load_rag_chain(_llm, _vector_store):
    log_step(logger, 3, "Assembling RAG chain")
    return build_rag_chain(_llm, _vector_store)


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    # Initialise chat state
    init_chat_state()

    # Render sidebar
    render_sidebar()

    # ── Header ──
    st.markdown(f"# {cfg.app_title}")
    st.caption(
        f"General health information assistant · Powered by {cfg.llm_model} via configured LLM endpoint + RAG · "
        "Sources: WHO / CDC · **Not a substitute for medical advice**"
    )
    st.divider()

    # ── Load backend resources (cached after first call) ──
    llm          = load_llm()
    vector_store = load_vector_store()
    rag_chain    = load_rag_chain(llm, vector_store)

    # ── Render chat history ──
    render_chat_history()

    # ── Check for prefilled query from sidebar structured input ──
    prefilled  = st.session_state.pop("prefilled_query", None)
    user_input = st.chat_input(
        "Describe your symptoms here… (e.g. 'I have a persistent cough and fever for 3 days')",
        key="chat_input",
    )

    # Allow sidebar-built query to auto-submit
    if prefilled and not user_input:
        user_input = prefilled

    if user_input:
        # Show user message immediately
        add_message("user", user_input)
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("🔍 Searching health knowledge base and generating response…"):
                try:
                    logger.info(f"Processing query: '{user_input[:80]}...'")

                    # Run RAG chain
                    result       = rag_chain.invoke({"query": user_input})
                    raw_response = result.get("result", "")

                    # Retrieve sources for display
                    sources      = retrieve_sources(vector_store, user_input)
                    safe_response = sanitize_response(raw_response)

                    log_success(logger, f"Response generated ({len(safe_response)} chars, {len(sources)} sources)")

                    # Display response
                    st.markdown(safe_response)

                    # Show sources
                    if sources:
                        with st.expander("📚 View retrieved health knowledge sources"):
                            for i, src in enumerate(sources, 1):
                                meta  = src.get("metadata", {})
                                label = (
                                    meta.get("condition")
                                    or meta.get("symptom")
                                    or meta.get("category")
                                    or meta.get("source_file", "Health Data")
                                )
                                st.markdown(f"**Source {i} — {label}**")
                                st.caption(src["content"])
                                st.divider()

                    # Persist to session state
                    add_message("assistant", safe_response, sources)

                except Exception as e:
                    log_error(logger, f"Error processing query: {e}", e)
                    error_msg = (
                        f"⚠️ An error occurred while generating the response: `{e}`\n\n"
                        "Please try rephrasing your question or verify the configured LLM endpoint settings."
                    )
                    st.error(error_msg)
                    add_message("assistant", error_msg)

    # ── Footer ──
    st.divider()
    st.markdown(
        "<div style='text-align:center; color:grey; font-size:0.8em;'>"
        "🩺 Healthcare Symptom Information Assistant · "
        "This tool provides general health information only · "
        "Always consult a qualified healthcare professional · "
        "Sources: WHO / CDC"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    log_section(logger, "Healthcare Symptom Information Assistant — Starting")
    main()

