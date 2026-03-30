"""
Healthcare Symptom Information Assistant — Main Streamlit Application
Powered by a configurable LLM endpoint · RAG · LangChain · FAISS
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import streamlit as st
from config import cfg
from llm.model_loader import get_llm
from rag.vector_store import build_or_load_vector_store
from rag.retriever import build_rag_chain, retrieve_sources
from ui.sidebar import render_sidebar
from ui.chat_interface import init_chat_state, render_chat_history, add_message
from utils.safety_filter import sanitize_response, is_health_related, get_off_domain_message
from logger_config import get_logger, log_section, log_step, log_success, log_error

logger = get_logger(__name__)


# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title=cfg.ui_text.page.page_title,
    page_icon=cfg.ui_text.page.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached resource loaders ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=cfg.ui_text.page.load_llm_spinner)
def load_llm():
    log_step(logger, 1, "Initialising LLM client")
    return get_llm()


@st.cache_resource(show_spinner=cfg.ui_text.page.load_vector_store_spinner)
def load_vector_store():
    log_step(logger, 2, "Building / loading FAISS vector store")
    return build_or_load_vector_store()


@st.cache_resource(show_spinner=cfg.ui_text.page.load_rag_spinner)
def load_rag_chain(_llm, _vector_store):
    log_step(logger, 3, "Assembling RAG chain")
    return build_rag_chain(_llm, _vector_store)


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    # Initialise session state for multi-turn chat across reruns
    init_chat_state()

    # Render controls (endpoint mode, presets, and app options)
    render_sidebar()

    # ── Header ──
    st.markdown(f"# {cfg.ui_text.page.app_title}")
    st.caption(cfg.ui_text.page.header_caption_template.format(llm_model=cfg.llm_model))
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
        cfg.ui_text.page.chat_input_placeholder,
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
            # Off-domain check — block non-health queries before hitting RAG
            if not is_health_related(user_input):
                off_domain_msg = get_off_domain_message()
                st.warning(off_domain_msg)
                add_message("assistant", off_domain_msg)

            # On-domain query — proceed with RAG pipeline
            else:
                with st.spinner(cfg.ui_text.page.response_spinner):
                    try:
                        logger.info(
                            cfg.ui_text.page.query_log_template.format(query=f"{user_input[:80]}...")
                        )

                        # Invoke RAG pipeline (retrieval + grounded generation)
                        logger.debug(f"[TRACE] rag_chain.invoke() entry point — mode={cfg.llm_mode}")
                        result        = rag_chain.invoke({"query": user_input})
                        raw_response  = result.get("result", "")

                        # Fetch top retrieved chunks separately for transparent source display
                        sources       = retrieve_sources(vector_store, user_input)
                        safe_response = sanitize_response(raw_response)

                        log_success(logger, f"Response generated ({len(safe_response)} chars, {len(sources)} sources)")

                        # Display response
                        st.markdown(safe_response)

                        # Show sources
                        if sources and cfg.show_retrieved_sources:
                            with st.expander(cfg.ui_text.page.source_expander_label):
                                for i, src in enumerate(sources, 1):
                                    meta  = src.get("metadata", {})
                                    label = (
                                        meta.get("condition")
                                        or meta.get("symptom")
                                        or meta.get("category")
                                        or meta.get("source_file", cfg.ui_text.page.source_fallback_label)
                                    )
                                    st.markdown(f"**Source {i} — {label}**")
                                    st.caption(src["content"])
                                    st.divider()

                        # Persist assistant output and sources for chat history replay
                        add_message("assistant", safe_response, sources)

                    except Exception as e:
                        log_error(logger, f"Error processing query: {e}", e)
                        error_msg = cfg.ui_text.page.error_message_template.format(error=e)
                        st.error(error_msg)
                        add_message("assistant", error_msg)

    # ── Footer ──
    st.divider()
    st.markdown(
        cfg.ui_text.page.footer_html,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    log_section(logger, "Healthcare Symptom Information Assistant — Starting")
    main()
