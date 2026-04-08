"""
Healthcare Symptom Information Assistant — Main Streamlit Application
Powered by a configurable LLM endpoint · RAG · LangChain · FAISS
"""

import time
import streamlit as st
from app.ui.styles import StreamlitStyles
from app.config.settings import cfg
from app.llm.model_loader import get_llm
from app.core.logging import get_logger, log_section, log_step, log_success, log_error
from app.core.metrics import build_metrics_entry, persist_metrics
from app.rag.retriever import build_rag_chain, retrieve_sources
from app.rag.vector_store import build_or_load_vector_store
from app.ui.chat_interface import init_chat_state, render_chat_history, add_message
from app.ui.metrics_dashboard import main as render_metrics_dashboard
from app.ui.sidebar import render_sidebar
from app.ui.source_utils import format_source_label, truncate_source_content
from app.safety.safety_filter import sanitize_response, is_health_related_with_mode, get_off_domain_message
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

    # Apply custom styles
    StreamlitStyles.apply_all_styles()

    # Track current view (chat vs. metrics dashboard) in session state.
    if "current_view" not in st.session_state:
        st.session_state["current_view"] = cfg.chat_view

    # Track active concurrent requests to provide context for performance metrics.
    if "active_requests" not in st.session_state:
        st.session_state.active_requests = 0

    # Render controls (endpoint mode, presets, and app options)
    render_sidebar()

    # In-app routing: if user navigates to metrics dashboard, render that instead of chat interface.
    if st.session_state.get("current_view") == cfg.metrics_view:
        st.markdown(cfg.ui_text.page.dashboard_app_title_html, unsafe_allow_html=True)
        st.markdown(cfg.ui_text.page.tagline_html, unsafe_allow_html=True)
        st.divider()
        render_metrics_dashboard()
        return

    # ── Header ──
    st.markdown(cfg.ui_text.page.app_title_html, unsafe_allow_html=True)
    st.markdown(cfg.ui_text.page.tagline_html, unsafe_allow_html=True)
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
        # ── Timing / metric accumulators ──────────────────────────────────────
        request_started   = time.perf_counter()
        retrieval_started = None
        retrieval_ended   = None
        generation_started = None
        generation_ended  = None
        first_token_ts    = None
        sources           = []
        safe_response     = ""
        error_message     = None

        st.session_state.active_requests += 1
        concurrent_users = st.session_state.active_requests

        # Generate unique key for this request to avoid duplicate element key errors
        request_key = f"msg-{int(request_started * 1e9)}"

        # Show user message immediately
        add_message("user", user_input)
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Assistant response block — will be populated after RAG processing.
        with st.chat_message("assistant", avatar="🩺"):

            # Off-domain check — block non-health queries before hitting RAG
            allowed, semantic_score = is_health_related_with_mode(user_input, vector_store)
            if not allowed:
                off_domain_msg = get_off_domain_message()
                st.warning(off_domain_msg)
                add_message("assistant", off_domain_msg)
                safe_response = off_domain_msg

                # Log semantic gate rejection when a score is available
                if semantic_score is not None:
                    logger.info(
                        f"Domain gate rejected query (mode={cfg.domain_filter_mode}, "
                        f"score={semantic_score:.4f}, threshold={cfg.domain_similarity_threshold:.4f})"
                    )

            # On-domain query — proceed with RAG pipeline
            else:
                with st.spinner(cfg.ui_text.page.response_spinner):
                    try:
                        logger.info(
                            cfg.ui_text.page.query_log_template.format(query=f"{user_input[:80]}...")
                        )

                        retrieval_started = time.perf_counter()
                        sources = retrieve_sources(vector_store, user_input)
                        retrieval_ended = time.perf_counter()

                        # Invoke RAG pipeline (retrieval + grounded generation)
                        logger.debug(f"[TRACE] rag_chain.invoke() entry point — mode={cfg.llm_mode}")
                        generation_started = time.perf_counter()
                        result = rag_chain.invoke({"query": user_input})
                        generation_ended = time.perf_counter()

                        raw_response  = result.get("result", "")
                        safe_response = sanitize_response(raw_response)
                        # Non-streaming invoke: approximate TTFT with generation completion timestamp.
                        first_token_ts = generation_ended

                        log_success(logger, f"Response generated ({len(safe_response)} chars, {len(sources)} sources)")

                        # Display response
                        st.markdown(safe_response)

                        # Show sources
                        if sources and cfg.show_retrieved_sources:
                            with st.container(key=f"container-{request_key}"):
                                with st.expander(cfg.ui_text.page.source_expander_label, key=f"expander-{request_key}"):
                                    for i, src in enumerate(sources, 1):
                                        meta  = src.get("metadata", {})
                                        label = format_source_label(meta, cfg.ui_text.page.source_fallback_label)
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
                        error_message = str(e)

        request_ended = time.perf_counter()
        st.session_state.active_requests = max(st.session_state.active_requests - 1, 0)

        # ── Build & persist performance metrics (delegated to metrics_tracker) ──
        metrics_entry = build_metrics_entry(
            user_input=user_input,
            request_started=request_started,
            request_ended=request_ended,
            retrieval_started=retrieval_started,
            retrieval_ended=retrieval_ended,
            generation_started=generation_started,
            generation_ended=generation_ended,
            first_token_ts=first_token_ts,
            sources=sources,
            safe_response=safe_response,
            concurrent_users=concurrent_users,
            error_message=error_message,
        )
        persist_metrics(metrics_entry)


    # ── Footer ──
    st.divider()
    st.markdown(cfg.ui_text.page.footer_html.format(llm_model=cfg.llm_model),unsafe_allow_html=True)
    st.divider()

if __name__ == "__main__":
    log_section(logger, "Healthcare Symptom Information Assistant — Starting")
    main()
  
