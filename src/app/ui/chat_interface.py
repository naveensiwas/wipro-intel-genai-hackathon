"""
Chat interface component.

Renders the message history and handles new messages in the Streamlit chat UI.
Session state is used to persist messages across reruns.
"""
import streamlit as st
from app.config.settings import cfg
from app.ui.source_utils import format_source_label


def init_chat_state():
    """
    Initialize the Streamlit session state for multi-turn chat.

    Creates an empty messages list with a welcome message on first page load.
    Subsequent reruns preserve the message history via session state.
    """
    if "messages" not in st.session_state:

        # Use welcome message template from config and format with any dynamic values (e.g. quick guide heading)
        template = cfg.ui_text.chat.welcome_message
        welcome_msg = template.format(smart_symptom_guide=cfg.ui_text.sidebar.quick_guide_heading)

        st.session_state["messages"] = [
            {"role": "assistant", "content": welcome_msg, "sources": []}
        ]

def render_chat_history():
    """
    Render all messages in the current chat history.

    Displays each message with appropriate role-based styling and avatars.
    Retrieved sources are rendered in collapsible sections if available.
    """
    for msg_idx, msg in enumerate(st.session_state["messages"]):
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        # Use different avatars for user and assistant messages
        with st.chat_message(role, avatar="🩺" if role == "assistant" else "👤"):
            st.markdown(content)

            # Show retrieved sources in a collapsible section
            if sources and cfg.show_retrieved_sources:
                with st.container(key=f"expander-container-{msg_idx}"):
                    with st.expander(cfg.ui_text.page.source_expander_label, key=f"expander-sources-{msg_idx}"):
                        for i, src in enumerate(sources, 1):
                            meta = src.get("metadata", {})
                            label = format_source_label(meta, cfg.ui_text.page.source_fallback_label)
                            st.markdown(f"**Source {i} — {label}**")
                            st.caption(src["content"])
                            st.divider()


def add_message(role: str, content: str, sources: list = None):
    """
    Append a message to the session state chat history.

    This persists the message across Streamlit reruns, enabling multi-turn
    conversations without losing history.

    Args:
        role: Either "user" or "assistant"
        content: The text content of the message
        sources: Optional list of retrieved source documents (RAG context)
    """
    st.session_state["messages"].append({
        "role": role,
        "content": content,
        "sources": sources or [],
    })
