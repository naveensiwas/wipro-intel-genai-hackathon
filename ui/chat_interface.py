"""
Chat interface component.
Renders the message history and handles new messages.
"""
import streamlit as st
from config import cfg


def init_chat_state():
    """Initialise session state for chat."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": cfg.ui_text.chat.welcome_message, "sources": []}
        ]


def render_chat_history():
    """Render all messages in the chat history."""
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        # Use different avatars for user and assistant messages
        with st.chat_message(role, avatar="🩺" if role == "assistant" else "👤"):
            st.markdown(content)

            # Show retrieved sources in a collapsible section
            if sources and cfg.show_retrieved_sources:
                with st.expander(cfg.ui_text.page.source_expander_label):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        label = (
                            meta.get("condition")
                            or meta.get("symptom")
                            or meta.get("category")
                            or meta.get("source_file", cfg.ui_text.page.source_fallback_label)
                        )
                        st.markdown(f"**Source {i} — {label}**")
                        st.caption(src["content"])
                        st.divider()


def add_message(role: str, content: str, sources: list = None):
    """Append a message to session state."""
    st.session_state["messages"].append({
        "role": role,
        "content": content,
        "sources": sources or [],
    })
