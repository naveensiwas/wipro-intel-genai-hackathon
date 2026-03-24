"""
Chat interface component.
Renders the message history and handles new messages.
"""
import streamlit as st


_WELCOME_MESSAGE = """👋 **Hello! I'm your Health Information Assistant.**

I can help you understand:
- 🤒 What conditions may be associated with your symptoms
- 💊 General care and wellbeing tips
- 🏥 When you should seek medical care
- 👨‍⚕️ What type of specialist to consult

**How to use:**
- Type your symptoms in the chat box below, or
- Use the **Quick Symptom Guide** in the sidebar to build a structured query

⚠️ *Remember: I provide general health information only — not medical diagnoses. Always consult a healthcare professional for medical concerns.*

---
*What symptoms are you experiencing today?*
"""


def init_chat_state():
    """Initialise session state for chat."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": _WELCOME_MESSAGE, "sources": []}
        ]


def render_chat_history():
    """Render all messages in the chat history."""
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        with st.chat_message(role, avatar="🩺" if role == "assistant" else "👤"):
            st.markdown(content)

            # Show retrieved sources in a collapsible section
            if sources:
                with st.expander("📚 View retrieved health knowledge sources"):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        label = (
                            meta.get("condition")
                            or meta.get("symptom")
                            or meta.get("category")
                            or meta.get("source_file", "Health Data")
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

