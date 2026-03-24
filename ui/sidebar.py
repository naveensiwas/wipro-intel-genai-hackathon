"""
Sidebar component: disclaimer banner, about section, and reset button.
"""
import streamlit as st
from config import cfg


def render_sidebar():
    with st.sidebar:
        # App logo / branding
        st.markdown("## 🩺 Symptom Checker")
        st.markdown(f"*Powered by {cfg.llm_model} · RAG · configurable endpoint*")
        st.divider()

        # Disclaimer — always visible
        st.error(cfg.disclaimer, icon="⚠️")
        st.divider()

        # Structured symptom helper
        st.markdown("### 🔍 Quick Symptom Guide")
        with st.expander("Use structured input"):
            body_area = st.selectbox(
                "Body area affected:",
                [
                    "Select...",
                    "Head / Neurological",
                    "Chest / Respiratory",
                    "Cardiovascular / Heart",
                    "Abdomen / Gastrointestinal",
                    "Skin / Dermatological",
                    "Joints / Musculoskeletal",
                    "Urinary / Kidney",
                    "Endocrine / Hormonal",
                    "Eyes / Vision",
                    "Ear, Nose, Throat",
                    "Mental Health",
                    "General / Whole Body",
                ],
                key="body_area_select",
            )
            duration = st.selectbox(
                "How long?",
                ["Select...", "Less than 1 day", "1–3 days", "4–7 days", "1–4 weeks", "More than 1 month"],
                key="duration_select",
            )
            severity = st.slider("Severity (1 = mild, 10 = severe):", 1, 10, 5, key="severity_slider")
            extra_symptoms = st.text_input("Additional symptoms (comma-separated):", key="extra_symptoms")

            if st.button("📋 Build Query", key="build_query_btn"):
                parts = []
                if body_area and body_area != "Select...":
                    parts.append(f"I have symptoms in the {body_area} area")
                if extra_symptoms.strip():
                    parts.append(f"including {extra_symptoms.strip()}")
                if duration and duration != "Select...":
                    parts.append(f"for {duration.lower()}")
                parts.append(f"with a severity of {severity}/10")

                composed = ", ".join(parts) + ". What could this be and what should I do?"
                st.session_state["prefilled_query"] = composed
                st.success("Query ready! It will appear in the chat input below ↓")

        st.divider()

        # Clear chat
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state.pop("prefilled_query", None)
            st.rerun()

        st.divider()

        # About
        with st.expander("ℹ️ About this app"):
            st.markdown(
                f"""
**Healthcare Symptom Information Assistant**

This app provides general health information using a
Retrieval-Augmented Generation (RAG) pipeline backed by
trusted public health data (WHO / CDC).

**Tech Stack**
- 🤖 Configured LLM endpoint (`{cfg.llm_model}`)
- 🔗 LangChain — RAG orchestration
- 🔍 FAISS — local vector search
- 🤗 all-MiniLM-L6-v2 — sentence embeddings
- 🖥️ Streamlit — UI

**Important**
This tool is for informational purposes only and does not
constitute medical advice. Always consult a qualified
healthcare professional.
                """
            )
