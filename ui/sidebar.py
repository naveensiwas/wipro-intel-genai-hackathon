"""
Sidebar component: disclaimer banner, about section, and reset button.
"""
import json
from pathlib import Path

import streamlit as st
from config import cfg
from logger_config import get_logger, log_warning, log_error

logger = get_logger(__name__)


ABOUT_THIS_APP = cfg.ui_text.sidebar.about_template.format(llm_model=cfg.llm_model)


def _load_symptom_options() -> list[str]:
    """Load symptom options from data; fallback to static defaults."""
    defaults = cfg.ui_text.sidebar.symptom_defaults

    try:
        data_path = Path(cfg.symptoms_conditions_path)
        if not data_path.exists():
            log_warning(logger, f"Symptom source not found: {data_path}. Using defaults.")
            return defaults

        payload = json.loads(data_path.read_text(encoding="utf-8"))
        records = payload.get("symptoms", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            log_warning(logger, f"Unexpected symptom schema in {data_path}. Using defaults.")
            return defaults

        symptoms: set[str] = set()
        for item in records:
            if not isinstance(item, dict):
                continue
            value = item.get("symptom")
            if isinstance(value, str) and value.strip():
                symptoms.add(value.strip().title())

        if not symptoms:
            log_warning(logger, f"No symptoms found in {data_path}. Using defaults.")
            return defaults

        options = ["Select..."] + sorted(symptoms, key=lambda item: item.lower())
        logger.info(f"Loaded {len(options) - 1} dynamic symptom options from {data_path}")
        return options

    except Exception as exc:
        log_error(logger, "Failed to load dynamic symptom options. Using defaults.", exc)
        return defaults


def render_sidebar():
    # Keep the sidebar width stable for better readability and consistent layout.
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 400px !important;
                min-width: 400px !important;
            }

            section[data-testid="stSidebar"] div.stButton > button {
                background-color: #2563eb !important;
                color: #ffffff !important;
                border: 1px solid #1d4ed8 !important;
                border-radius: 8px !important;
                justify-content: flex-start !important;
                text-align: left !important;
            }

            section[data-testid="stSidebar"] div.stButton > button:hover {
                background-color: #1d4ed8 !important;
                border-color: #1e40af !important;
                color: #ffffff !important;
            }

            section[data-testid="stSidebar"] div.stButton > button:active {
                background-color: #1e40af !important;
                border-color: #1e3a8a !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        # App logo / branding
        st.markdown(cfg.ui_text.sidebar.title)
        st.markdown(cfg.ui_text.sidebar.tagline)
        st.divider()

        # Disclaimer — always visible
        st.error(cfg.ui_text.sidebar.disclaimer, icon="⚠️")
        st.divider()

        # Structured symptom helper
        st.markdown(cfg.ui_text.sidebar.quick_guide_heading)
        with st.expander(cfg.ui_text.sidebar.structured_input_expander):
            symptom = st.selectbox(
                cfg.ui_text.sidebar.symptom_label,
                _load_symptom_options(),
                key="symptom_select",
            )
            duration = st.selectbox(
                cfg.ui_text.sidebar.duration_label,
                cfg.ui_text.sidebar.duration_options,
                key="duration_select",
            )
            severity = st.slider(cfg.ui_text.sidebar.severity_label, 1, 10, 5, key="severity_slider")
            extra_symptoms = st.text_input(cfg.ui_text.sidebar.extra_symptoms_label, key="extra_symptoms")

            # Build a structured query based on inputs
            if st.button(cfg.ui_text.sidebar.build_query_button, key="build_query_btn"):
                if not symptom or symptom == "Select...":
                    st.warning(cfg.ui_text.sidebar.primary_symptom_required_warning)
                else:
                    parts = []
                    parts.append(f"My primary symptom is {symptom}")
                    if extra_symptoms.strip():
                        parts.append(f"including {extra_symptoms.strip()}")
                    if duration and duration != "Select...":
                        parts.append(f"for {duration.lower()}")
                    parts.append(f"with a severity of {severity}/10")

                    composed = ", ".join(parts) + ". What could this be and what should I do?"
                    st.session_state["prefilled_query"] = composed
                    st.success(cfg.ui_text.sidebar.build_query_success)

        st.divider()

        # Clear chat history
        if st.button(cfg.ui_text.sidebar.clear_chat_button, use_container_width=True):
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": cfg.ui_text.chat.welcome_message,
                    "sources": [],
                }
            ]
            st.session_state.pop("prefilled_query", None)
            st.rerun()

        st.divider()

        # About this app
        with st.expander(cfg.ui_text.sidebar.about_expander):
            st.markdown(ABOUT_THIS_APP)
