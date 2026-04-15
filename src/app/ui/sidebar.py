"""
Sidebar component: disclaimer banner, about section, and reset button.
"""
import json
from pathlib import Path
import streamlit as st
from app.config.settings import cfg
from app.core.logging import get_logger, log_warning, log_error
from app.ui.styles import StreamlitStyles

logger = get_logger(__name__)

# Pre-format the about text with the LLM model name for efficiency, since it doesn't change at runtime.
ABOUT_THIS_APP = cfg.ui_text.sidebar.about_template.format(llm_model=cfg.llm_model)

# Resolve repository root and logo paths independent of working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_WIPRO_LOGO_PATH = _PROJECT_ROOT / "assets" / "images" / "wipro.png"
_INTEL_LOGO_PATH = _PROJECT_ROOT / "assets" / "images" / "intel.png"


@st.cache_resource(show_spinner=False)
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

        options = ["Choose an option…"] + sorted(symptoms, key=lambda item: item.lower())
        logger.info(f"Loaded {len(options) - 1} dynamic symptom options from {data_path}")
        return options

    except Exception as exc:
        log_error(logger, "Failed to load dynamic symptom options. Using defaults.", exc)
        return defaults


def render_sidebar():
    # Apply centralized styles (moved to styles.py BASE_STYLES)
    StreamlitStyles.apply_all_styles()

    # Render all sidebar content within a single context to ensure consistent styling and layout.
    with st.sidebar:

        # App logo / branding with styled container for better control over layout and spacing
        _, logo_col1, logo_col2, _ = st.columns([1, 1, 1, 1], gap="xxsmall")
        with logo_col1:
            if _WIPRO_LOGO_PATH.exists():
                st.image(str(_WIPRO_LOGO_PATH), width=120)
            else:
                log_warning(logger, f"Wipro logo not found at {_WIPRO_LOGO_PATH}; skipping")
        with logo_col2:
            if _INTEL_LOGO_PATH.exists():
                st.image(str(_INTEL_LOGO_PATH), width=70)
            else:
                log_warning(logger, f"Intel logo not found at {_INTEL_LOGO_PATH}; skipping")

        st.space()
        st.markdown(f'<div class="sidebar-title">{cfg.ui_text.sidebar.title}</div>', unsafe_allow_html=True)
        st.divider()

        # Two-option left menu for navigation between chat and metrics dashboard views, with dynamic styling based on active view.
        st.markdown(f"### {cfg.ui_text.sidebar.navigation_heading}")
        is_chat = st.session_state["current_view"] == cfg.chat_view
        is_metrics = st.session_state["current_view"] == cfg.metrics_view

        with st.container(key="menu-chat-active" if is_chat else "menu-chat"):
            if st.button(cfg.ui_text.sidebar.chat_menu_button, key="menu_chat_btn", use_container_width=True):
                if not is_chat:
                    st.session_state["current_view"] = cfg.chat_view
                    st.rerun()

        with st.container(key="menu-metrics-active" if is_metrics else "menu-metrics"):
            if st.button(cfg.ui_text.sidebar.metrics_menu_button, key="menu_metrics_btn", use_container_width=True):
                if not is_metrics:
                    st.session_state["current_view"] = cfg.metrics_view
                    st.rerun()

        st.divider()

        # Only show quick symptom helper and clear chat history when user is in the chat view.
        if st.session_state["current_view"] == cfg.chat_view:

            # Structured symptom helper
            st.markdown(f"### {cfg.ui_text.sidebar.quick_guide_heading}")
            with st.container(key="expander-blue-structured-input"):
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
                    with st.container(key="btn-blue-build-query"):
                        if st.button(
                                cfg.ui_text.sidebar.build_query_button,
                                key="build_query_btn",
                                use_container_width=True,
                        ):
                            if not symptom or symptom == "Choose an option…":
                                st.error(cfg.ui_text.sidebar.primary_symptom_required_warning)
                            else:
                                parts = []
                                parts.append(f"My primary symptom is {symptom}")
                                if extra_symptoms.strip():
                                    parts.append(f"including {extra_symptoms.strip()}")
                                if duration and duration != "Select timeframe…":
                                    parts.append(f"for {duration.lower()}")
                                parts.append(f"with a severity of {severity}/10")

                                composed = ", ".join(parts) + ". What could this be and what should I do?"
                                st.session_state["prefilled_query"] = composed
                                st.success(cfg.ui_text.sidebar.build_query_success)

            # Space out the quick guide and clear chat sections for better visual separation.
            st.space()

            # Clear chat history
            with st.container(key="btn-red-clear-chat"):
                if st.button(cfg.ui_text.sidebar.clear_chat_button, use_container_width=True):
                    # Reset to initial welcome message defined in config, which can also be updated to reflect any changes in the quick guide heading or other dynamic values.
                    template = cfg.ui_text.chat.welcome_message
                    welcome_msg = template.format(smart_symptom_guide=cfg.ui_text.sidebar.quick_guide_heading)

                    st.session_state["messages"] = [
                        {
                            "role": "assistant",
                            "content": welcome_msg,
                            "sources": [],
                        }
                    ]
                    st.session_state.pop("prefilled_query", None)
                    st.rerun()
            st.divider()

        # About this app
        with st.container(key="expander-blue-about-panel"):
            with st.expander(cfg.ui_text.sidebar.about_expander):
                st.markdown(ABOUT_THIS_APP)
        st.divider()
