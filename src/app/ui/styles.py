"""
Streamlit styling utilities for the metrics dashboard.
"""

import streamlit as st


class StreamlitStyles:
    """Styles and helpers used by the metrics dashboard."""

    BASE_STYLES = """
    <style>
    :root {
        /* Info / Blue */
        --color-info-50: #f8fbff;
        --color-info-100: #eff6ff;
        --color-info-200: #dbeafe;
        --color-info-300: #bfdbfe;
        --color-info-400: #93c5fd;
        --color-info-500: #60a5fa;
        --color-info-700: #1e40af;
        --color-info-800: #1e3a8a;
        --color-info-accent: #3b82f6;
        --shadow-info-sm: 0 2px 8px rgba(59, 130, 246, 0.18);
        --shadow-info-lg: 0 4px 14px rgba(59, 130, 246, 0.30);
        --shadow-info-hover: 0 6px 14px rgba(59, 130, 246, 0.12);

        /* Danger / Red */
        --color-danger-200: #fee2e2;
        --color-danger-300: #fecaca;
        --color-danger-400: #fca5a5;
        --color-danger-800: #991b1b;
        --shadow-danger-sm: 0 2px 8px rgba(239, 68, 68, 0.18);
        --shadow-danger-lg: 0 4px 14px rgba(239, 68, 68, 0.30);

        /* Neutrals */
        --color-white: #ffffff;
        --color-text-muted: #94a3b8;
        --color-border-muted: #cbd5e1;
        --color-bg-disabled: #f1f5f9;
        --shadow-soft: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    /* ── Expander styles (actual Streamlit DOM: details > summary) ── */
    [data-testid="stExpander"] > details {
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid var(--color-info-400) !important;
        background: transparent !important;
    }

    [data-testid="stExpander"] > details > summary {
        background: transparent !important;
        cursor: pointer !important;
    }

    [data-testid="stExpander"] > details > summary:hover {
        background: transparent !important;
    }

    [data-testid="stExpander"] > details > summary p,
    [data-testid="stExpander"] > details > summary span,
    [data-testid="stExpander"] > details > summary [data-testid="stIconMaterial"] {
        color: var(--color-info-800) !important;
        fill: var(--color-info-800) !important;
        font-weight: 700 !important;
    }

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: transparent !important;
        border-left: 3px solid var(--color-info-400) !important;
        padding: 16px !important;
        border-radius: 0 8px 8px 0 !important;
    }

    /* ── Expander colours via container key token ── */
    [class*="st-key-expander-blue"] [data-testid="stExpander"] > details {
        border: 1px solid var(--color-info-400) !important;
        background: transparent !important;
    }
    [class*="st-key-expander-blue"] [data-testid="stExpander"] > details > summary {
        background: transparent !important;
    }
    [class*="st-key-expander-blue"] [data-testid="stExpander"] > details > summary:hover {
        background: transparent !important;
    }
    [class*="st-key-expander-blue"] [data-testid="stExpander"] > details > summary * {
        color: var(--color-info-800) !important;
        fill: var(--color-info-800) !important;
    }
    [class*="st-key-expander-blue"] [data-testid="stExpanderDetails"] {
        background-color: transparent !important;
        border-left: 3px solid var(--color-info-400) !important;
    }

    /* ── Metric card styles ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--color-white) 0%, var(--color-info-50) 100%) !important;
        border-radius: 12px !important;
        border: 1px solid var(--color-info-200) !important;
        padding: 16px !important;
        box-shadow: var(--shadow-soft) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="metric-container"]:hover {
        box-shadow: var(--shadow-info-hover) !important;
        border-color: var(--color-info-400) !important;
    }

    /* ── Metric definition box ── */
    .metric-definition {
        background-color: var(--color-info-100) !important;
        border-left: 4px solid var(--color-info-accent) !important;
        padding: 10px 12px !important;
        border-radius: 4px !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        color: var(--color-info-700) !important;
        margin-bottom: 8px !important;
    }

    /* ── Button colours via container key token ── */
    [class*="st-key-btn-blue"] button {
        background: var(--color-info-200) !important;
        color: var(--color-info-800) !important;
        border: 1.5px solid var(--color-info-400) !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: var(--shadow-info-sm) !important;
        transition: background 0.2s ease, box-shadow 0.2s ease !important;
    }
    [class*="st-key-btn-blue"] button:hover {
        background: var(--color-info-300) !important;
        box-shadow: var(--shadow-info-lg) !important;
    }

    [class*="st-key-btn-red"] button {
        background: var(--color-danger-200) !important;
        color: var(--color-danger-800) !important;
        border: 1.5px solid var(--color-danger-400) !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: var(--shadow-danger-sm) !important;
        transition: background 0.2s ease, box-shadow 0.2s ease !important;
    }
    [class*="st-key-btn-red"] button:hover {
        background: var(--color-danger-300) !important;
        box-shadow: var(--shadow-danger-lg) !important;
    }
    [class*="st-key-btn-blue"] button:disabled,
    [class*="st-key-btn-red"] button:disabled {
        background: var(--color-bg-disabled) !important;
        color: var(--color-text-muted) !important;
        border: 1.5px solid var(--color-border-muted) !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }

    /* Logo/Branding container in sidebar */
    .logo-container img {
        display: block;
        max-width: 100%;
        height: auto;
    }

    .sidebar-title {
        color: #0c4a6e;
        font-weight: 700;
        line-height: 1.4;
        font-size: 16px;
        text-align: center;
    }

    /* ── Sidebar menu buttons ── */
    section[data-testid="stSidebar"] {
        width: 400px !important;
        min-width: 400px !important;
        max-width: 400px !important;
        flex: 0 0 400px !important;
    }
    [data-testid="stBaseButton-headerNoPadding"] {
        display: none;
    }

    /* Base clickable menu items */
    [class*="st-key-menu-chat"] button,
    [class*="st-key-menu-metrics"] button,
    [class*="st-key-menu-chat-active"] button,
    [class*="st-key-menu-metrics-active"] button {
        background-color: var(--color-info-100) !important;
        border-left: 5px solid var(--color-info-accent) !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        margin-bottom: 8px !important;
    }
    
    /* Hover state */
    [class*="st-key-menu-chat"] button:hover,
    [class*="st-key-menu-metrics"] button:hover,
    [class*="st-key-menu-chat-active"] button:hover,
    [class*="st-key-menu-metrics-active"] button:hover {
        background-color: var(--color-info-200) !important;
    }
    
    /* Active state */
    [class*="st-key-menu-chat-active"] button,
    [class*="st-key-menu-metrics-active"] button {
        background-color: var(--color-info-300) !important;
    }

    /* ── Page titles and headings ── */
    .page-title-main {
        text-align: center;
        font-weight: 700;
        color: #0c4a6e;
        font-size: 50px;
    }

    .page-tagline {
        text-align: center;
        color: grey;
        font-size: 0.9em;
    }

    .page-footer {
        text-align: center;
        font-size: 0.9em;
    }

    .footer-disclaimer {
        color: darkorange;
    }

    /* ── Metrics dashboard headings ── */
    .metrics-section-title {
        color: #1f2937;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .metrics-recent-requests-title {
        color: #1f2937;
        font-weight: 700;
        margin-bottom: 12px;
        font-size: 2em;
    }

    .metrics-recent-requests-subtitle {
        color: #4b5563;
        font-size: 13px;
        margin-bottom: 16px;
    }
    
    /* Hide the stray code block generated after custom HTML table render */
    div.stElementContainer:has(.stDataFrame) + div[data-testid="stCode"],
    div[data-testid="stCode"].stCode.st-emotion-cache-w7p8pq {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    
    </style>
    """

    @staticmethod
    def apply_all_styles() -> None:
        """Inject all base + button CSS once per page."""
        st.markdown(StreamlitStyles.BASE_STYLES, unsafe_allow_html=True)

    @staticmethod
    def render_metric_definition(metric_name: str, description: str, emoji: str = "📊") -> None:
        """Render a styled metric definition row."""
        st.markdown(
            f'<div class="metric-definition"><strong>{emoji} {metric_name}:</strong> {description}</div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_definition_group(definitions: list[tuple[str, str, str]]) -> None:
        """Render a set of metric definitions."""
        for emoji, metric_name, description in definitions:
            StreamlitStyles.render_metric_definition(metric_name, description, emoji)
