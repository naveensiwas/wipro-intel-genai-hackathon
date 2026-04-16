"""
Metrics Dashboard UI
Renders the in-app metrics dashboard for request-level performance telemetry.

What this module does
──────────────────────────────────────────────────────────
- Loads persisted metrics from `data/runtime/metrics/performance_metrics.json`
- Computes dashboard aggregates (averages focused on successful requests)
- Displays latency metrics in seconds for UI consistency
- Exposes CSV download for raw request metrics (stored in ms)
- Supports reset action via `metrics_tracker.reset_metrics()`
- Shows a recent requests table with display-friendly latency fields
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
from app.core.metrics import reset_metrics
from app.config.settings import cfg
from app.ui.styles import StreamlitStyles

# Resolve the metrics file path from config, ensuring it's absolute and exists relative to the repo root.
METRICS_FILE_PATH = Path(cfg.metrics_file_path)


def _load_metrics_payload() -> dict:
    """Load the metrics payload from the JSON file, with error handling and fallback."""
    if not METRICS_FILE_PATH.exists():
        return {"requests": [], "summary": {}}
    try:
        return json.loads(METRICS_FILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"requests": [], "summary": {}}


def _build_dataframe(requests: list[dict]) -> pd.DataFrame:
    """Convert the list of request dictionaries into a pandas DataFrame, with type coercion."""
    if not requests:
        return pd.DataFrame()

    df = pd.DataFrame(requests)
    numeric_cols = [
        "ttft_ms",
        "total_latency_ms",
        "tokens_per_sec",
        "retrieval_latency_ms",
        "end_to_end_latency_ms",
        "context_size_tokens",
        "output_tokens",
        "concurrent_users",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)

    return df


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert the DataFrame to CSV format and return as bytes for download."""
    export_cols = [
        "timestamp_utc",
        "query_preview",
        "ttft_ms",
        "total_latency_ms",
        "tokens_per_sec",
        "retrieval_latency_ms",
        "end_to_end_latency_ms",
        "context_size_tokens",
        "output_tokens",
        "concurrent_users",
        "error",
    ]
    cols = [c for c in export_cols if c in df.columns]
    export_df = df[cols].copy()
    if "timestamp_utc" in export_df.columns:
        export_df["timestamp_utc"] = export_df["timestamp_utc"].astype(str)

    return export_df.to_csv(index=False).encode("utf-8")


def _ms_to_seconds(ms_value: float | None) -> float | None:
    """Convert milliseconds to seconds, handling None and NaN values gracefully."""
    if ms_value is None or pd.isna(ms_value):
        return None
    return float(ms_value) / 1000.0


def _format_duration_from_ms(ms_value: float | None) -> str:
    """Format a duration given in milliseconds into a human-readable string in seconds."""
    seconds = _ms_to_seconds(ms_value)
    if seconds is None:
        return "N/A"
    return f"{seconds:.2f} sec"


def _format_timestamp_for_display(ts_value) -> str:
    """Format timestamp into IST for table display."""
    if ts_value is None or pd.isna(ts_value):
        return "N/A"
    ts = pd.to_datetime(ts_value, errors="coerce", utc=True)
    if pd.isna(ts):
        return "N/A"

    # Convert UTC -> IST
    ts_ist = ts.tz_convert("Asia/Kolkata")
    return ts_ist.strftime("%a, %d %b %Y %I:%M %p")


def _get_health_status(df: pd.DataFrame, success_df: pd.DataFrame) -> tuple[str, str]:
    """Determine system health based on latency and success rate."""
    if df.empty:
        return "🔘 No Data", "grey"

    success_rate = len(success_df) / len(df) if len(df) > 0 else 0
    # avg_latency = success_df["total_latency_ms"].mean() if "total_latency_ms" in success_df.columns else 0

    if success_rate >= 0.80:
        return "🟢 Healthy", "green"
    elif success_rate >= 0.60:
        return "🟡 Degraded", "orange"
    else:
        return "🔴 Unhealthy", "red"


def main() -> None:
    """Render the performance metrics dashboard, including summary metrics, averages, and a table of recent requests."""
    # Apply custom styles
    StreamlitStyles.apply_all_styles()

    # Load and prepare data
    payload = _load_metrics_payload()
    requests = payload.get("requests", [])
    summary = payload.get("summary", {})

    # Convert to DataFrame for easier manipulation and display
    df = _build_dataframe(requests)

    # Display empty state if no metrics are available
    if df.empty:
        st.warning(cfg.ui_text.page.dashboard_no_records)
        return

    # For summary metrics, focus on successful requests
    success_df = df[df["error"].isna()] if "error" in df.columns else df

    # ── Health Status Indicator ──
    health_status, health_color = _get_health_status(df, success_df)
    st.markdown(f"<h3 style='color: {health_color};'>{health_status}</h3>", unsafe_allow_html=True)

    success_rate = (len(success_df) / len(df) * 100) if len(df) > 0 else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Success Rate", f"{success_rate:.1f}%")
    col2.metric("Total Requests", len(df))
    col3.metric("Successful", len(success_df))

    st.divider()

    # ── Controls: Download CSV and Reset Metrics ──
    st.markdown("### 🎛️ Dashboard Controls")
    controls_left, controls_right = st.columns([1, 1])

    with controls_left:
        with st.container(key="btn-blue-download"):
            st.download_button(
                label=cfg.ui_text.page.metrics_download_button,
                data=_to_csv_bytes(df),
                file_name="performance_metrics_requests.csv",
                mime="text/csv",
                key="download_metrics_btn",
                use_container_width=True,
            )

    with controls_right:
        with st.container(key="btn-red-reset"):
            with st.popover(cfg.ui_text.page.metrics_reset_popover, use_container_width=True):
                st.warning(cfg.ui_text.page.metrics_reset_warning)
                confirm_reset = st.checkbox(
                    cfg.ui_text.page.metrics_reset_checkbox,
                    key="confirm_reset_checkbox",
                )
                with st.container(key="btn-red-confirm-reset"):
                    if st.button(
                            cfg.ui_text.page.metrics_confirm_reset_button,
                            key="confirm_reset_btn",
                            disabled=not confirm_reset,
                            use_container_width=True,
                    ):
                        reset_metrics()
                        st.success(cfg.ui_text.page.metrics_reset_success)
                        st.rerun()

    st.divider()

    # Summary metrics: total requests, successful requests, average latency, max concurrency observed
    with st.container(key="expander-blue-summary-metrics"):
        with st.expander(cfg.ui_text.page.summary_expander_title, expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(cfg.ui_text.page.summary_total_requests_label, int(summary.get("total_requests", len(df))))
            c2.metric(cfg.ui_text.page.summary_successful_requests_label, int(summary.get("successful_requests", len(success_df))))
            avg_total_latency_ms = (
                success_df["total_latency_ms"].dropna().mean()
                if "total_latency_ms" in success_df.columns
                else None
            )
            c3.metric(cfg.ui_text.page.summary_avg_total_latency_label, _format_duration_from_ms(avg_total_latency_ms))
            c4.metric(
                cfg.ui_text.page.summary_max_concurrent_label,
                int(summary.get("max_concurrent_users_observed", df.get("concurrent_users", pd.Series([0])).max())),
            )
            st.markdown(cfg.ui_text.page.summary_help_html, unsafe_allow_html=True)
            StreamlitStyles.render_definition_group([
                ("📊", "Total Requests", "Count of all requests logged in the metrics file (successful + failed)."),
                ("✅", "Successful Requests", "Requests with no recorded error; used for most average calculations."),
                ("⏱️", "Avg Total Latency (avg sec)", "Mean response wait time per successful request, displayed in seconds."),
                ("👥", "Max Concurrent Users", "Peak number of in-flight requests observed at any point in session history."),
            ])
    st.divider()

    # Latency averages for successful requests (end-to-end, retrieval, total, TTFT)
    with st.container(key="expander-blue-latency-averages"):
        with st.expander(cfg.ui_text.page.latency_expander_title, expanded=True):
            avg_end_to_end = (
                success_df["end_to_end_latency_ms"].dropna().mean()
                if "end_to_end_latency_ms" in success_df.columns
                else None
            )
            avg_retrieval = (
                success_df["retrieval_latency_ms"].dropna().mean()
                if "retrieval_latency_ms" in success_df.columns
                else None
            )
            avg_total = (
                success_df["total_latency_ms"].dropna().mean()
                if "total_latency_ms" in success_df.columns
                else None
            )
            avg_ttft = (
                success_df["ttft_ms"].dropna().mean()
                if "ttft_ms" in success_df.columns
                else None
            )

            l1, l2, l3, l4 = st.columns(4)
            l1.metric(cfg.ui_text.page.latency_end_to_end_label, _format_duration_from_ms(avg_end_to_end))
            l2.metric(cfg.ui_text.page.latency_retrieval_label, _format_duration_from_ms(avg_retrieval))
            l3.metric(cfg.ui_text.page.latency_total_label, _format_duration_from_ms(avg_total))
            l4.metric(cfg.ui_text.page.latency_ttft_label, _format_duration_from_ms(avg_ttft))

            # Definitions for Latency Metrics
            StreamlitStyles.render_definition_group([
                ("🚀", "End-to-End Latency (avg sec)", "Average full round-trip time from user submission to final response shown."),
                ("🔍", "Retrieval Latency (avg sec)", "Average time spent in context retrieval (vector search / RAG lookup)."),
                ("⚙️", "Total Latency (avg sec)", "Average total processing time for request handling, generation, and post-processing."),
                ("⚡", "TTFT (avg sec)", "Average Time To First Token — how quickly generation starts after submission."),
            ])
    st.divider()

    # Throughput and context averages: tokens/sec, context size, output tokens, concurrent users
    with st.container(key="expander-blue-throughput-context"):
        with st.expander(cfg.ui_text.page.throughput_expander_title, expanded=True):
            avg_tokens_per_sec = (
                success_df["tokens_per_sec"].dropna().mean()
                if "tokens_per_sec" in success_df.columns
                else None
            )
            avg_context_size = (
                success_df["context_size_tokens"].dropna().mean()
                if "context_size_tokens" in success_df.columns
                else None
            )
            avg_output_tokens = (
                success_df["output_tokens"].dropna().mean()
                if "output_tokens" in success_df.columns
                else None
            )
            avg_concurrent_users = (
                success_df["concurrent_users"].dropna().mean()
                if "concurrent_users" in success_df.columns
                else None
            )

            t1, t2, t3, t4 = st.columns(4)
            t1.metric(cfg.ui_text.page.throughput_tokens_per_sec_label, f"{avg_tokens_per_sec:.2f}" if pd.notna(avg_tokens_per_sec) else "N/A")
            t2.metric(cfg.ui_text.page.throughput_context_size_label, f"{avg_context_size:.2f}" if pd.notna(avg_context_size) else "N/A")
            t3.metric(cfg.ui_text.page.throughput_output_tokens_label, f"{avg_output_tokens:.2f}" if pd.notna(avg_output_tokens) else "N/A")
            t4.metric(cfg.ui_text.page.throughput_concurrent_users_label, f"{avg_concurrent_users:.2f}" if pd.notna(avg_concurrent_users) else "N/A")

            # Definitions for Throughput & Context Metrics
            StreamlitStyles.render_definition_group([
                ("🎯", "Tokens/sec (avg)", "Average generation throughput speed during response creation (higher = faster)."),
                ("📚", "Context Size (avg tokens)", "Average number of context tokens supplied to the model from retrieved sources."),
                ("📝", "Output Tokens (avg)", "Average response length per successful request (token/word-count approximation)."),
                ("👤", "Concurrent Users (avg)", "Average in-flight request count at the time successful responses were processed."),
            ])


    st.divider()
    st.markdown(cfg.ui_text.page.recent_requests_title_html, unsafe_allow_html=True)
    st.markdown(cfg.ui_text.page.recent_requests_subtitle_html, unsafe_allow_html=True)
    table_df = df.copy()

    if "ttft_ms" in table_df.columns:
        table_df["TTFT (avg sec)"] = table_df["ttft_ms"].apply(_format_duration_from_ms)
    if "retrieval_latency_ms" in table_df.columns:
        table_df["Retrieval Latency (avg sec)"] = table_df["retrieval_latency_ms"].apply(_format_duration_from_ms)
    if "total_latency_ms" in table_df.columns:
        table_df["Total Latency (avg sec)"] = table_df["total_latency_ms"].apply(_format_duration_from_ms)
    if "end_to_end_latency_ms" in table_df.columns:
        table_df["End-to-End Latency (avg sec)"] = table_df["end_to_end_latency_ms"].apply(_format_duration_from_ms)

    # Normalize raw field names to human-readable, consistent headers
    header_rename_map = {
        "timestamp_utc": "Timestamp (IST)",
        "query_preview": "User Query",
        "tokens_per_sec": "Tokens/sec",
        "context_size_tokens": "Context Size (tokens)",
        "concurrent_users": "Concurrent Users",
        "error": "Error",
    }

    table_df = table_df.rename(columns=header_rename_map)
    view_cols = [
        "Timestamp (IST)",
        "User Query",
        "TTFT (avg sec)",
        "Retrieval Latency (avg sec)",
        "Total Latency (avg sec)",
        "End-to-End Latency (avg sec)",
        "Tokens/sec",
        "Context Size (tokens)",
        "Concurrent Users",
        "Error"
    ]

    if "Timestamp (IST)" in table_df.columns:
        table_df["Timestamp (IST)"] = table_df["Timestamp (IST)"].apply(_format_timestamp_for_display)

    view_cols = [c for c in view_cols if c in table_df.columns]
    display_df = table_df.sort_values("Timestamp (IST)", ascending=False)[view_cols].head(50)

    display_df = display_df.reset_index(drop=True).reset_index()
    display_df = display_df.rename(columns={"index": "SR No."})
    display_df["SR No."] = display_df["SR No."] +1

    # Use pandas Styler to apply custom header CSS
    styled_df = display_df.style.hide(axis="index").set_properties(**{
        'border': '1px solid #dbeafe',
    }).set_table_styles([
        # Header row styling — matches expander header style
        {
            'selector': 'thead tr th',
            'props': [
                ('background-color', '#eff6ff'),
                ('color', '#1e3a8a'),
                ('font-weight', '700'),
                ('border', '1px solid #dbeafe'),
                ('text-align', 'left'),
                ('border-bottom', '2px solid #dbeafe'),
            ]
        },
        # Body rows — alternating stripe
        {
            'selector': 'tbody tr:nth-child(even)',
            'props': [('background-color', '#f8fbff')]
        },
        # Body cells
        {
            'selector': 'tbody td',
            'props': [
                ('border', '1px solid #dbeafe'),
            ]
        },
    ])

    # Convert Styler to HTML and render
    html_table = styled_df.to_html(escape=False, max_rows=50)

    # st.markdown(html_table, unsafe_allow_html=True)
    # Wrap in a container with overflow for horizontal scroll
    st.markdown(
        f"""
        <div style="border: 1px solid #93c5fd; border-radius: 8px; overflow: hidden; overflow-x: auto; border-left: 3px solid var(--color-info-400) !important;">
            {html_table}
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
