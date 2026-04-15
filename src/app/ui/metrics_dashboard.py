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
import altair as alt
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


def _prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a clean UTC time-series DataFrame for charting."""
    if df.empty or "timestamp_utc" not in df.columns:
        return pd.DataFrame()

    ts_df = df.copy()
    ts_df = ts_df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    if ts_df.empty:
        return pd.DataFrame()

    # Convert milliseconds to seconds for consistent chart units.
    for ms_col in ["ttft_ms", "retrieval_latency_ms", "total_latency_ms", "end_to_end_latency_ms"]:
        if ms_col in ts_df.columns:
            ts_df[f"{ms_col}_sec"] = ts_df[ms_col] / 1000.0

    return ts_df


def _render_latency_trend_chart(ts_df: pd.DataFrame) -> None:
    """Render multi-series latency trend chart over time."""
    required_cols = [
        "timestamp_utc",
        "ttft_ms_sec",
        "retrieval_latency_ms_sec",
        "total_latency_ms_sec",
    ]
    if ts_df.empty or not all(c in ts_df.columns for c in required_cols):
        st.info("Latency trend chart unavailable: insufficient latency data.")
        return

    plot_df = ts_df[[
        "timestamp_utc",
        "ttft_ms_sec",
        "retrieval_latency_ms_sec",
        "total_latency_ms_sec",
    ]].rename(columns={
        "ttft_ms_sec": "TTFT",
        "retrieval_latency_ms_sec": "Retrieval",
        "total_latency_ms_sec": "Total",
    })

    melted = plot_df.melt(
        id_vars=["timestamp_utc"],
        var_name="Metric",
        value_name="Latency (sec)",
    ).dropna(subset=["Latency (sec)"])

    if melted.empty:
        st.info("Latency trend chart unavailable: no valid latency values.")
        return

    chart = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp_utc:T", title="Request Time (UTC)"),
            y=alt.Y("Latency (sec):Q", title="Latency (sec)", scale=alt.Scale(zero=True)),
            color=alt.Color("Metric:N", title="Latency Metric"),
            tooltip=[
                alt.Tooltip("timestamp_utc:T", title="Timestamp"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Latency (sec):Q", format=".3f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def _render_latency_distribution_chart(df: pd.DataFrame) -> None:
    """Render histogram of total latency distribution."""
    if df.empty or "total_latency_ms" not in df.columns:
        st.info("Latency distribution unavailable: total latency data missing.")
        return

    lat_df = df[["total_latency_ms"]].dropna().copy()
    if lat_df.empty:
        st.info("Latency distribution unavailable: no total latency values.")
        return

    lat_df["Total Latency (sec)"] = lat_df["total_latency_ms"] / 1000.0

    chart = (
        alt.Chart(lat_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Total Latency (sec):Q", bin=alt.Bin(maxbins=24), title="Total Latency (sec)"),
            y=alt.Y("count():Q", title="Request Count"),
            tooltip=[
                alt.Tooltip("count():Q", title="Requests"),
            ],
        )
        .properties(height=280)
    )

    st.altair_chart(chart, use_container_width=True)


def _render_success_failure_chart(df: pd.DataFrame) -> None:
    """Render success vs failure composition chart."""
    if df.empty:
        st.info("Health composition unavailable: no requests found.")
        return

    status = pd.Series(
        ["Successful" if pd.isna(err) else "Failed" for err in df.get("error", pd.Series(dtype=object))],
        name="Status",
    )
    status_df = status.value_counts().rename_axis("Status").reset_index(name="Count")

    if status_df.empty:
        st.info("Health composition unavailable.")
        return

    chart = (
        alt.Chart(status_df)
        .mark_arc(innerRadius=55)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Status:N", scale=alt.Scale(range=["#16a34a", "#dc2626"])),
            tooltip=[alt.Tooltip("Status:N"), alt.Tooltip("Count:Q")],
        )
        .properties(height=280)
    )

    st.altair_chart(chart, use_container_width=True)


def _render_throughput_trend_chart(ts_df: pd.DataFrame) -> None:
    """Render tokens/sec and output token trends over time."""
    if ts_df.empty:
        st.info("Throughput trend unavailable: no timestamped data.")
        return

    cols = [c for c in ["timestamp_utc", "tokens_per_sec", "output_tokens"] if c in ts_df.columns]
    if len(cols) < 2:
        st.info("Throughput trend unavailable: missing throughput columns.")
        return

    frames = []
    if "tokens_per_sec" in ts_df.columns:
        tmp = ts_df[["timestamp_utc", "tokens_per_sec"]].dropna().copy()
        tmp["Metric"] = "Tokens/sec"
        tmp["Value"] = tmp["tokens_per_sec"]
        frames.append(tmp[["timestamp_utc", "Metric", "Value"]])

    if "output_tokens" in ts_df.columns:
        tmp = ts_df[["timestamp_utc", "output_tokens"]].dropna().copy()
        tmp["Metric"] = "Output Tokens"
        tmp["Value"] = tmp["output_tokens"]
        frames.append(tmp[["timestamp_utc", "Metric", "Value"]])

    if not frames:
        st.info("Throughput trend unavailable: no valid throughput values.")
        return

    plot_df = pd.concat(frames, ignore_index=True)

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp_utc:T", title="Request Time (UTC)"),
            y=alt.Y("Value:Q", title="Metric Value"),
            color=alt.Color("Metric:N"),
            tooltip=[
                alt.Tooltip("timestamp_utc:T", title="Timestamp"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Value:Q", format=".3f"),
            ],
        )
        .properties(height=300)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def _render_context_output_bar_chart(df: pd.DataFrame) -> None:
    """Render average context-size vs output-token bar chart."""
    if df.empty:
        st.info("Context/output chart unavailable: no requests found.")
        return

    metrics = []
    if "context_size_tokens" in df.columns:
        v = df["context_size_tokens"].dropna().mean()
        if pd.notna(v):
            metrics.append({"Metric": "Avg Context Size", "Tokens": float(v)})

    if "output_tokens" in df.columns:
        v = df["output_tokens"].dropna().mean()
        if pd.notna(v):
            metrics.append({"Metric": "Avg Output Tokens", "Tokens": float(v)})

    if not metrics:
        st.info("Context/output chart unavailable: missing token columns.")
        return

    plot_df = pd.DataFrame(metrics)

    chart = (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Metric:N", title=""),
            y=alt.Y("Tokens:Q", title="Average Tokens"),
            color=alt.Color("Metric:N", legend=None),
            tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Tokens:Q", format=".2f")],
        )
        .properties(height=280)
    )

    st.altair_chart(chart, use_container_width=True)


def _render_concurrency_chart(ts_df: pd.DataFrame) -> None:
    """Render active concurrency trend over time."""
    if ts_df.empty or "concurrent_users" not in ts_df.columns:
        st.info("Concurrency trend unavailable: missing concurrent user data.")
        return

    plot_df = ts_df[["timestamp_utc", "concurrent_users"]].dropna().copy()
    if plot_df.empty:
        st.info("Concurrency trend unavailable: no valid values.")
        return

    chart = (
        alt.Chart(plot_df)
        .mark_area(opacity=0.35)
        .encode(
            x=alt.X("timestamp_utc:T", title="Request Time (UTC)"),
            y=alt.Y("concurrent_users:Q", title="Concurrent Users", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("timestamp_utc:T", title="Timestamp"),
                alt.Tooltip("concurrent_users:Q", title="Concurrent Users", format=".0f"),
            ],
        )
        .properties(height=260)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def _render_visual_analytics(df: pd.DataFrame, success_df: pd.DataFrame) -> None:
    """Render chart-driven analytics sections for better observability and UX."""
    st.markdown("### 📈 Visual Analytics")
    st.caption(
        "These charts help you quickly understand response speed, reliability, throughput, "
        "and load over time. Latency values are shown in seconds."
    )

    # Compact interpretation guide shown before charts to improve readability.
    with st.expander("ℹ️ Quick guide: how to read these metrics", expanded=False):
        st.caption(
            "Use this as a quick reference while reading charts and KPI cards. "
            "In general: lower latency is better, higher success rate is better."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**⚡ Speed Metrics**")
            st.markdown(
                "- **TTFT**: Time to first token; lower = faster perceived response.\n"
                "- **Total Latency**: End-to-end time; lower = better.\n"
                "- **Retrieval Latency**: Time spent finding context documents."
            )

        with c2:
            st.markdown("**📈 Quality & Load Metrics**")
            st.markdown(
                "- **Tokens/sec**: Generation speed; higher = faster generation.\n"
                "- **Success Rate**: Requests completed without errors.\n"
                "- **Concurrent Users**: Active in-flight requests at once."
            )

        st.info(
            "Tip: If concurrency spikes and total latency rises together, "
            "the system is likely under higher load."
        )
    st.divider()

    ts_df = _prepare_time_series(df)

    st.markdown("### ⏱️ Latency Trends")
    st.caption("Track how response times change over time. Lower lines generally indicate faster responses.")
    _render_latency_trend_chart(ts_df)

    left, right = st.columns(2)
    with left:
        st.markdown("### 📊 Latency Distribution")
        st.caption(
            "Shows how many requests fall into each latency bucket. "
            "A tighter cluster toward lower latency values is better."
        )
        _render_latency_distribution_chart(success_df if not success_df.empty else df)
    with right:
        st.markdown("### ✅ Health Composition")
        st.caption("Breakdown of successful vs failed requests for quick reliability checks.")
        _render_success_failure_chart(df)

    st.markdown("### 🚀 Throughput Over Time")
    st.caption("Tokens/sec indicates generation speed; output tokens indicates response length over time.")
    _render_throughput_trend_chart(ts_df)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🧠 Context vs Output Tokens")
        st.caption("Compares average retrieved context size against average generated response size.")
        _render_context_output_bar_chart(success_df if not success_df.empty else df)
    with c2:
        st.markdown("### 👥 Concurrency Trend")
        st.caption("Shows active concurrent requests over time. Spikes can correlate with slower responses.")
        _render_concurrency_chart(ts_df)

    st.divider()


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

    # Render chart-first visual analytics for quicker insights.
    _render_visual_analytics(df, success_df)

    # Recent requests table with human-friendly latency formatting and key metrics for observability.
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
