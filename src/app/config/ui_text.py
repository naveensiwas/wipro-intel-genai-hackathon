"""
Centralized UI text configuration.
Keeps user-facing Streamlit copy separate from runtime app configuration.
"""
from dataclasses import dataclass, field

# All Page-level text and labels
@dataclass(frozen=True)
class PageText:
    # ── Page metadata ──
    page_title: str = "Healthcare Symptom Checker"
    page_icon: str = "🩺"

    # ── Page titles and branding ──
    app_title_html: str = ("<div class='page-title-main'>"
                    "🩺 Healthcare Symptom Information Assistant"
                    "</div>")
    dashboard_app_title_html: str = ("<div class='page-title-main'>"
                    "📊 Model Performance & Evaluation Dashboard"
                    "</div>")
    dashboard_no_records: str = "📈 No metrics available yet. Run a query in the main chat interface to generate performance metrics."
    tagline_html: str = ("<div class='page-tagline'>"
                    "Health insights powered by AI with efficient execution on Intel® CPUs"
                    "</div>")

    # ── Chat interface spinners and messages ──
    load_llm_spinner: str = "🔄 Connecting to configured LLM endpoint…"
    load_vector_store_spinner: str = "⏳ Almost ready — loading health knowledge…"
    load_rag_spinner: str = "🔗 Assembling RAG pipeline…"
    chat_input_placeholder: str = (
        "Describe your symptoms here… (e.g., 'Persistent cough and fever for the past 3 days')"
    )
    response_spinner: str = "🔎 Looking up health information and preparing your answer…"
    query_log_template: str = "🔍 Invoking RAG pipeline with query: '{query}'"
    source_expander_label: str = "📚 View retrieved health knowledge sources"
    source_fallback_label: str = "Health Data"
    error_message_template: str = (
        "⚠️ An error occurred while generating the response: `{error}`\n\n"
        "Please try rephrasing your question or verify the configured LLM endpoint settings."
    )
    footer_html: str = (
        "<div class='page-footer'>"
        "<span class='footer-disclaimer'>⚠️ General health information only · Not a medical diagnosis · Consult a qualified healthcare professional · Sources: WHO, CDC</span>\n\n"
        "</div>"
    )

    # ── Metrics dashboard controls ──
    metrics_download_button: str = "📥 Download Metrics (CSV)"
    metrics_reset_popover: str = "🔄 Reset Metrics"
    metrics_reset_warning: str = "⚠️ This will permanently clear all saved metrics."
    metrics_reset_checkbox: str = "I understand this will clear all saved metrics"
    metrics_confirm_reset_button: str = "🔴 Confirm Reset"
    metrics_reset_success: str = "✅ Metrics reset completed."

    # ── Summary metrics expander ──
    summary_expander_title: str = "📊 Summary metrics"
    summary_help_html: str = "<h5 class='metrics-section-title'>📖 What these metrics mean:</h5>"
    summary_total_requests_label: str = "Total Requests"
    summary_successful_requests_label: str = "Successful Requests"
    summary_avg_total_latency_label: str = "Avg Total Latency (avg sec)"
    summary_max_concurrent_label: str = "Max Concurrent Users"

    # ── Latency metrics expander ──
    latency_expander_title: str = "⏱️ Latency averages (seconds)"
    latency_end_to_end_label: str = "End-to-End Latency (avg sec)"
    latency_retrieval_label: str = "Retrieval Latency (avg sec)"
    latency_total_label: str = "Total Latency (avg sec)"
    latency_ttft_label: str = "TTFT (avg sec)"

    # ── Throughput metrics expander ──
    throughput_expander_title: str = "🚀 Throughput & context averages"
    throughput_tokens_per_sec_label: str = "Tokens/sec (avg)"
    throughput_context_size_label: str = "Context Size (avg tokens)"
    throughput_output_tokens_label: str = "Output Tokens (avg)"
    throughput_concurrent_users_label: str = "Concurrent Users (avg)"

    # ── Recent requests table ──
    recent_requests_title_html: str = "<h2 class='metrics-recent-requests-title'>📋 Recent Requests</h2>"
    recent_requests_subtitle_html: str = "<p class='metrics-recent-requests-subtitle'>Last 50 requests with detailed performance metrics</p>"

# All sidebar text and labels
@dataclass(frozen=True)
class SidebarText:
    # ── Sidebar branding ──
    title: str = ("<div class='logo-title'>"
                    "🩺 AI‑Powered Symptom Information Assistant"
                    "</div>")

    # ── Quick symptom guide (structured input) ──
    quick_guide_heading: str = "🔍 Smart symptom guide"
    structured_input_expander: str = "Provide your symptom details"
    symptom_label: str = "Primary symptom:"
    symptom_defaults: list[str] = field(default_factory=lambda: [
        "Choose an option…", "Fever", "Persistent Cough", "Shortness of Breath", "Chest Pain", "Headache"
    ])
    primary_symptom_required_warning: str = "Please select a primary symptom before generating the query."
    duration_label: str = "Duration of symptoms:"
    duration_options: list[str] = field(default_factory=lambda: [
        "Select timeframe…", "Less than 1 day", "1–3 days", "4–7 days", "1–4 weeks", "More than 1 month"
    ])
    severity_label: str = "Symptom severity (1 = mild, 10 = severe):"
    extra_symptoms_label: str = "Additional symptoms (separated by commas):"
    build_query_button: str = "📋 Prepare Symptom Details"
    build_query_success: str = "Your query is ready and will appear in the chat input below ↓"

    # ── Chat controls ──
    clear_chat_button: str = "🗑️ Clear Conversation History"

    # ── Sidebar navigation ──
    navigation_heading: str = "🔽 Choose an option to continue"
    chat_menu_button: str = "💬 Chat — Describe Your Symptoms"
    metrics_menu_button: str = "📊 Model Latency & Throughput Metrics"

    # ── About section ──
    about_expander: str = "ℹ️ About this application"
    about_template: str = (
        "**Healthcare Symptom Information Assistant:**\n\n"
        "- This application provides general health information using a RAG‑based approach with trusted sources like WHO and CDC, delivering clear symptom insights through Intel® CPU‑optimized AI inference without requiring GPUs.\n\n"
        "**Technology Stack:**\n"
        "- 🤖 LLM: `{llm_model}`\n"
        "- 🔗 LangChain: RAG orchestration and prompt management\n"
        "- 🔍 FAISS: Local vector‑based similarity search\n"
        "- 🤗 all‑MiniLM‑L6‑v2: Sentence embeddings\n"
        "- 🖥️ Streamlit: Interactive user interface\n"
        "- ⚙️ Intel® CPU: Optimized for fast, low‑latency inference on standard compute.\n\n"
        "**Important Notice:**\n"
        "- This tool provides general health information only and is not a substitute for professional medical advice. Always consult a qualified healthcare professional for medical concerns."
    )

# All chat interface text and labels
@dataclass(frozen=True)
class ChatText:
    # ── Chat welcome message ──
    welcome_message: str = """👋 **Welcome! I'm your Health Information Assistant**

I can help you with:
- 🤒 Understanding conditions that may be associated with your symptoms
- 💊 General wellness and self‑care tips
- 🏥 Knowing when it's important to seek medical attention
- 👨‍⚕️ Identifying the right type of specialist to consult

**How to get started:**
- Type your symptoms directly in the chat below, or
- Use the **{smart_symptom_guide}** in the sidebar to create a structured query

⚠️ *Please note: I provide general health information only and do not offer medical diagnoses. Always consult a qualified healthcare professional for medical concerns.*

---
**What symptoms are you experiencing today?**
"""

# Centralized UI text configuration object
@dataclass(frozen=True)
class UIText:
    page: PageText = field(default_factory=PageText)
    sidebar: SidebarText = field(default_factory=SidebarText)
    chat: ChatText = field(default_factory=ChatText)
