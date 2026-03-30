"""
Centralized UI text configuration.
Keeps user-facing Streamlit copy separate from runtime app configuration.
"""
from dataclasses import dataclass, field

# All Page-level text and labels
@dataclass(frozen=True)
class PageText:
    page_title: str = "Healthcare Symptom Checker"
    page_icon: str = "🩺"
    app_title: str = "🩺 Healthcare Symptom Information Assistant"
    load_llm_spinner: str = "🔄 Connecting to configured LLM endpoint…"
    load_vector_store_spinner: str = "⏳ Almost ready — loading health knowledge…"
    load_rag_spinner: str = "🔗 Assembling RAG pipeline…"
    header_caption_template: str = (
        "General health information assistant - Powered by **{llm_model}** OpenSource LLM Model."
    )
    chat_input_placeholder: str = (
        "Describe your symptoms here… (e.g. 'I have a persistent cough and fever for 3 days')"
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
        "<div style='text-align:center; color:grey; font-size:0.8em;'>"
        "🩺 Healthcare Symptom Information Assistant · "
        "This tool provides general health information only · "
        "Always consult a qualified healthcare professional · "
        "Sources: WHO / CDC"
        "</div>"
    )


# All sidebar text and labels
@dataclass(frozen=True)
class SidebarText:
    title: str = "## 🩺 Your Symptom Assistant"
    tagline: str = "*Get quick health guidance powered by trusted knowledge and AI*"
    disclaimer: str = (
        "**Important Health Notice**\n\n"
        "This assistant is designed to provide **general health information** and quick guidance based on trusted sources. "
        "It should **not** be used as a replacement for professional medical advice, diagnosis, or treatment.\n\n"
        "If your symptoms are severe, worsening, or urgent, please contact a qualified healthcare provider right away."
    )
    quick_guide_heading: str = "### 🔍 Quick Symptom Guide"
    structured_input_expander: str = "Use Structured Input"
    symptom_label: str = "Primary symptom:"
    primary_symptom_required_warning: str = "Please select a primary symptom before generating your question."
    duration_label: str = "How long?"
    duration_options: list[str] = field(default_factory=lambda: [
        "Select...", "Less than 1 day", "1–3 days", "4–7 days", "1–4 weeks", "More than 1 month"
    ])
    severity_label: str = "Severity (1 = mild, 10 = severe):"
    extra_symptoms_label: str = "Additional symptoms (comma-separated):"
    symptom_defaults: list[str] = field(default_factory=lambda: [
        "Select...", "Fever", "Persistent Cough", "Shortness of Breath", "Chest Pain", "Headache"
    ])
    build_query_button: str = "📋 Prepare My Symptom Question"
    build_query_success: str = "Query ready! It will appear in the chat input below ↓"
    clear_chat_button: str = "🗑️ Clear Chat History"
    about_expander: str = "ℹ️ About this app"
    about_template: str = (
        "**Healthcare Symptom Information Assistant:**\n\n"
        "- This app provides general health information using a Retrieval-Augmented Generation (RAG) pipeline backed by trusted public health data (WHO / CDC).\n\n"
        "**Tech Stack:**\n"
        "- 🤖 Configured LLM endpoint (`{llm_model}`)\n"
        "- 🔗 LangChain — RAG orchestration\n"
        "- 🔍 FAISS — local vector search\n"
        "- 🤗 all-MiniLM-L6-v2 — sentence embeddings\n"
        "- 🖥️ Streamlit — UI\n\n"
        "**Important:**\n"
        "- This tool is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional."
    )

# All chat interface text and labels
@dataclass(frozen=True)
class ChatText:
    welcome_message: str = """👋 **Hello! I'm your Health Information Assistant.**

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

# Centralized UI text configuration object
@dataclass(frozen=True)
class UIText:
    page: PageText = field(default_factory=PageText)
    sidebar: SidebarText = field(default_factory=SidebarText)
    chat: ChatText = field(default_factory=ChatText)
  
