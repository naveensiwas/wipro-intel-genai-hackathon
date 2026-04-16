# 📋 Module Breakdown

> [← Back to README](../README.md)

---

## 🎨 UI Layer (`src/app/ui/`)

### `main.py` — Application Entry Point
- Sets Streamlit page config (`page_title="Healthcare Symptom Checker"`, icon `🩺`, wide layout)
- Three `@st.cache_resource` loaders: `load_llm()`, `load_vector_store()`, `load_rag_chain()` — each cached for process lifetime
- Routes between `"chat"` and `"metrics"` views via `st.session_state["current_view"]`
- Tracks `st.session_state.active_requests` for concurrent user counting
- Generates a unique `request_key = f"msg-{int(request_started * 1e9)}"` per request to avoid Streamlit duplicate element key errors
- Supports prefilled queries from the sidebar Smart Symptom Guide via `st.session_state["prefilled_query"]`
- Greeting fast-path: simple greetings are handled via `is_simple_greeting()` and `get_greeting_response()` without retrieval or LLM invocation
- Domain-gated routing: non-health inputs are blocked via `is_health_related_with_mode()` before RAG execution
- On-domain flow: `retrieve_sources()` + `rag_chain.invoke()` with history-aware query built via `_build_history_aware_query()`
- Safety post-processing: generated responses are passed through `sanitize_response()` before display

### `chat_interface.py` — Multi-Turn Chat State
- `init_chat_state()`: Creates `st.session_state["messages"]` with a welcome message on first load; subsequent reruns preserve full history
- `render_chat_history()`: Iterates all messages with `st.chat_message()`; assistant uses `🩺` avatar, user uses `👤`; renders source expanders per message with indexed keys to avoid conflicts
- `add_message(role, content, sources)`: Appends `{"role", "content", "sources"}` dicts to session state

> ✅ **Note:** Chat history is persisted in session state, and recent turns are incorporated in `main.py` via `_build_recent_history()` and `_build_history_aware_query()` to support follow-up question resolution.

| | |
|:---:|:---:|
| ![Chat Interface 1](screenshots/chat_interface_1.png) | ![Chat Interface 2](screenshots/chat_interface_2.png) |
|                                            Chat-screen with welcome message                                             |            Chat screen with user question response            |
| Smart Symptom Guide in sidebar with structured query form | Chat screen showing retrieved sources (expanders) |

### `sidebar.py` — Navigation & Smart Symptom Guide
- Renders Wipro and Intel logos side-by-side using `st.columns`
- Two-button navigation (`💬 Chat` / `📊 Metrics`) with `st.rerun()` on switch
- **Smart Symptom Guide** (`🔍 Smart symptom guide`):
  - Dynamic `st.selectbox` of symptoms loaded from `symptoms_conditions.json` at startup (falls back to 5 defaults on error)
  - Duration `st.selectbox` with 5 timeframe options
  - Severity `st.slider` (1–10)
  - Free-text additional symptoms field
  - "📋 Prepare Symptom Details" button composes a structured natural-language query and stores in `st.session_state["prefilled_query"]`
- Clear chat button resets messages to the welcome message and removes any prefilled query

### `metrics_dashboard.py` — Performance Dashboard
- Loads `data/runtime/metrics/performance_metrics.json`; gracefully handles missing/corrupt files via `_load_metrics_payload()`
- `_build_dataframe()`: Converts request list to a typed pandas DataFrame; coerces numeric columns and parses `timestamp_utc` to UTC-aware datetimes
- `_get_health_status()`: Returns a color-coded label — 🟢 Healthy (≥80% success rate) / 🟡 Degraded (≥60%) / 🔴 Unhealthy (<60%)
- `_prepare_time_series()`: Produces a UTC-sorted copy of the DataFrame with ms → sec conversion columns for charting (`ttft_ms_sec`, `retrieval_latency_ms_sec`, `total_latency_ms_sec`)
- `_format_timestamp_for_display()`: Converts UTC → IST (`Asia/Kolkata`), formats as `Ddd, DD Mon YYYY HH:MM AM/PM`
- `_ms_to_seconds()` / `_format_duration_from_ms()`: Utility helpers for unit conversion and `X.XX sec` display strings

**Chart-first layout** — `_render_visual_analytics()` is called before the requests table so aggregate trends are immediately visible.

**Visual Analytics — `_render_visual_analytics(df, success_df)`**

A collapsible **ℹ️ Quick guide** expander above the charts explains each metric category to reduce interpretation burden:
- Speed metrics (TTFT, Total Latency, Retrieval Latency) — lower is better
- Quality & load metrics (Tokens/sec, Success Rate, Concurrent Users)

Six Altair charts rendered within this section:

| Function | Chart Type | Metrics |
|----------|-----------|---------|
| `_render_latency_trend_chart(ts_df)` | Multi-series line + points (interactive) | TTFT, Retrieval, Total Latency over time (sec) |
| `_render_latency_distribution_chart(df)` | Histogram (`maxbins=24`) | Total Latency distribution (sec), successful requests |
| `_render_success_failure_chart(df)` | Donut chart (`innerRadius=55`) | Successful (#16a34a) vs Failed (#dc2626) request counts |
| `_render_throughput_trend_chart(ts_df)` | Multi-series line + points (interactive) | Tokens/sec and Output Tokens over time |
| `_render_context_output_bar_chart(df)` | Grouped bar | Avg Context Size vs Avg Output Tokens |
| `_render_concurrency_chart(ts_df)` | Area chart (opacity=0.35, interactive) | Concurrent users over time |

Each chart function handles missing/empty data gracefully with `st.info()` fallback messages.

**Recent Requests Table**
- Last 50 requests sorted descending; latency columns converted to `X.XX sec` display strings
- Timestamps converted UTC → IST via `_format_timestamp_for_display()`
- `SR No.` column added as 1-based sequential index
- Rendered as a custom styled HTML table via pandas `Styler`: blue header (`#1e3a8a` on `#eff6ff`), alternating row stripes (`#f8fbff`), horizontal scroll wrapper

**Controls**
- CSV download button (exports raw ms values; produced by `_to_csv_bytes()`)
- Reset popover (requires checkbox confirmation to prevent accidental reset; calls `reset_metrics()`)


|                                                                                |                                                                  |
|:------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
|        ![Metrics Dashboard 1](screenshots/metrics_dashboard_1.png)        | ![Metrics Dashboard 2](screenshots/metrics_dashboard_2.png) |
| Health status and controls to download and reset the recent user conversations |       Latency trends, distribution and health composition        |

|                                                                  |                                                                  |
|:----------------------------------------------------------------:|:----------------------------------------------------------------:|
| ![Metrics Dashboard 3](screenshots/metrics_dashboard_3.png) | ![Metrics Dashboard 4](screenshots/metrics_dashboard_4.png) |
| Throughput, context vs output token charts and concurrency trend |                 User recent conversation history                 |

### `styles.py` — Centralized CSS
- `StreamlitStyles.apply_all_styles()`: Injects CSS for page titles, taglines, sidebar branding, expander styling, button colors, footer
- `StreamlitStyles.render_definition_group()`: Renders icon + label + description rows inside metric expanders

### `source_utils.py` — Source Formatting
- `format_source_label(meta, fallback)`: Produces human-readable source labels from document metadata (`symptom`, `condition`, `category` fields)
- `truncate_source_content(content, max_len)`: Clips content for UI rendering

---

## 🧠 RAG Pipeline (`src/app/rag/`)

### `data_loader.py` — Knowledge Base Ingestion

Each JSON file is loaded and converted to structured `langchain_core.documents.Document` objects:

| Loader | JSON File | Document Format | Metadata Keys |
|--------|-----------|-----------------|---------------|
| `load_symptom_documents()` | `symptoms_conditions.json` | `Symptom / Body System / Description / Conditions / Source` | `source_file`, `symptom`, `body_system` |
| `load_condition_documents()` | `conditions_info.json` | `Condition / Description / Causes / Symptoms / Care Tips / When to Seek Care / Specialist / Prevention / Source` | `source_file`, `condition`, `specialist` |
| `load_preventive_documents()` | `preventive_tips.json` | `Category / Tips (bulleted) / Source` | `source_file`, `category` |
| `load_all_documents()` | All three | Combined list | All of the above |

**Knowledge Base JSON Schemas:**

`symptoms_conditions.json`:
- `symptom` (str), `body_system` (str), `description` (str), `related_conditions` (list[str]), `source` (str)

`conditions_info.json`:
- `name`, `common_name`, `description`, `common_causes` (list), `typical_symptoms` (list), `general_care_tips` (list), `when_to_seek_care`, `specialist_type`, `prevention` (list), `source`

`preventive_tips.json`:
- `category` (str), `tips` (list[str]), `source` (str)

### `embedder.py` — Embedding Model
- Initializes `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2`
- Default path: `./assets/models/all-MiniLM-L6-v2` (bundled locally)
- Overridable via `EMBEDDING_MODEL` env var (any HuggingFace model ID or local path)
- CPU-only inference; 384-dimensional L2-normalized output vectors

### `vector_store.py` — FAISS Index Management
- `_split_documents()`: Uses `RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", " ", ""])`; logs first-few-chunk previews at DEBUG level
- `build_or_load_vector_store()`:
  - **Fast path** (index exists): Loads from `VECTOR_STORE_DIR` in ~1-2 seconds
  - **Slow path** (first run): Documents → chunks → embed → `FAISS.from_documents()` → save to disk (~1-2 minutes)
  - Uses `allow_dangerous_deserialization=True` for local trusted index loading

### `retriever.py` — RAG Chain & Similarity Scoring
- `build_rag_chain(llm, vector_store)`: Assembles `RetrievalQA.from_chain_type(chain_type="stuff", return_source_documents=True)` with `RAG_PROMPT`
- `retrieve_sources(vector_store, query)`: Retrieves top-K docs; truncates content to 300 chars for UI display; returns `list[{content, metadata}]`
- `get_top_similarity_score(vector_store, query, k)`:
  - Calls `similarity_search_with_score` to get raw FAISS scores
  - **Score normalization**: If all scores are in `[0.0, 1.0]` → use as-is (similarity); if any score > 1.0 → treat as distance, transform: `similarity = 1 / (1 + distance)`
  - Clamps final score to `[0.0, 1.0]`

---

## 🤖 LLM Integration (`src/app/llm/`)

### `endpoint_client.py` — Dual Backend Clients

Both clients share the `EndpointResponse(content, raw)` return type and a `_extract_content()` helper that handles both OpenAI-style `choices[0].message.content` and TGI-style `[0].generated_text` responses.

**`SageMakerEndpointClient`** (when `LLM_MODE=sagemaker`):
- Creates a named `boto3.Session(profile_name, region_name)` → `sagemaker-runtime` client
- TGI payload: `{"inputs": prompt, "parameters": {"temperature", "max_new_tokens", "return_full_text": false}}`
- Raises `RuntimeError` wrapping `ClientError` / `BotoCoreError` with descriptive messages

**`LlamaEndpointClient`** (when `LLM_MODE=llama`):
- Posts to `{base_url}/chat/completions` with OpenAI chat completions payload: `{"model", "messages": [{"role": "user", "content": prompt}], "temperature", "max_tokens"}`
- Sets `Authorization: Bearer {api_key}` header only if `api_key` is non-empty and not `"dummy"`
- Raises specific exceptions for `HTTPError`, `ConnectionError`, and `Timeout`

### `langchain_adapter.py` — `EndpointLLM`
- Extends `langchain_core.language_models.llms.LLM` (Pydantic v2, `ConfigDict(arbitrary_types_allowed=True)`)
- Fields: `client: Any`, `llm_mode: str`, `llm_name: str`
- `_call()`: Delegates to `client.invoke(prompt)`; applies `stop` token splitting if provided
- `_llm_type` property returns `f"endpoint_{llm_mode}"` for LangChain tracing
- `_identifying_params` returns `{llm_mode, llm_name}` for caching/logging

### `model_loader.py` — Factory
- `@lru_cache(maxsize=1)` ensures the LLM is initialized exactly once per process
- Raises `ValueError` for unknown `LLM_MODE` values
- Both clients initialized with: `temperature`, `max_tokens`, `timeout`, `verbose` from `cfg`

### `prompt_templates.py` — `RAG_PROMPT`

A `PromptTemplate` with input variables `{context}` and `{question}`:

**8 Grounding Rules:**
1. Ground in provided `CONTEXT`; no external medical facts
2. Connect and explain context — don't just quote it
3. If insufficient context: say so and recommend professional consultation
4. Handle greetings warmly and invite a health question
5. For non-health, non-greeting queries: redirect politely
6. No diagnoses, prescriptions, or personalized treatment plans
7. Use cautious phrasing: *"may be associated with"*, *"could be related to"*
8. For severe/life-threatening symptoms: strongly recommend urgent/emergency care

**10 Style Rules:**
1. Respond naturally and conversationally
2. Start directly with helpful content — no filler phrases
3. Stay focused and concise — don't over-explain
4. Forbidden phrases: *"Based on your description"*, *"According to the context"*
5. Use bullet points for lists; bold for critical warnings only
6. Strict bullet format: each bullet on its own line starting with `"- "`
7. Implicitly adapt tone to urgency — never show severity scores
8. Calm tone for mild; direct tone for serious — through natural language only
9. No random/excessive formatting
10. Headings/subheadings optional but allowed for clarity

---

## 🔒 Safety & Filtering (`src/app/safety/`)

### `safety_filter.py` — Full Safety Pipeline

**Input Domain Gate** — `is_health_related_with_mode(query, vector_store)`:

```
DOMAIN_FILTER_MODE
     │
     ├── "keyword" ──────────────────────→ keyword check only → (bool, None)
     │
     ├── "semantic" ──→ semantic check
     │                      │ pass → (True, score)
     │                      │ fail + KEYWORD_FALLBACK=true → keyword check
     │                      │         │ pass → (True, score)
     │                      │         └ fail → (False, score)
     │                      └ fail + KEYWORD_FALLBACK=false → (False, score)
     │
     └── "hybrid" ────→ semantic check
                            │ pass → (True, score)
                            └ fail + KEYWORD_FALLBACK=true → keyword check
                                        │ pass → (True, score)
                                        └ fail → (False, score)
```

**Health Keywords** focus on medical/symptom terminology for domain gating; greeting handling is performed separately using greeting detection utilities.

**Semantic Gate** — `is_health_related_semantic(query, vector_store)`:
- Calls `get_top_similarity_score()` with `DOMAIN_SIMILARITY_K` docs (default: 1)
- Returns `(True, score)` if `score >= DOMAIN_SIMILARITY_THRESHOLD` (default: 0.40)
- Falls back to keyword check if FAISS raises an exception

**Output Sanitization** — `sanitize_response(text)`:

| Pattern | Replacement |
|---------|-------------|
| `"You have X"` | `"This may be associated with X"` |
| `"you have X"` | `"this may be associated with X"` |
| `"You are diagnosed with X"` | `"The symptoms described could be related to X"` |
| `"you are diagnosed with X"` | `"the symptoms described could be related to X"` |
| `"You definitely have X"` | `"The information suggests a possible association with X"` |
| `"you definitely have X"` | `"the information suggests a possible association with X"` |
| `"This is clearly X"` | `"This may be X"` |
| `"this is clearly X"` | `"this may be X"` |
| `"You suffer from X"` | `"You may be experiencing symptoms associated with X"` |
| `"you suffer from X"` | `"you may be experiencing symptoms associated with X"` |

- Each replacement is logged as a `WARNING` with the pattern, replacement, and count
- **Mandatory disclaimer** always appended to every response:
  > `⚕️ This information is for general educational purposes only and does not constitute medical advice, diagnosis, or treatment. Please consult a qualified healthcare professional for personalised medical guidance.`

---

## 🛠️ Core Services (`src/app/core/`)

### `logging.py` — Structured Colored Logging
- `StructuredFormatter`: Produces millisecond-precision, pipe-delimited, ANSI-colored output:
  ```
  2026-04-08 10:32:45.123 | ℹ️  INFO     | app.rag.vector_store      | FAISS index loaded
  ```
- Per-level colors: DEBUG=Blue, INFO=Green, WARNING=Yellow, ERROR=Red, CRITICAL=Bright Red
- Per-level emojis: 🔧 DEBUG, ℹ️ INFO, ⚠️ WARNING, ❌ ERROR, 🚨 CRITICAL
- `ContextFilter`: Injects `job_id` and `document_id` into records (for job-level tracing)
- Helper functions: `get_logger()`, `set_context()`, `log_section()`, `log_step()`, `log_success()`, `log_error()`, `log_warning()`

### `metrics.py` — Thread-Safe JSON Metrics
- Module-level `_METRICS_LOCK = threading.Lock()` protects all file writes
- `_ensure_metrics_file_exists()`: Creates parent directories and initializes default JSON structure on first run
- `build_metrics_entry()`: Computes all metric values from raw timing floats (converts `perf_counter` floats to ms)
- `persist_metrics(entry)`: Acquires lock → reads file → appends entry → updates rolling summary → writes atomically
- `reset_metrics()`: Acquires lock → overwrites file with default empty payload
- **Token approximation**: `len(text.split())` — word count used as proxy for token count (no tokenizer import)

### `error_handling.py` — Exception Utilities
- Wraps exceptions with user-friendly messages for Streamlit display
- Used by `main.py` to render `st.error()` messages with the error template from `ui_text.py`

---

## ⚙️ Configuration (`src/app/config/`)

### `settings.py` — `AppConfig` Dataclass
- Centralized configuration with typed fields for all settings (LLM, embedding, vector store, metrics)
- Environment-configurable fields resolved at import time via `os.getenv()` with typed defaults; some internal constants (e.g. chunking, view labels) are defined directly in code
- `_REPO_ROOT = Path(__file__).resolve().parents[3]` — resolves all file paths relative to repo root regardless of working directory
- Singleton instance `cfg` imported across all modules for consistent configuration access
- Full reference: → [CONFIGURATION.md](CONFIGURATION.md)

### `ui_text.py` — `UIText` Dataclass Hierarchy
- `PageText` (frozen): All page-level strings — titles, spinners, placeholders, error templates, footer, dashboard labels, metric definitions
- `SidebarText` (frozen): Sidebar branding, navigation labels, Smart Symptom Guide options, about template
- `ChatText` (frozen): Welcome message template with `{smart_symptom_guide}` placeholder
- `UIText`: Composes all three via `field(default_factory=...)`; singleton `cfg.ui_text` accessed everywhere

---

*[← Back to README](../README.md)*

