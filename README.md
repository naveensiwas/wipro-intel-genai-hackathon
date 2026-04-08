# 🩺 Healthcare Symptom Information Assistant

> **Wipro × Intel Hackathon — April 2026**

A safe, trustworthy, and informative conversational AI system that helps users understand health-related symptoms through a **Retrieval-Augmented Generation (RAG)** pipeline. Built with **Streamlit**, powered by configurable LLM backends (AWS SageMaker or Llama HTTP endpoints), and grounded in a curated healthcare knowledge base.

> ⚕️ **Medical Disclaimer:** This assistant provides **general health information only** and explicitly avoids diagnosis, prescription, or personalized medical advice. Always consult a qualified healthcare professional for medical guidance. Sources: WHO, CDC.

---

## 🎯 Key Features

| Feature | Detail |
|---------|--------|
| **Conversational Chat** | Multi-turn chat with full persistent history via Streamlit session state; welcome message on first load |
| **RAG-Powered Responses** | All answers grounded in a curated knowledge base of symptoms, conditions, and preventive tips |
| **Semantic Search** | FAISS vector store + local `all-MiniLM-L6-v2` embeddings; CPU-only, no GPU required |
| **Smart Symptom Guide** | Sidebar form to build structured queries from symptom, duration, severity, and additional symptoms |
| **Dual Safety Layers** | Hybrid keyword + semantic input domain gate; output regex sanitization + mandatory disclaimer |
| **Flexible LLM Backend** | AWS SageMaker (TGI via boto3 + SSO) or Llama HTTP (OpenAI-compatible `/v1/chat/completions`) |
| **Performance Monitoring** | Per-request TTFT, total/retrieval latency, tokens/sec, concurrent users — stored as JSON |
| **Metrics Dashboard** | In-app dashboard with health status indicator, latency/throughput panels, IST-formatted request table, CSV export, and reset |
| **Transparent Source Attribution** | Retrieved context documents shown in collapsible expanders next to each response |
| **Production-Ready Architecture** | Modular package structure, centralized config dataclass, ANSI-colored structured logging, thread-safe metrics |

---

## 🏗️ Architecture Overview

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Web Interface                           │
│  ┌───────────────┐  ┌─────────────────────────┐  ┌────────────────────┐ │
│  │  Chat Window  │  │         Sidebar          │  │ Metrics Dashboard  │ │
│  │  • Messages   │  │  • Navigation (2 views)  │  │  • Health Status   │ │
│  │  • Sources    │  │  • Smart Symptom Guide   │  │  • Latency Panels  │ │
│  │  • Avatars    │  │  • Wipro / Intel Logos   │  │  • Request Table   │ │
│  └───────────────┘  │  • About / Clear Chat    │  │  • CSV Export      │ │
│                     └─────────────────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
                              ▲ st.session_state ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        Main Application Logic (main.py)                 │
│  • @st.cache_resource — LLM, vector store, RAG chain loaded once        │
│  • Session state: messages, active_requests, current_view              │
│  • Prefilled query support from sidebar structured input                │
│  • Per-request timing accumulators → metrics entry on completion        │
└──────────────────────────────────────────────────────────────────────────┘
         ▲             ▲              ▲             ▲              ▲
    ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐
    │ Safety  │  │   RAG    │  │ Metrics  │  │ Logging │  │   Config     │
    │ Filter  │  │ Pipeline │  │ Tracker  │  │ System  │  │  (AppConfig) │
    └─────────┘  └──────────┘  └──────────┘  └─────────┘  └──────────────┘
                      ▲
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
  ┌──────────┐  ┌──────────┐  ┌───────────┐
  │   LLM    │  │  Vector  │  │   Data    │
  │ Adapter  │  │  Store   │  │  Loader   │
  │(EndpointL│  │ (FAISS)  │  │           │
  │   LM)    │  └──────────┘  └───────────┘
  └────┬─────┘       ▲               ▲
       │             │               │
       ▼             ▼               ▼
  ┌─────────────┐ ┌──────────────┐ ┌────────────────────────┐
  │  LLM Server │ │  Embeddings  │ │  Healthcare Knowledge  │
  │ AWS/HTTP    │ │  all-MiniLM  │ │  • symptoms_conditions │
  └─────────────┘ │  L6-v2       │ │  • conditions_info     │
                  │  (Local CPU) │ │  • preventive_tips     │
                  └──────────────┘ └────────────────────────┘
```

### Request Data Flow

```
User types message (or uses Smart Symptom Guide to compose query)
    │
    ▼
┌────────────────────────────────────┐
│  Domain Gate: is_health_related_  │
│  with_mode(query, vector_store)    │
│  Mode: keyword | semantic | hybrid │
└────────────────────────────────────┘
         │ ALLOWED            BLOCKED → off-domain warning shown
         ▼
┌──────────────────────────┐
│  retrieve_sources()      │
│  FAISS top-K similarity  │
│  → list[{content, meta}] │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  rag_chain.invoke()      │
│  RetrievalQA (stuff)     │
│  → RAG_PROMPT filled     │
│    with context+question │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  LLM generates response  │
│  (SageMaker or Llama)    │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  sanitize_response()     │
│  • Regex replacements    │
│  • Appends ⚕️ disclaimer │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  build_metrics_entry()   │
│  persist_metrics()       │
│  (thread-safe JSON write)│
└──────────────────────────┘
         │
         ▼
Response + Source Expander shown in chat
```

---

## 📋 Module Breakdown

### 🎨 UI Layer (`src/app/ui/`)

#### `main.py` — Application Entry Point
- Sets Streamlit page config (`page_title="Healthcare Symptom Checker"`, icon `🩺`, wide layout)
- Three `@st.cache_resource` loaders: `load_llm()`, `load_vector_store()`, `load_rag_chain()` — each cached for process lifetime
- Routes between `"chat"` and `"metrics"` views via `st.session_state["current_view"]`
- Tracks `st.session_state.active_requests` for concurrent user counting
- Generates a unique `request_key = f"msg-{int(request_started * 1e9)}"` per request to avoid Streamlit duplicate element key errors
- Supports prefilled queries from the sidebar Smart Symptom Guide via `st.session_state["prefilled_query"]`

#### `chat_interface.py` — Multi-Turn Chat State
- `init_chat_state()`: Creates `st.session_state["messages"]` with a welcome message on first load; subsequent reruns preserve full history
- `render_chat_history()`: Iterates all messages with `st.chat_message()`; assistant uses `🩺` avatar, user uses `👤`; renders source expanders per message with indexed keys to avoid conflicts
- `add_message(role, content, sources)`: Appends `{"role", "content", "sources"}` dicts to session state

#### `sidebar.py` — Navigation & Smart Symptom Guide
- Renders Wipro and Intel logos side-by-side using `st.columns`
- Two-button navigation (`💬 Chat` / `📊 Metrics`) with `st.rerun()` on switch
- **Smart Symptom Guide** (`🔍 Smart symptom guide`):
  - Dynamic `st.selectbox` of symptoms loaded from `symptoms_conditions.json` at startup (falls back to 5 defaults on error)
  - Duration `st.selectbox` with 5 timeframe options
  - Severity `st.slider` (1–10)
  - Free-text additional symptoms field
  - "📋 Prepare Symptom Details" button composes a structured natural-language query and stores in `st.session_state["prefilled_query"]`
- Clear chat button resets messages to the welcome message and removes any prefilled query

#### `metrics_dashboard.py` — Performance Dashboard
- Loads `data/runtime/metrics/performance_metrics.json`; gracefully handles missing/corrupt files
- **Health Status Indicator**: 🟢 Healthy (≥80% success rate) / 🟡 Degraded (≥60%) / 🔴 Unhealthy (<60%)
- Dashboard panels (each in a collapsible `st.expander`):
  - **Summary**: Total requests, successful requests, avg total latency (sec), max concurrent users
  - **Latency Averages**: End-to-end, retrieval, total, TTFT — all converted from ms → sec for display
  - **Throughput & Context**: Tokens/sec, context size (tokens), output tokens, avg concurrent users
- **Recent Requests Table**: Last 50 requests sorted descending; timestamps converted from UTC → IST (`Asia/Kolkata`); styled HTML table with blue header and alternating row stripes
- **Controls**: CSV download button + reset popover (requires checkbox confirmation to prevent accidental reset)

#### `styles.py` — Centralized CSS
- `StreamlitStyles.apply_all_styles()`: Injects CSS for page titles, taglines, sidebar branding, expander styling, button colors, footer
- `StreamlitStyles.render_definition_group()`: Renders icon + label + description rows inside metric expanders

#### `source_utils.py` — Source Formatting
- `format_source_label(meta, fallback)`: Produces human-readable source labels from document metadata (`symptom`, `condition`, `category` fields)
- `truncate_source_content(content, max_len)`: Clips content for UI rendering

---

### 🧠 RAG Pipeline (`src/app/rag/`)

#### `data_loader.py` — Knowledge Base Ingestion

Each JSON file is loaded and converted to structured `langchain_core.documents.Document` objects with rich `page_content` text and typed `metadata`:

| Loader | JSON File | Document Format | Metadata Keys |
|--------|-----------|-----------------|---------------|
| `load_symptom_documents()` | `symptoms_conditions.json` | `Symptom / Body System / Description / Conditions / Source` | `source_file`, `symptom`, `body_system` |
| `load_condition_documents()` | `conditions_info.json` | `Condition / Description / Causes / Symptoms / Care Tips / When to Seek Care / Specialist / Prevention / Source` | `source_file`, `condition`, `specialist` |
| `load_preventive_documents()` | `preventive_tips.json` | `Category / Tips (bulleted) / Source` | `source_file`, `category` |
| `load_all_documents()` | All three | Combined list | All of the above |

#### `embedder.py` — Embedding Model
- Initializes `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2`
- Default path: `./assets/models/all-MiniLM-L6-v2` (bundled locally)
- Overridable via `EMBEDDING_MODEL` env var (any HuggingFace model ID or local path)
- CPU-only inference; 384-dimensional L2-normalized output vectors

#### `vector_store.py` — FAISS Index Management
- `_split_documents()`: Uses `RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", " ", ""])`; logs first-few-chunk previews at DEBUG level
- `build_or_load_vector_store()`:
  - **Fast path** (index exists): Loads from `VECTOR_STORE_DIR` in ~1-2 seconds
  - **Slow path** (first run): Documents → chunks → embed → `FAISS.from_documents()` → save to disk (~1-2 minutes)
  - Uses `allow_dangerous_deserialization=True` for local trusted index loading

#### `retriever.py` — RAG Chain & Similarity Scoring
- `build_rag_chain(llm, vector_store)`: Assembles `RetrievalQA.from_chain_type(chain_type="stuff", return_source_documents=True)` with `RAG_PROMPT`
- `retrieve_sources(vector_store, query)`: Retrieves top-K docs; truncates content to 300 chars for UI display; returns `list[{content, metadata}]`
- `get_top_similarity_score(vector_store, query, k)`:
  - Calls `similarity_search_with_score` to get raw FAISS scores
  - **Score normalization**: If all scores are in `[0.0, 1.0]` → use as-is (similarity); if any score > 1.0 → treat as distance, transform: `similarity = 1 / (1 + distance)`
  - Clamps final score to `[0.0, 1.0]`

---

### 🤖 LLM Integration (`src/app/llm/`)

#### `endpoint_client.py` — Dual Backend Clients

Both clients share the `EndpointResponse(content, raw)` return type and a `_extract_content()` helper that handles both OpenAI-style `choices[0].message.content` and TGI-style `[0].generated_text` responses.

**`SageMakerEndpointClient`** (when `LLM_MODE=sagemaker`):
- Creates a named `boto3.Session(profile_name, region_name)` → `sagemaker-runtime` client
- TGI payload: `{"inputs": prompt, "parameters": {"temperature", "max_new_tokens", "return_full_text": false}}`
- Raises `RuntimeError` wrapping `ClientError` / `BotoCoreError` with descriptive messages

**`LlamaEndpointClient`** (when `LLM_MODE=llama`):
- Posts to `{base_url}/chat/completions` with OpenAI chat completions payload: `{"model", "messages": [{"role": "user", "content": prompt}], "temperature", "max_tokens"}`
- Sets `Authorization: Bearer {api_key}` header only if `api_key` is non-empty and not `"dummy"`
- Raises specific exceptions for `HTTPError`, `ConnectionError`, and `Timeout`

#### `langchain_adapter.py` — `EndpointLLM`
- Extends `langchain_core.language_models.llms.LLM` (Pydantic v2, `ConfigDict(arbitrary_types_allowed=True)`)
- Fields: `client: Any`, `llm_mode: str`, `llm_name: str`
- `_call()`: Delegates to `client.invoke(prompt)`; applies `stop` token splitting if provided
- `_llm_type` property returns `f"endpoint_{llm_mode}"` for LangChain tracing
- `_identifying_params` returns `{llm_mode, llm_name}` for caching/logging

#### `model_loader.py` — Factory
- `@lru_cache(maxsize=1)` ensures the LLM is initialized exactly once per process
- Raises `ValueError` for unknown `LLM_MODE` values
- Both clients initialized with: `temperature`, `max_tokens`, `timeout`, `verbose` from `cfg`

#### `prompt_templates.py` — `RAG_PROMPT`
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

### 🔒 Safety & Filtering (`src/app/safety/`)

#### `safety_filter.py` — Full Safety Pipeline

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

**80+ Health Keywords** span: greetings (`hi`, `hello`, `hey`), symptoms (`fever`, `cough`, `pain`, `rash`, `nausea`, etc.), body systems (`heart`, `skin`, `throat`, `blood`), conditions (`diabetes`, `asthma`, `cancer`, `stroke`), procedures (`mri`, `x-ray`, `blood test`), and medications (`tablet`, `pill`, `prescription`).

**Semantic Gate** — `is_health_related_semantic(query, vector_store)`:
- Calls `get_top_similarity_score()` with `DOMAIN_SIMILARITY_K` docs (default: 1)
- Returns `(True, score)` if `score >= DOMAIN_SIMILARITY_THRESHOLD` (default: 0.40)
- Falls back to keyword check if FAISS raises an exception

**Output Sanitization** — `sanitize_response(text)`:
- 5 regex replacements using `re.subn()`:

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
- **Mandatory disclaimer** always appended:
  > `⚕️ This information is for general educational purposes only and does not constitute medical advice, diagnosis, or treatment. Please consult a qualified healthcare professional for personalised medical guidance.`

---

### 🛠️ Core Services (`src/app/core/`)

#### `logging.py` — Structured Colored Logging
- `StructuredFormatter`: Produces millisecond-precision, pipe-delimited, ANSI-colored output:
  ```
  2026-04-08 10:32:45.123 | ℹ️  INFO     | app.rag.vector_store      | FAISS index loaded
  ```
- Per-level colors: DEBUG=Blue, INFO=Green, WARNING=Yellow, ERROR=Red, CRITICAL=Bright Red
- Per-level emojis: 🔧 DEBUG, ℹ️ INFO, ⚠️ WARNING, ❌ ERROR, 🚨 CRITICAL
- `ContextFilter`: Injects `job_id` and `document_id` into records (for job-level tracing)
- Helper functions: `get_logger()`, `set_context()`, `log_section()`, `log_step()`, `log_success()`, `log_error()`, `log_warning()`

#### `metrics.py` — Thread-Safe JSON Metrics
- Module-level `_METRICS_LOCK = threading.Lock()` protects all file writes
- `_ensure_metrics_file_exists()`: Creates parent directories and initializes default JSON structure on first run
- `build_metrics_entry()`: Computes all metric values from raw timing floats (converts `perf_counter` floats to ms)
- `persist_metrics(entry)`: Acquires lock → reads file → appends entry → updates rolling summary → writes atomically
- `reset_metrics()`: Acquires lock → overwrites file with default empty payload
- **Token approximation**: `len(text.split())` — word count used as proxy for token count (no tokenizer import)

#### `error_handling.py` — Exception Utilities
- Wraps exceptions with user-friendly messages for Streamlit display
- Used by `main.py` to render `st.error()` messages with the error template from `ui_text.py`

---

### ⚙️ Configuration (`src/app/config/`)

#### `settings.py` — `AppConfig` Dataclass
- All fields are resolved at import time from `os.getenv()` with typed defaults
- `_REPO_ROOT = Path(__file__).resolve().parents[3]` — resolves paths relative to repo root regardless of working directory
- Data paths, vector store path, and metrics path all use `str(_REPO_ROOT / ...)` for absolute paths

#### `ui_text.py` — `UIText` Dataclass Hierarchy
- `PageText` (frozen): All page-level strings — titles, spinners, placeholders, error templates, footer, dashboard labels, metric definitions
- `SidebarText` (frozen): Sidebar branding, navigation labels, Smart Symptom Guide options, about template
- `ChatText` (frozen): Welcome message template with `{smart_symptom_guide}` placeholder
- `UIText`: Composes all three via `field(default_factory=...)`; singleton `cfg.ui_text` accessed everywhere

---

## 🚀 RAG Pipeline Deep Dive

### Knowledge Base Structure

**`symptoms_conditions.json`** — Each record contains:
- `symptom` (str), `body_system` (str), `description` (str)
- `related_conditions` (list[str]), `source` (str)

**`conditions_info.json`** — Each record contains:
- `name`, `common_name`, `description`, `common_causes` (list), `typical_symptoms` (list)
- `general_care_tips` (list), `when_to_seek_care`, `specialist_type`, `prevention` (list), `source`

**`preventive_tips.json`** — Each record contains:
- `category` (str), `tips` (list[str]), `source` (str)

### Chunking & Indexing Flow

```
load_all_documents()           →  List[Document]  (raw, full-text)
        │
        ▼
RecursiveCharacterTextSplitter  →  List[Document]  (chunked, with metadata preserved)
  chunk_size=400, overlap=60
  separators=["\n\n","\n"," ",""]
        │
        ▼
HuggingFaceEmbeddings           →  384-dim float32 vectors per chunk
  all-MiniLM-L6-v2 (local CPU)
        │
        ▼
FAISS.from_documents()          →  FAISS index (L2 metric, normalized = cosine)
        │
        ▼
FAISS.save_local(VECTOR_STORE_DIR)   # persists index.faiss + index.pkl
```

### Retrieval & Generation

```python
# On each user query:
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

chain = RetrievalQA.from_chain_type(
    llm=EndpointLLM(...),
    chain_type="stuff",          # All retrieved docs concatenated into one prompt
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": RAG_PROMPT},
)

result = chain.invoke({"query": user_input})
# result["result"]           → generated text
# result["source_documents"] → List[Document] (used for source attribution)
```

---

## 🛠️ Local Setup & Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| RAM | ~4 GB (embeddings + FAISS) |
| Disk | ~2 GB (model weights + index) |
| OS | macOS / Linux / Windows |
| GPU | Not required (CPU-only) |

### Step 1: Clone & Install

```bash
cd /path/to/Wipro_Intel_Hackathon_LLM_EP

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2: Configure Environment

Create a `.env` file in the project root:

#### AWS SageMaker (default — local dev)

```bash
LLM_MODE=sagemaker
AWS_PROFILE=my-sso
AWS_REGION=us-east-2
SAGEMAKER_ENDPOINT=intel-llama3-1-8b-cpu-tgi-endpoint

LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=512
LLM_TIMEOUT=120

EMBEDDING_MODEL=./assets/models/all-MiniLM-L6-v2

CHUNK_SIZE=400
CHUNK_OVERLAP=60
RETRIEVER_K=4

DOMAIN_FILTER_MODE=hybrid
DOMAIN_SIMILARITY_THRESHOLD=0.40
DOMAIN_SIMILARITY_K=1
DOMAIN_USE_KEYWORD_FALLBACK=true

SHOW_RETRIEVED_SOURCES=true
```

#### Llama HTTP Endpoint (server deployment)

```bash
LLM_MODE=llama
LLM_BASE_URL=http://your-llama-server.com:8000/v1
LLM_API_KEY=your-api-key-if-needed
LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
# all other settings same as above
```

### Step 3: Verify Data Files

```bash
ls -la data/seed/
# symptoms_conditions.json
# conditions_info.json
# preventive_tips.json
```

### Step 4: Run the App

```bash
# Recommended: root launcher (automatically sets PYTHONPATH)
python app_launcher.py

# Alternative: direct Streamlit
streamlit run src/app/main.py

# With inline env vars
LLM_MODE=llama LLM_BASE_URL=http://your-server:8000/v1 python app_launcher.py
```

**Startup sequence:**
1. 🔄 Connecting to configured LLM endpoint (~2-3s)
2. ⏳ Loading health knowledge — build or load FAISS index (~1-2 min first run; <1s cached)
3. 🔗 Assembling RAG pipeline (~<1s)
4. App opens at `http://localhost:8501`

---

## 🧪 Testing & Validation

### Sanity Check Scripts

```bash
# Verify all required libraries import correctly
python scripts/sanity/sanity_library_imports.py

# Test Llama HTTP endpoint connection
python scripts/sanity/sanity_llama_model.py

# Test all app component initializations (LLM, embeddings, FAISS, RAG chain)
python scripts/sanity/sanity_app_components.py

# Full end-to-end app startup test
python scripts/sanity/sanity_app.py
```

### Example Queries

| Type | Example |
|------|---------|
| Simple symptom | `"I have a persistent cough for 3 days"` |
| Condition lookup | `"What are common symptoms of diabetes?"` |
| Multi-symptom | `"I'm experiencing chest pain and shortness of breath"` |
| Preventive | `"How can I prevent flu?"` |
| Greeting (accepted) | `"Hi"` / `"Hello"` |
| Off-domain (rejected) | `"Tell me about the weather"` / `"Write me a Python script"` |

### Using the Smart Symptom Guide

1. Open sidebar → **🔍 Smart symptom guide** → expand *"Provide your symptom details"*
2. Select primary symptom from dynamic dropdown (populated from knowledge base)
3. Select duration, set severity slider (1–10), optionally add extra symptoms
4. Click **📋 Prepare Symptom Details** — composes: *"My primary symptom is Fever, for 1–3 days, with a severity of 7/10. What could this be and what should I do?"*
5. Query auto-submits on next chat render

---

## 📊 Performance Monitoring

### Metrics Schema

Stored in `data/runtime/metrics/performance_metrics.json`:

```json
{
  "requests": [
    {
      "timestamp_utc": "2026-04-08T10:32:45Z",
      "query_preview": "I have a persistent cough and fever...",
      "ttft_ms": 234.56,
      "total_latency_ms": 2456.78,
      "end_to_end_latency_ms": 2456.78,
      "tokens_per_sec": 12.34,
      "retrieval_latency_ms": 123.45,
      "context_size_tokens": 450,
      "output_tokens": 234,
      "concurrent_users": 1,
      "error": null
    }
  ],
  "summary": {
    "total_requests": 42,
    "successful_requests": 41,
    "avg_total_latency_ms": 1234.56,
    "max_concurrent_users_observed": 3,
    "last_updated_utc": "2026-04-08T10:32:45Z"
  }
}
```

### Metrics Field Reference

| Field | Unit | Description |
|-------|------|-------------|
| `timestamp_utc` | ISO-8601 | UTC request completion time (displayed as IST in dashboard) |
| `query_preview` | String | First 120 chars of user query |
| `ttft_ms` | ms | Time To First Token (approximated as generation end for non-streaming) |
| `total_latency_ms` | ms | Wall-clock time from user input to response ready |
| `end_to_end_latency_ms` | ms | Alias for `total_latency_ms` |
| `tokens_per_sec` | float | Output tokens ÷ generation duration |
| `retrieval_latency_ms` | ms | Duration of FAISS similarity search only |
| `context_size_tokens` | count | Approx. tokens in all retrieved source chunks (word-split) |
| `output_tokens` | count | Approx. tokens in generated response (word-split) |
| `concurrent_users` | count | Active in-flight requests at submission time |
| `error` | string\|null | Exception message if request failed; null otherwise |

> **Token approximation note**: Token counts use `len(text.split())` (whitespace splitting) to avoid loading a heavyweight tokenizer per request. Accuracy is ~85-90% vs. sub-word tokenization.

### Metrics Dashboard Panels

| Panel | Metrics Shown |
|-------|---------------|
| **Health Status** | 🟢/🟡/🔴 indicator based on success rate threshold |
| **Summary** | Total requests, successful, avg total latency (sec), max concurrent users |
| **Latency Averages** | End-to-end, retrieval, total, TTFT — all in seconds |
| **Throughput & Context** | Tokens/sec, context size (tokens), output tokens, avg concurrent users |
| **Recent Requests Table** | Last 50 requests; IST timestamps; styled HTML table with SR No. |
| **Controls** | CSV download (`performance_metrics_requests.csv`) + Reset (with confirmation) |

---

## 🔧 Configuration Reference

All settings are in `src/app/config/settings.py` as the `AppConfig` dataclass. Override via environment variables or a `.env` file:

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `LLM_MODE` | `sagemaker` | str | LLM backend: `sagemaker` or `llama` |
| `LLM_TEMPERATURE` | `0.2` | float | Response creativity (0.0–1.0) |
| `LLM_MAX_TOKENS` | `512` | int | Max response tokens |
| `LLM_TIMEOUT` | `120` | int | Request timeout (seconds) |
| `LLM_VERBOSE` | `false` | bool | Log raw request/response payloads |
| `LLM_MODEL` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | str | Model ID (Llama mode) |
| `LLM_BASE_URL` | *(internal server)* | str | Base URL for Llama HTTP endpoint |
| `LLM_API_KEY` | `""` | str | Bearer token for Llama endpoint (optional) |
| `AWS_PROFILE` | `my-sso` | str | Named boto3 SSO profile |
| `AWS_REGION` | `us-east-2` | str | AWS region for SageMaker |
| `SAGEMAKER_ENDPOINT` | `intel-llama3-1-8b-cpu-tgi-endpoint` | str | SageMaker endpoint name |
| `EMBEDDING_MODEL` | `./assets/models/all-MiniLM-L6-v2` | str | Local path or HuggingFace model ID |
| `CHUNK_SIZE` | `400` | int | Document chunk size (tokens) |
| `CHUNK_OVERLAP` | `60` | int | Overlap between chunks |
| `RETRIEVER_K` | `4` | int | Documents retrieved per query |
| `VECTOR_STORE_DIR` | `./data/runtime/vector_store/faiss_index` | str | FAISS index directory |
| `METRICS_FILE_PATH` | `./data/runtime/metrics/performance_metrics.json` | str | Metrics JSON path |
| `SYMPTOMS_CONDITIONS_PATH` | `./data/seed/symptoms_conditions.json` | str | Symptoms knowledge base |
| `CONDITIONS_INFO_PATH` | `./data/seed/conditions_info.json` | str | Conditions knowledge base |
| `PREVENTIVE_TIPS_PATH` | `./data/seed/preventive_tips.json` | str | Preventive tips knowledge base |
| `DOMAIN_FILTER_MODE` | `hybrid` | str | Domain gate: `keyword`, `semantic`, or `hybrid` |
| `DOMAIN_SIMILARITY_THRESHOLD` | `0.40` | float | Semantic similarity cutoff (0.0–1.0) |
| `DOMAIN_SIMILARITY_K` | `1` | int | Docs retrieved for similarity scoring |
| `DOMAIN_USE_KEYWORD_FALLBACK` | `true` | bool | Keyword fallback in semantic/hybrid modes |
| `SHOW_RETRIEVED_SOURCES` | `true` | bool | Show source documents in chat UI |

---

## 📁 Project Structure

```
Wipro_Intel_Hackathon_LLM_EP/
├── app_launcher.py              # Root entry point — sets PYTHONPATH, runs Streamlit
├── requirements.txt             # Pinned Python dependencies
├── pyproject.toml               # Build system (setuptools); package root = src/
├── README.md                    # This file
├── SUMMARY.md                   # High-level project overview
│
├── src/
│   ├── __init__.py
│   └── app/
│       ├── __init__.py
│       ├── main.py              # Streamlit app entry point (UI orchestration)
│       │
│       ├── config/
│       │   ├── settings.py      # AppConfig dataclass + all env var resolution
│       │   ├── ui_text.py       # UIText / PageText / SidebarText / ChatText dataclasses
│       │   └── __init__.py
│       │
│       ├── core/
│       │   ├── logging.py       # StructuredFormatter, ContextFilter, helper functions
│       │   ├── metrics.py       # Thread-safe JSON metrics (build, persist, reset)
│       │   ├── error_handling.py # Exception wrapping utilities
│       │   └── __init__.py
│       │
│       ├── llm/
│       │   ├── model_loader.py  # get_llm() factory with @lru_cache
│       │   ├── endpoint_client.py # SageMakerEndpointClient + LlamaEndpointClient
│       │   ├── langchain_adapter.py # EndpointLLM (LangChain LLM subclass)
│       │   ├── prompt_templates.py  # RAG_PROMPT (8 grounding + 10 style rules)
│       │   └── __init__.py
│       │
│       ├── rag/
│       │   ├── data_loader.py   # JSON → LangChain Document (3 loaders + combined)
│       │   ├── embedder.py      # get_embeddings() — HuggingFaceEmbeddings cached
│       │   ├── vector_store.py  # build_or_load_vector_store() — FAISS + splitter
│       │   ├── retriever.py     # build_rag_chain(), retrieve_sources(), get_top_similarity_score()
│       │   └── __init__.py
│       │
│       ├── safety/
│       │   ├── safety_filter.py # Domain gate (keyword/semantic/hybrid) + output sanitizer
│       │   └── __init__.py
│       │
│       └── ui/
│           ├── chat_interface.py   # init_chat_state, render_chat_history, add_message
│           ├── metrics_dashboard.py # Full metrics dashboard with health status + IST table
│           ├── sidebar.py          # Navigation, logos, Smart Symptom Guide, about, clear
│           ├── styles.py           # StreamlitStyles (CSS injection + definition renderer)
│           ├── source_utils.py     # format_source_label, truncate_source_content
│           └── __init__.py
│
├── data/
│   ├── seed/                    # Healthcare knowledge base (committed)
│   │   ├── symptoms_conditions.json
│   │   ├── conditions_info.json
│   │   └── preventive_tips.json
│   │
│   └── runtime/                 # Auto-generated at runtime (not committed to git)
│       ├── vector_store/
│       │   └── faiss_index/
│       │       ├── index.faiss  # FAISS index binary
│       │       └── index.pkl    # FAISS docstore / metadata
│       │
│       └── metrics/
│           └── performance_metrics.json
│
├── assets/
│   ├── images/
│   │   ├── wipro.png            # Wipro logo (sidebar branding)
│   │   └── intel.png            # Intel logo (sidebar branding)
│   │
│   └── models/
│       └── all-MiniLM-L6-v2/   # Bundled local embedding model
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           ├── modules.json
│           ├── sentence_bert_config.json
│           ├── 1_Pooling/config.json
│           └── 2_Normalize/
│
├── scripts/
│   └── sanity/                  # Startup verification scripts
│       ├── __init__.py
│       ├── sanity_app.py
│       ├── sanity_app_components.py
│       ├── sanity_library_imports.py
│       └── sanity_llama_model.py
│
└── docs/
    └── Question_Coverage_Matrix.md
```

---

## 🔐 Safety & Ethical Design

### Three-Layer Safety Architecture

```
Layer 1 — Input Domain Gate (before RAG)
  └── Prevents wasted LLM calls on irrelevant queries
  └── Hybrid mode balances coverage vs. precision

Layer 2 — Prompt-Level Grounding (inside LLM)
  └── 8 grounding rules prevent hallucination
  └── Forbids diagnoses and prescriptions in output
  └── Instructs cautious language throughout

Layer 3 — Output Sanitization (after LLM)
  └── 10 regex patterns catch any diagnostic slip-through
  └── Mandatory disclaimer on every response
```

### Responsible AI Principles Applied

- **No Diagnosis**: The system explicitly cannot and will not diagnose conditions
- **No Prescription**: Medication recommendations are blocked at prompt and output layers
- **Grounded Only**: Responses cite only information present in the retrieved knowledge base
- **Transparent Sources**: Every response optionally shows which knowledge documents were used
- **Urgent Escalation**: Severe symptom descriptions trigger natural-language urgency escalation — without clinical labels
- **Educational Framing**: All information positioned as general health education, not personalized advice

---

## 🧑‍💻 Development & Customization

### Adding New Knowledge Sources

1. Create a new JSON file in `data/seed/` following existing schemas
2. Add a new loader function in `src/app/rag/data_loader.py` following the `load_*_documents()` pattern
3. Call it inside `load_all_documents()`
4. Delete `data/runtime/vector_store/faiss_index/` to force a FAISS index rebuild
5. Restart the app — the new data will be embedded and indexed automatically

### Switching the LLM Backend

```bash
# SageMaker (default)
export LLM_MODE=sagemaker
export AWS_PROFILE=my-sso
export SAGEMAKER_ENDPOINT=intel-llama3-1-8b-cpu-tgi-endpoint
python app_launcher.py

# Llama HTTP
export LLM_MODE=llama
export LLM_BASE_URL=http://my-server:8000/v1
export LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
python app_launcher.py
```

### Tuning the Domain Gate

```bash
# Stricter (fewer false accepts)
export DOMAIN_SIMILARITY_THRESHOLD=0.55
export DOMAIN_FILTER_MODE=semantic

# More permissive (fewer false rejects)
export DOMAIN_SIMILARITY_THRESHOLD=0.30
export DOMAIN_FILTER_MODE=hybrid
```

### Custom Prompt Engineering

Edit `src/app/llm/prompt_templates.py`. The `RAG_PROMPT` template uses:
- `{context}` — the concatenated retrieved document chunks
- `{question}` — the user's query

The `chain_type="stuff"` strategy concatenates all retrieved docs into a single prompt context window. For very large knowledge bases, consider `chain_type="map_reduce"` or `chain_type="refine"`.

### Extending the Metrics Dashboard

Add new metrics fields in `src/app/core/metrics.py` → `build_metrics_entry()`, then render them in `src/app/ui/metrics_dashboard.py`.

---

## 🚨 Troubleshooting

### FAISS Index Issues

```bash
# Force full rebuild
rm -rf data/runtime/vector_store/faiss_index/
python app_launcher.py
```

### LLM Endpoint Timeout

```bash
# Test Llama HTTP endpoint directly
curl -X POST http://your-server:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "hello"}]}'

# Test SageMaker endpoint
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name your-endpoint \
  --region us-east-2 \
  --content-type application/json \
  --body '{"inputs": "test"}' \
  /tmp/output.txt
```

### AWS SSO Authentication

```bash
aws sso login --profile my-sso
# Then re-run the app
python app_launcher.py
```

### Out of Memory During Index Build

```bash
export CHUNK_SIZE=300
export RETRIEVER_K=2
python app_launcher.py
```

### Slow Search / High Latency

```bash
# Check index size on disk
du -sh data/runtime/vector_store/faiss_index/

# Verify embedding model loads
python -c "from src.app.rag.embedder import get_embeddings; e = get_embeddings(); print('OK')"
```

### Streamlit "Duplicate Element Key" Errors

These are handled automatically — `main.py` generates unique keys per request using nanosecond timestamps: `f"msg-{int(request_started * 1e9)}"`.

---

## 📚 Dependencies

### Key Libraries (Pinned)

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.55.0 | Web UI framework |
| `langchain` | 0.3.0 | RAG orchestration |
| `langchain-community` | 0.3.0 | FAISS integration |
| `langchain-huggingface` | 0.1.0 | HuggingFace embeddings |
| `langchain-core` | 0.3.0 | Base LLM, PromptTemplate, Runnable |
| `langchain-text-splitters` | 0.3.0 | RecursiveCharacterTextSplitter |
| `faiss-cpu` | 1.13.2 | Vector similarity search |
| `sentence-transformers` | 5.3.0 | all-MiniLM-L6-v2 embeddings |
| `transformers` | 5.3.0 | HuggingFace model loading |
| `torch` | 2.10.0 | PyTorch backend (CPU) |
| `boto3` / `botocore` | latest | AWS SageMaker client |
| `pydantic` | 2.12.5 | Config validation (AppConfig, EndpointLLM) |
| `pydantic-settings` | 2.13.1 | Settings management |
| `python-dotenv` | 1.2.2 | `.env` file loading |
| `pandas` | 2.3.3 | Metrics DataFrame processing |
| `requests` | 2.32.5 | Llama HTTP endpoint client |
| `numpy` | 1.26.4 | Numerical operations (FAISS) |
| `safetensors` | 0.7.0 | Model weight loading |

### External References

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Sentence-Transformers — all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Meta Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [AWS SageMaker Runtime API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html)

---

## ❓ FAQ

**Q: Can this system diagnose diseases?**  
A: No. It provides general health information only. All responses include a mandatory disclaimer and avoid diagnostic language.

**Q: What is the difference between `LLM_MODE=sagemaker` and `llama`?**  
A: SageMaker uses AWS for LLM inference via boto3 + a named SSO profile (default for local dev). Llama mode POSTs to an OpenAI-compatible HTTP endpoint (for server deployments). Switching only requires changing `LLM_MODE` and relevant endpoint settings.

**Q: Why does the first startup take 1-2 minutes?**  
A: On first run, there is no FAISS index on disk. The app loads all knowledge base documents, splits them into ~400-token chunks, runs all chunks through the embedding model, and builds the FAISS index. The result is saved to `data/runtime/vector_store/faiss_index/`. All subsequent starts load the pre-built index in under 1 second.

**Q: How is TTFT measured for non-streaming responses?**  
A: Since `RetrievalQA.invoke()` is non-streaming, TTFT is approximated as the generation completion timestamp (`generation_ended`). True TTFT would require streaming token callbacks.

**Q: Can I use a different embedding model?**  
A: Yes — set `EMBEDDING_MODEL` to any HuggingFace model ID or local path. Delete the FAISS index directory afterwards to force a rebuild with the new embeddings.

**Q: How do I update the knowledge base?**  
A: Edit or add JSON files in `data/seed/`, register new loaders in `data_loader.py`, then delete the FAISS index directory to trigger a rebuild on next startup.

**Q: Is the system thread-safe?**  
A: Yes. Metrics JSON writes are protected by `threading.Lock`. Streamlit isolates session state per browser session. The LLM and vector store are initialized once via `@st.cache_resource` and `@lru_cache`, which are safe for concurrent reads.

**Q: What happens when a greeting like "Hi" is sent?**  
A: Greetings (`hi`, `hello`, `hey`, `greetings`, `greeting`) are in the keyword allowlist, so they pass the domain gate. The system prompt instructs the LLM to respond warmly and invite a health question — not reject the message.

**Q: What does the Health Status indicator in the Metrics Dashboard mean?**  
A: It reflects the current success rate: 🟢 Healthy = ≥80% requests without errors; 🟡 Degraded = 60–79%; 🔴 Unhealthy = <60%. It resets when metrics are cleared.

**Q: Why are token counts approximate?**  
A: Token counts use whitespace word-splitting (`len(text.split())`) rather than a sub-word tokenizer, for performance. Accuracy is ~85-90% vs. actual tokenization.

---

## 🤝 Support & Contributing

1. Check the [Troubleshooting](#-troubleshooting) section first
2. Review terminal logs (color-coded, millisecond-precision)
3. Check the Metrics Dashboard for request-level insights
4. Run sanity scripts in `scripts/sanity/` to isolate component failures
5. Contact the development team

---

## 📝 Attribution

**Project:** Healthcare Symptom Information Assistant  
**Event:** Wipro × Intel Hackathon — April 2026  
**Tech:** Streamlit · LangChain · FAISS · all-MiniLM-L6-v2 · Meta Llama 3.1 8B · Intel® CPU  
**Created by:** Team [Your Name/Team Name]

---

*Built with ❤️ for safe, trustworthy healthcare information delivery — optimized for Intel® CPU inference.*
