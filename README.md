# 🩺 Healthcare Symptom Information Assistant

> Built for the **Wipro × Intel Hackathon** · March 2026

A conversational health-information assistant that accepts natural-language symptom descriptions, retrieves relevant knowledge from a structured WHO/CDC-curated knowledge base using RAG, and returns safe, non-diagnostic health guidance via a configurable remote LLM endpoint.

---

## ⚠️ Medical Disclaimer

- This tool provides **general health information only**.  
- It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.  
- Always consult a qualified healthcare provider for any medical concerns.  
- For severe, sudden, or worsening symptoms — **seek emergency care immediately**.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Project Structure](#-project-structure)
4. [Module Responsibilities](#-module-responsibilities)
5. [End-to-End Workflow](#-end-to-end-workflow)
6. [Tech Stack](#-tech-stack)
7. [LLM Backend Configuration](#-llm-backend-configuration)
8. [All Configuration Parameters](#-all-configuration-parameters)
9. [UI Text Configuration](#-ui-text-configuration)
10. [Healthcare Data](#-healthcare-data)
11. [RAG Pipeline — How It Works](#-rag-pipeline--how-it-works)
12. [Prompt Engineering and Safety Policy](#-prompt-engineering-and-safety-policy)
13. [Safety Filter — Input and Output Guards](#-safety-filter--input-and-output-guards)
14. [Logging and Observability](#-logging-and-observability)
15. [Prerequisites](#-prerequisites)
16. [Installation](#-installation)
17. [Running the App](#-running-the-app)
18. [Sanity Checks](#-sanity-checks)
19. [UI Features](#-ui-features)
20. [Example Prompts](#-example-prompts)
21. [Troubleshooting](#-troubleshooting)
22. [File Reference](#-file-reference)

---

## 🎯 Project Overview

The assistant:

- Accepts freeform or structured symptom queries from the user
- **Validates** that the query is health-related before hitting any LLM or retrieval pipeline (input domain gate)
- Embeds the query using a locally cached `sentence-transformers` model (`all-MiniLM-L6-v2`)
- Retrieves the top-4 most relevant chunks from a local FAISS index built over WHO/CDC-curated JSON data
- Sends a safety-constrained prompt (retrieved context + user question) to a configurable remote LLM endpoint (`llama` HTTP mode or `sagemaker` AWS mode)
- Post-processes the response through a regex-based safety filter that softens diagnostic language
- Displays the answer in a Streamlit chat UI with optional expandable source citations

The app **never diagnoses**, **never prescribes**, and **always recommends consulting a healthcare professional**.

---

## 🏗️ Architecture

### Layered Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                               │
│                                                                         │
│   app.py — Main Streamlit entry point and request orchestrator          │
│                                                                         │
│   ┌──────────────────────────┐   ┌──────────────────────────────────┐   │
│   │      ui/sidebar.py       │   │     ui/chat_interface.py         │   │
│   │  - Disclaimer banner     │   │  - Welcome message               │   │
│   │  - Dynamic symptom list  │   │  - Multi-turn session state      │   │
│   │    (from JSON data)      │   │  - User 👤 / Assistant 🩺        │   │
│   │  - Duration / severity   │   │  - Source citation expander      │   │
│   │  - Structured query      │   │    (SHOW_RETRIEVED_SOURCES)      │   │
│   │  - Clear chat            │   │                                  │   │
│   │  - About section         │   └──────────────────────────────────┘   │
│   └──────────────────────────┘                                          │
│                                                                         │
│   ui_text_config.py — All UI copy decoupled from logic                  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ User Query
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SAFETY LAYER — INPUT GATE                            │
│                                                                         │
│   utils/safety_filter.py → is_health_related(query)                     │
│     • Checks query against 60+ health keywords                          │
│     • Blocks non-health / off-domain prompts before any LLM call        │
│     • Returns get_off_domain_message() with usage examples              │
│     • Logs matched keywords (debug) or warning (off-domain)             │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ Health-domain query only
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG PIPELINE LAYER                             │
│                                                                         │
│  Step 1 — Embed query         rag/embedder.py                           │
│           HuggingFaceEmbeddings (all-MiniLM-L6-v2, CPU, normalised)     │
│                                                                         │
│  Step 2 — Vector search       rag/vector_store.py                       │
│           FAISS local index ← built from 3 JSON healthcare files        │
│           top-k=4 chunks via cosine similarity                          │
│                                                                         │
│  Step 3 — Assemble RAG chain  rag/retriever.py                          │
│           LangChain RetrievalQA ("stuff" chain type)                    │
│           Safety-constrained RAG_PROMPT (context + question)            │
│                                                                         │
│  Step 4 — Call LLM backend    llm/model_loader.py                       │
│           EndpointLLM (LangChain adapter over endpoint client)          │
│           ├─ llama mode  → LlamaEndpointClient                          │
│           │                POST /v1/chat/completions                    │
│           └─ sagemaker   → SageMakerEndpointClient                      │
│                             boto3 invoke_endpoint                       │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ Raw LLM response
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   SAFETY LAYER — OUTPUT SANITIZER                       │
│                                                                         │
│   utils/safety_filter.py → sanitize_response(text)                      │
│     • 10 regex replacements: softens hard diagnostic phrasing           │
│     • Appends mandatory consultation disclaimer to every response       │
│     • Each replacement individually logged as a warning                 │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ Safe response + retrieved sources
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER — OUTPUT                           │
│                                                                         │
│   st.markdown(safe_response)                                            │
│   Optional: st.expander() → source snippets with metadata labels        │
│   add_message() → persisted to session_state for multi-turn replay      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Supporting Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  CONFIGURATION LAYER                                                 │
│  config.py — AppConfig dataclass (singleton: cfg)                    │
│    • LLM mode switch (llama / sagemaker), endpoint URLs              │
│    • Generation params: temperature, max_tokens, timeout, verbose    │
│    • RAG: chunk_size=400, chunk_overlap=60, retriever_k=4            │
│    • Data paths, vector store dir                                    │
│    • show_retrieved_sources toggle (SHOW_RETRIEVED_SOURCES)          │
│    • ui_text: UIText injected from ui_text_config.py                 │
│    • All fields env-var overridable; active overrides logged         │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  OBSERVABILITY LAYER                                                 │
│  logger_config.py — StructuredFormatter                              │
│    • Colored, timestamped, emoji-prefixed console output             │
│    • Helpers: get_logger, log_section, log_step, log_success,        │
│      log_error, log_warning                                          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                          │
│  data/ — Three WHO/CDC-curated JSON files                            │
│  vector_store/faiss_index/ — Persisted FAISS index (auto-generated)  │
│  models/all-MiniLM-L6-v2/ — Local embedding model (no download)      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
wipro-intel-genai-hackathon/
│
├── app.py                          ← Main Streamlit entry point and orchestrator
├── config.py                       ← Centralised AppConfig dataclass (all env-overridable)
├── ui_text_config.py               ← All user-facing UI copy (PageText, SidebarText, ChatText)
├── logger_config.py                ← Structured, colored logging utilities
├── requirements.txt                ← Pinned Python 3.11 dependencies
│
├── data/
│   ├── symptoms_conditions.json    ← 30 symptoms mapped to conditions (WHO/CDC)
│   ├── conditions_info.json        ← 26 conditions with full care detail (WHO/CDC)
│   └── preventive_tips.json        ← 10 prevention categories (WHO)
│
├── llm/
│   ├── __init__.py
│   ├── endpoint_client.py          ← SageMakerEndpointClient + LlamaEndpointClient
│   │                                  Shared: _coerce_prompt(), _extract_content()
│   ├── langchain_adapter.py        ← EndpointLLM: wraps clients as LangChain LLM
│   ├── model_loader.py             ← get_llm() factory with @lru_cache(maxsize=1)
│   └── prompt_templates.py         ← RAG_PROMPT: safety-constrained PromptTemplate
│
├── rag/
│   ├── __init__.py
│   ├── data_loader.py              ← JSON → LangChain Documents (3 loaders + combined)
│   ├── embedder.py                 ← get_embeddings() with @lru_cache (CPU, normalised)
│   ├── vector_store.py             ← FAISS build/load/persist + per-chunk debug logging
│   └── retriever.py                ← RetrievalQA chain + retrieve_sources() for UI display
│
├── ui/
│   ├── __init__.py
│   ├── sidebar.py                  ← Dynamic symptom list, query builder, disclaimer, about
│   └── chat_interface.py           ← Session state, history rendering, add_message()
│
├── utils/
│   ├── __init__.py
│   └── safety_filter.py            ← Input domain gate + output phrase sanitizer + disclaimer
│
├── sanity_scripts/
│   ├── sanity_library_imports.py   ← Validates 13 required library imports
│   ├── sanity_llama_model.py       ← End-to-end LLM endpoint connectivity + inference test
│   ├── sanity_app.py               ← Minimal Streamlit interactive endpoint sanity UI
│   └── sanity_app_components.py    ← Full module import + data load validation (all layers)
│
├── models/
│   └── all-MiniLM-L6-v2/          ← Local embedding model (fast load, no internet required)
│
└── vector_store/
    └── faiss_index/                ← Auto-generated FAISS index (created on first run)
        ├── index.faiss
        └── index.pkl
```

---

## 📦 Module Responsibilities

### `app.py` — Orchestrator
The single Streamlit entry point. Responsibilities:
- Sets page config (title, icon, spinners, captions) from `cfg.ui_text.page`
- Initialises multi-turn chat session state via `init_chat_state()`
- Renders sidebar controls via `render_sidebar()`
- Caches three heavy resources using `@st.cache_resource`:
  - `load_llm()` → `get_llm()`
  - `load_vector_store()` → `build_or_load_vector_store()`
  - `load_rag_chain(_llm, _vector_store)` → `build_rag_chain()`
- On each user query:
  1. Adds user message to session state
  2. Calls `is_health_related(query)` — blocks off-domain prompts
  3. Invokes `rag_chain.invoke({"query": user_input})`
  4. Calls `retrieve_sources()` separately for transparent source display
  5. Calls `sanitize_response()` on raw result
  6. Renders response; conditionally shows source expander (`cfg.show_retrieved_sources`)
  7. Persists to session state via `add_message()`

---

### `config.py` — Configuration
- Single `AppConfig` dataclass, exported as `cfg` singleton
- Every field is overridable via environment variable
- Imports `UIText` from `ui_text_config.py` and attaches as `cfg.ui_text`
- On startup: logs all active environment overrides using `log_section`

---

### `ui_text_config.py` — UI Text
Contains three frozen dataclasses:
- `PageText` — page title, icon, app title, spinner messages, input placeholder, source expander label, footer HTML, error message template
- `SidebarText` — title, tagline, disclaimer, symptom defaults, duration options, build query button label, about template
- `ChatText` — welcome message

All assembled into `UIText` and injected as `cfg.ui_text`.  
**To change any wording: edit `ui_text_config.py` only — no changes needed in `app.py`, `sidebar.py`, or `chat_interface.py`.**

---

### `llm/endpoint_client.py` — Endpoint Clients
Two client classes with a shared `.invoke(prompt) → EndpointResponse` interface:

**`SageMakerEndpointClient`**
- Creates a `boto3.Session` with named SSO profile and region
- Sends TGI-style payload: `{"inputs": "<prompt>", "parameters": {"temperature": ..., "max_new_tokens": ..., "return_full_text": false}}`
- Handles both TGI list/dict and OpenAI-compatible choices response shapes

**`LlamaEndpointClient`**
- Sends OpenAI chat completions payload: `{"model": ..., "messages": [{"role": "user", "content": ...}], ...}`
- Adds `Authorization: Bearer <token>` header only if `api_key` is non-empty and not `"dummy"`
- Handles `requests.exceptions.HTTPError`, `ConnectionError`, `Timeout` distinctly

**Shared module-level helpers:**
- `_coerce_prompt(prompt)` — coerces LangChain prompt objects to plain strings
- `_extract_content(data, logger, client_name)` — handles OpenAI `choices[0].message.content`, flat `generated_text`, and TGI `list[0].generated_text` response shapes

---

### `llm/langchain_adapter.py` — LangChain Adapter
`EndpointLLM` extends LangChain's `LLM` base class:
- `_call(prompt, stop, run_manager)` — invokes client, applies stop tokens
- `_llm_type` — returns `"endpoint_{llm_mode}"`
- `_identifying_params` — exposes `llm_mode` and `llm_name` for caching/logging
- `model_config = ConfigDict(arbitrary_types_allowed=True)` enables Pydantic v2 compatibility

---

### `llm/model_loader.py` — LLM Factory
`get_llm()` decorated with `@lru_cache(maxsize=1)`:
- Reads `cfg.llm_mode` (`"llama"` or `"sagemaker"`)
- Constructs the appropriate client and wraps it in `EndpointLLM`
- Raises `ValueError` with a clear message for unknown modes

---

### `llm/prompt_templates.py` — Prompt Template
`RAG_PROMPT` is a LangChain `PromptTemplate` with:
- Input variables: `{context}` (retrieved chunks) and `{question}` (user input)
- Safety rules, style rules, and a structured output format enforced at inference time (see [Prompt Engineering](#-prompt-engineering-and-safety-policy))

---

### `rag/data_loader.py` — Data Loader
- `load_symptom_documents()` — converts `symptoms_conditions.json` → `Document` objects with metadata `{source_file, symptom, body_system}`
- `load_condition_documents()` — converts `conditions_info.json` → `Document` objects with metadata `{source_file, condition, specialist}`
- `load_preventive_documents()` — converts `preventive_tips.json` → `Document` objects with metadata `{source_file, category}`
- `load_all_documents()` — aggregates all three with error handling and success logging

---

### `rag/embedder.py` — Embedding Model
`get_embeddings()` with `@lru_cache(maxsize=1)`:
- Returns `HuggingFaceEmbeddings` with `model_name=cfg.embedding_model`, `device="cpu"`, `normalize_embeddings=True`
- Default model path: `models/all-MiniLM-L6-v2` (local directory — no internet required)

---

### `rag/vector_store.py` — FAISS Index Manager
`build_or_load_vector_store()`:
- If `vector_store/faiss_index/index.faiss` exists → `FAISS.load_local()` (fast)
- Otherwise → `load_all_documents()` → `_split_documents()` → `FAISS.from_documents()` → `save_local()`

`_split_documents(docs)`:
- Uses `RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", " ", ""])`
- Logs each chunk's source, length, and a 200-char content preview at DEBUG level

---

### `rag/retriever.py` — RAG Chain + Source Retrieval
`build_rag_chain(llm, vector_store)`:
- Builds `RetrievalQA.from_chain_type()` with:
  - `chain_type="stuff"` (concatenates all chunks into one context block)
  - `retriever=FAISS.as_retriever(search_type="similarity", k=4)`
  - `return_source_documents=True`
  - `chain_type_kwargs={"prompt": RAG_PROMPT}`

`retrieve_sources(vector_store, query)`:
- Separate top-k retrieval used only for UI source display
- Content truncated to 300 characters per chunk

---

### `ui/sidebar.py` — Sidebar Component
`_load_symptom_options()`:
- Reads `symptoms_conditions.json`, extracts and title-cases symptom names
- Returns `["Select..."] + sorted(symptoms)` alphabetically
- Falls back to static defaults from `cfg.ui_text.sidebar.symptom_defaults` on any failure

`render_sidebar()`:
- Injects CSS: fixes sidebar width at 400px, styles buttons with blue theme
- Renders: branding, disclaimer, structured query builder (symptom → duration → severity → extra → build button), clear chat button, about expander
- "Build Query" validates that a primary symptom is selected before composing the query string
- "Clear Chat" resets session to the welcome message

---

### `ui/chat_interface.py` — Chat State and History
- `init_chat_state()` — seeds `st.session_state["messages"]` with welcome message from `cfg.ui_text.chat.welcome_message`
- `render_chat_history()` — replays full message history; conditionally shows source expanders per message when `cfg.show_retrieved_sources=True`
- `add_message(role, content, sources)` — appends to session state

---

### `utils/safety_filter.py` — Safety Guards
Three public functions:
- `is_health_related(query)` — input domain gate (see [Safety Filter](#-safety-filter--input-and-output-guards))
- `get_off_domain_message()` — returns the standard off-domain user-facing message with usage examples
- `sanitize_response(text)` — output post-processor (regex replacements + disclaimer append)

---

## 🔄 End-to-End Workflow

### Phase 1: Startup and Resource Initialisation

```
python -m streamlit run app.py
        │
        ├─ st.set_page_config()      reads cfg.ui_text.page (title, icon)
        ├─ init_chat_state()         seeds session with welcome message
        ├─ render_sidebar()          loads symptom list from JSON, renders controls
        │
        ├─ load_llm()  (@st.cache_resource — runs once per session)
        │   └─ get_llm() (@lru_cache)
        │       ├─ llm_mode="llama"
        │       │   └─ LlamaEndpointClient(base_url, model, api_key, temp, tokens, timeout)
        │       │      └─ EndpointLLM(client, llm_mode="llama", llm_name=cfg.llm_model)
        │       └─ llm_mode="sagemaker"
        │           └─ SageMakerEndpointClient(endpoint, profile, region, temp, tokens, timeout)
        │              └─ EndpointLLM(client, llm_mode="sagemaker", llm_name=cfg.sagemaker_endpoint)
        │
        ├─ load_vector_store()  (@st.cache_resource — runs once per session)
        │   └─ build_or_load_vector_store()
        │       ├─ index.faiss exists?
        │       │   └─ YES → FAISS.load_local("vector_store/faiss_index")  [fast]
        │       └─ NO  → load_all_documents()
        │                  → _split_documents()  [chunk_size=400, overlap=60]
        │                  → get_embeddings()    [all-MiniLM-L6-v2, CPU]
        │                  → FAISS.from_documents(chunks, embeddings)
        │                  → FAISS.save_local("vector_store/faiss_index")
        │
        └─ load_rag_chain(llm, vector_store)  (@st.cache_resource — runs once per session)
            └─ build_rag_chain()
               └─ RetrievalQA.from_chain_type(
                      llm=EndpointLLM,
                      chain_type="stuff",
                      retriever=FAISS.as_retriever(search_type="similarity", k=4),
                      return_source_documents=True,
                      chain_type_kwargs={"prompt": RAG_PROMPT}
                  )
```

### Phase 2: Query Processing

```
User submits query  (chat input or sidebar prefilled query)
        │
        ├─ add_message("user", query)        persist to session state
        ├─ st.chat_message("user", avatar="👤")  render immediately
        │
        ├─ is_health_related(query)
        │     │
        │     ├─ FALSE (no health keyword matched)
        │     │   └─ st.warning(get_off_domain_message())
        │     │   └─ add_message("assistant", off_domain_msg)
        │     │   └─ STOP  ← RAG pipeline is NOT called
        │     │
        │     └─ TRUE (at least one health keyword matched)
        │         │
        │         ├─ rag_chain.invoke({"query": user_input})
        │         │   └─ FAISS.as_retriever().invoke(query)
        │         │      └─ embed query (all-MiniLM-L6-v2)
        │         │      └─ cosine similarity search → top-4 chunks
        │         │   └─ "stuff" chain: join chunks → {context}
        │         │   └─ RAG_PROMPT.format(context=..., question=...)
        │         │   └─ EndpointLLM._call(assembled_prompt)
        │         │      └─ _coerce_prompt(prompt) → plain string
        │         │      └─ LlamaEndpointClient.invoke(prompt)
        │         │         └─ POST {LLM_BASE_URL}/chat/completions
        │         │         └─ _extract_content(response_json)
        │         │         └─ EndpointResponse(content=..., raw=...)
        │         │   └─ raw_response = result.get("result", "")
        │         │
        │         ├─ retrieve_sources(vector_store, query)
        │         │   └─ separate top-4 retrieval for UI source display
        │         │   └─ content truncated to 300 chars per chunk
        │         │
        │         ├─ sanitize_response(raw_response)
        │         │   └─ re.subn() for each of 10 _DIAGNOSIS_PATTERNS
        │         │   └─ append _CONSULT_REMINDER
        │         │   └─ log each replacement as warning
        │         │
        │         ├─ st.markdown(safe_response)
        │         ├─ if cfg.show_retrieved_sources:
        │         │     st.expander("📚 View retrieved health knowledge sources")
        │         │     └─ for each source: show label + content snippet + divider
        │         │
        │         └─ add_message("assistant", safe_response, sources)
```

### Phase 3: Sidebar Structured Query Builder

```
User opens "Use Structured Input" expander in sidebar
        │
        ├─ symptom      = st.selectbox()   ← dynamic from symptoms_conditions.json
        │                                     fallback: cfg.ui_text.sidebar.symptom_defaults
        ├─ duration     = st.selectbox()   ← static: cfg.ui_text.sidebar.duration_options
        ├─ severity     = st.slider(1–10, default=5)
        ├─ extra        = st.text_input()
        │
        └─ "📋 Prepare My Symptom Question"  button
                │
                ├─ validates: symptom != "Select..."  (warns if empty)
                ├─ composes query:
                │   "My primary symptom is {symptom}, including {extra},
                │    for {duration}, with a severity of {severity}/10.
                │    What could this be and what should I do?"
                └─ st.session_state["prefilled_query"] = composed_query
                        └─ picked up by app.py on next rerun
                        └─ auto-submitted as chat input
```

---

## 🛠️ Tech Stack

| Layer | Library / Tool | Version |
|---|---|---|
| UI | Streamlit | 1.55.0 |
| LLM (HTTP mode) | requests | 2.32.5 |
| LLM (SageMaker mode) | boto3 / botocore | latest |
| RAG orchestration | LangChain | 0.3.0 |
| RAG community | langchain-community | 0.3.0 |
| LangChain core | langchain-core | 0.3.0 |
| Embeddings integration | langchain-huggingface | 0.1.0 |
| Embedding model | sentence-transformers | 5.3.0 |
| Deep learning runtime | torch | 2.10.0 |
| Transformers | transformers | 5.3.0 |
| Vector store | faiss-cpu | 1.13.2 |
| Text splitting | langchain-text-splitters | 0.3.0 |
| Data validation | pydantic | 2.12.5 |
| Language | Python | 3.11 |

> All pinned versions are defined in `requirements.txt` — the source of truth for reproducible installs.

---

## ⚙️ LLM Backend Configuration

The app supports two LLM backends, selected by the `LLM_MODE` environment variable.

### Mode: `llama` (current default)

Calls a remote OpenAI-compatible HTTP endpoint via `LlamaEndpointClient`:

```
POST {LLM_BASE_URL}/chat/completions

{
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "messages": [{"role": "user", "content": "<assembled RAG prompt>"}],
  "temperature": 0.3,
  "max_tokens": 512
}
```

The `Authorization: Bearer <LLM_API_KEY>` header is added **only** if `LLM_API_KEY` is non-empty and not `"dummy"`.

Response parsing handles both:
- OpenAI shape: `choices[0].message.content` (string or list)
- TGI shape: `generated_text` (dict) or `[0].generated_text` (list)

### Mode: `sagemaker`

Calls an AWS SageMaker real-time endpoint via `SageMakerEndpointClient`:

```python
boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    .client("sagemaker-runtime")
    .invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="application/json",
        Body={
            "inputs": "<prompt>",
            "parameters": {
                "temperature": 0.3,
                "max_new_tokens": 512,
                "return_full_text": false
            }
        }
    )
```

### Setting your endpoint

**Option A — Environment variables (recommended):**

```bash
# Llama HTTP mode (default)
export LLM_MODE="llama"
export LLM_BASE_URL="http://<your-tgi-host>:<port>/v1"
export LLM_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export LLM_API_KEY=""
export LLM_TIMEOUT="120"
export LLM_TEMPERATURE="0.3"
export LLM_MAX_TOKENS="512"

# SageMaker mode
export LLM_MODE="sagemaker"
export AWS_PROFILE="my-sso"
export AWS_REGION="us-east-2"
export SAGEMAKER_ENDPOINT="llama-3-2-3b-tgi-cpu-endpoint"
```

**Option B — Edit `config.py` defaults directly:**

```python
llm_mode:     str = os.getenv("LLM_MODE",     "llama")
llm_base_url: str = os.getenv("LLM_BASE_URL", "http://<your-host>/v1")
llm_model:    str = os.getenv("LLM_MODEL",    "meta-llama/Meta-Llama-3.1-8B-Instruct")
```

---

## 📐 All Configuration Parameters

| Parameter | Env Var | Default | Description |
|---|---|---|---|
| `llm_mode` | `LLM_MODE` | `llama` | Backend selector: `"llama"` or `"sagemaker"` |
| `llm_base_url` | `LLM_BASE_URL` | `http://wiphackq0vcsii.cloudloka.com:8000/v1` | Base URL for Llama HTTP endpoint |
| `llm_api_key` | `LLM_API_KEY` | `""` | Bearer token (omitted from header if empty or `"dummy"`) |
| `llm_model` | `LLM_MODEL` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Model name sent in the request payload |
| `llm_timeout` | `LLM_TIMEOUT` | `120` | HTTP request timeout in seconds |
| `temperature` | `LLM_TEMPERATURE` | `0.3` | Generation temperature (lower = more deterministic) |
| `max_tokens` | `LLM_MAX_TOKENS` | `512` | Maximum tokens in LLM response |
| `verbose` | `LLM_VERBOSE` | `false` | Enable verbose payload/response debug logging |
| `aws_profile` | `AWS_PROFILE` | `my-sso` | boto3 SSO profile (SageMaker mode only) |
| `aws_region` | `AWS_REGION` | `us-east-2` | AWS region (SageMaker mode only) |
| `sagemaker_endpoint` | `SAGEMAKER_ENDPOINT` | `llama-3-2-3b-tgi-cpu-endpoint` | SageMaker endpoint name |
| `embedding_model` | *(code default)* | `models/all-MiniLM-L6-v2` | Local embedding model path |
| `chunk_size` | *(code default)* | `400` | Document chunk size for text splitting |
| `chunk_overlap` | *(code default)* | `60` | Overlap between consecutive chunks |
| `retriever_k` | *(code default)* | `4` | Number of FAISS chunks retrieved per query |
| `vector_store_dir` | *(code default)* | `vector_store/faiss_index` | Path where FAISS index is saved/loaded |
| `symptoms_conditions_path` | *(code default)* | `data/symptoms_conditions.json` | Symptom-to-condition data |
| `conditions_info_path` | *(code default)* | `data/conditions_info.json` | Condition detail data |
| `preventive_tips_path` | *(code default)* | `data/preventive_tips.json` | Preventive tips data |
| `show_retrieved_sources` | `SHOW_RETRIEVED_SOURCES` | `false` | Show source expanders in chat UI |

---

## 🎨 UI Text Configuration

All user-facing copy is decoupled from runtime logic in `ui_text_config.py`.

| Class | Key Fields |
|---|---|
| `PageText` | `page_title`, `page_icon`, `app_title`, `load_llm_spinner`, `load_vector_store_spinner`, `load_rag_spinner`, `header_caption_template`, `chat_input_placeholder`, `response_spinner`, `source_expander_label`, `source_fallback_label`, `error_message_template`, `footer_html` |
| `SidebarText` | `title`, `tagline`, `disclaimer`, `quick_guide_heading`, `structured_input_expander`, `symptom_label`, `duration_label`, `duration_options`, `severity_label`, `extra_symptoms_label`, `symptom_defaults`, `build_query_button`, `build_query_success`, `primary_symptom_required_warning`, `clear_chat_button`, `about_expander`, `about_template` |
| `ChatText` | `welcome_message` |

To update any UI text: edit `ui_text_config.py` — no changes needed anywhere else.

---

## 🏥 Healthcare Data

All data is curated from **WHO and CDC** public health guidance.

### `data/symptoms_conditions.json` — 30 entries

Each entry converted to a `Document` as:
```
Symptom: fever
Body System: Immune / General
Description: A temporary increase in body temperature...
Commonly associated conditions: influenza, common cold, COVID-19, ...
Source: WHO / CDC
```
Metadata: `{source_file: "symptoms_conditions", symptom, body_system}`

### `data/conditions_info.json` — 26 entries

Each entry converted to a `Document` as:
```
Condition: Flu (influenza)
Description: Influenza is a contagious respiratory illness...
Common Causes: Influenza A or B virus; ...
Typical Symptoms: Fever, Chills, Cough, ...
General Care Tips: Rest and stay hydrated | ...
When to Seek Care: Seek care immediately if you have difficulty breathing...
Recommended Specialist: General Practitioner (GP) / Family Medicine
Prevention: Annual flu vaccine, Hand hygiene, ...
Source: CDC
```
Metadata: `{source_file: "conditions_info", condition, specialist}`

### `data/preventive_tips.json` — 10 entries

Each entry converted to a `Document` as:
```
Preventive Health Category: Nutrition
Tips:
- Eat a variety of fruits and vegetables...
Source: WHO
```
Metadata: `{source_file: "preventive_tips", category}`

**Categories:** Nutrition, Physical Activity, Sleep, Mental Health, Vaccination, Hygiene, Smoking Cessation, Alcohol, Cardiovascular Health, Diabetes Prevention.

**Total: ~66 documents → ~90+ FAISS chunks** (chunk_size=400, overlap=60)

---

## 🔄 RAG Pipeline — How It Works

### Step 1 — Data Loading (`rag/data_loader.py`)
Three JSON files loaded via `load_all_documents()`. Each record converted to a structured `LangChain Document` with typed `metadata` used for source labelling in the UI.

### Step 2 — Chunking and Indexing (`rag/vector_store.py`)
- `RecursiveCharacterTextSplitter` with `chunk_size=400`, `chunk_overlap=60`, separators `["\n\n", "\n", " ", ""]`
- Each chunk debug-logged with source, length, and 200-char preview
- `HuggingFaceEmbeddings` (all-MiniLM-L6-v2, CPU, normalised) generates dense vectors
- `FAISS.from_documents()` builds the index; `save_local()` persists it
- On subsequent runs: `FAISS.load_local()` skips rebuild entirely

### Step 3 — Query Retrieval (`rag/retriever.py`)
- User query embedded with the same model
- FAISS retrieves `top-k=4` most similar chunks by cosine similarity
- Chunks concatenated and passed as `{context}` in `RAG_PROMPT`

### Step 4 — LLM Generation (`llm/`)
- `get_llm()` returns cached `EndpointLLM`
- `RetrievalQA` invokes `EndpointLLM._call(prompt)` with the fully assembled prompt
- Response extracted by `_extract_content()` handling multiple response shapes

### Step 5 — Safety Post-Processing (`utils/safety_filter.py`)
- `sanitize_response()`: 10 regex replacements + `_CONSULT_REMINDER` appended

### Step 6 — Display (`app.py`, `ui/chat_interface.py`)
- `st.markdown(safe_response)` renders the answer
- `retrieve_sources()` called separately — source expander shown when `cfg.show_retrieved_sources=True`
- Source labels resolved: `condition` → `symptom` → `category` → `source_file`
- `add_message()` persists response + sources to session state

---

## 🔐 Prompt Engineering and Safety Policy

`RAG_PROMPT` in `llm/prompt_templates.py` enforces at inference time:

### Safety Rules

| Rule | Instruction in prompt |
|---|---|
| No diagnosis | "Do NOT diagnose, prescribe, or provide personalized treatment plans" |
| Cautious language | Must use "may be associated with", "could be related to", "commonly linked to" |
| Context-only answers | "Use only the context below. Do not invent facts." |
| Off-topic deflection | Non-health questions must be redirected |
| Emergency escalation | Severe/sudden symptoms must be directed to emergency care |
| One consultation reminder | "End with exactly one consultation reminder sentence" |

### Style Rules

| Rule | Instruction in prompt |
|---|---|
| No preamble | "Start directly with helpful content. No preamble." |
| No meta-output | "Do NOT output instruction text" |
| No boilerplate phrases | Banned: "Based on your description", "Based on the provided context", "According to the context" |
| Concise and specific | "Keep response concise, specific, and user-friendly" |

### Structured Output Format

Every response must follow:

```
What this may be related to:
- 2–4 concrete possibilities grounded in context

General care tips:
- 3–5 practical, low-risk self-care suggestions

When to seek medical care:
- Clear red-flag symptoms and urgency guidance

Recommended specialist:
- Most relevant clinician type(s) with brief reason

[one consultation reminder sentence]
```

---

## 🛡️ Safety Filter — Input and Output Guards

Implemented in `utils/safety_filter.py`.

### Layer 1 — Input Domain Gate

`is_health_related(query)` checks the lowercased query against 60+ health keywords including:

```
symptom, pain, fever, cough, headache, doctor, medicine, treatment, diagnosis,
disease, condition, health, hospital, medication, infection, injury, rash, nausea,
fatigue, breathing, chest, blood, heart, skin, stomach, throat, cold, flu, allergy,
dizzy, swelling, ache, hurts, sore, tired, weakness, appetite, urination, itching,
burn, numbness, tingling, vision, hearing, anxiety, depression, mental, sleep,
insomnia, diabetes, cancer, stroke, asthma, hypertension, pressure, pulse,
temperature, clinic, surgery, specialist, consult, prescription, tablet, pill,
dose, medical, diagnose, examine, test, scan, xray, x-ray, mri, blood test,
urine, stool, pregnant, pregnancy, period, menstrual, blister, wound, bleeding ...
```

- Returns `True` + logs matched keywords (debug level) if health-domain
- Returns `False` + logs a warning if off-domain → RAG pipeline **not called**
- User sees a polite off-domain message with usage examples

### Layer 2 — Output Phrase Sanitizer

`sanitize_response(text)` applies 10 regex substitutions:

| Detected phrase | Replaced with |
|---|---|
| `You have` | `This may be associated with` |
| `you have` | `this may be associated with` |
| `You are diagnosed with` | `The symptoms described could be related to` |
| `you are diagnosed with` | `the symptoms described could be related to` |
| `You definitely have` | `The information suggests a possible association with` |
| `you definitely have` | `the information suggests a possible association with` |
| `This is clearly` | `This may be` |
| `this is clearly` | `this may be` |
| `You suffer from` | `You may be experiencing symptoms associated with` |
| `you suffer from` | `you may be experiencing symptoms associated with` |

**Disclaimer appended to every response:**

> ⚕️ *This information is for general educational purposes only and does not constitute medical advice, diagnosis, or treatment. Please consult a qualified healthcare professional for personalised medical guidance.*

Each replacement is individually logged as a warning with pattern, replacement, and count for full auditability.

---

## 📊 Logging and Observability

`logger_config.py` provides `StructuredFormatter` — colored, pipe-delimited, emoji-prefixed console output.

### Log Format

```
2026-03-30 19:01:23.456 | ℹ️  INFO    | app | Initialising LLM client
2026-03-30 19:01:24.123 | ✅ INFO    | rag.vector_store | FAISS index loaded
2026-03-30 19:01:25.789 | ⚠️  WARNING | utils.safety_filter | Off-domain query detected
2026-03-30 19:01:26.001 | ❌ ERROR   | llm.LlamaClient | Connection error
```

### Log Levels

| Level | Color | Emoji |
|---|---|---|
| DEBUG | Blue | 🔧 |
| INFO | Green | ℹ️ |
| WARNING | Yellow | ⚠️ |
| ERROR | Red | ❌ |
| CRITICAL | Bright Red | 🚨 |

### Logging Helpers

| Helper | Purpose |
|---|---|
| `get_logger(name)` | Returns a configured logger for a module |
| `log_section(logger, title)` | Prints a bold visual section header separator |
| `log_step(logger, n, msg)` | Numbered pipeline step (e.g. `[Step 1]`) |
| `log_success(logger, msg)` | INFO-level with ✅ prefix |
| `log_error(logger, msg, exc)` | ERROR-level with ❌ prefix + optional exception |
| `log_warning(logger, msg)` | WARNING-level with ⚠️ prefix |

---

## 📦 Prerequisites

- **Python 3.11** (dependency set built for 3.11)
- **A running LLM endpoint** accessible from your machine:
  - Llama HTTP mode: any `/v1/chat/completions`-compatible endpoint
  - SageMaker mode: AWS credentials with access to your endpoint
- **pip** with `venv`
- The embedding model is loaded from `models/all-MiniLM-L6-v2/` (local copy included — **no internet required**)

---

## 🚀 Installation

### 1. Open the project directory

```bash
cd "wipro-intel-genai-hackathon"
```

### 2. Create and activate virtual environment

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
```

### 3. Upgrade pip and install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> ⚠️ `torch==2.10.0` and `sentence-transformers==5.3.0` are large. First install may take several minutes.

### 4. Download the embedding model locally

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('models/all-MiniLM-L6-v2')"
```

> 📥 **What this does:** Downloads and saves the `all-MiniLM-L6-v2` embedding model to `models/all-MiniLM-L6-v2/` locally.
>
> 🌐 **Why:** The app loads the model from disk at runtime — **no internet connection required** after this one-time download.
>
> ⏭️ **Skip if:** The `models/all-MiniLM-L6-v2/` directory already exists in the project root (it is included by default in this repo).

### 5. Configure your LLM endpoint

```bash
export LLM_MODE="llama"
export LLM_BASE_URL="http://<your-tgi-host>:<port>/v1"
export LLM_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export LLM_API_KEY=""
```

Or edit defaults directly in `config.py`.

---

## ▶️ Running the App

```bash
python -m streamlit run app.py
```

On a specific port:

```bash
python -m streamlit run app.py --server.port 8502
```

#### First run
1. LLM client initialised from config
2. Embedding model loaded from `models/all-MiniLM-L6-v2/` (local, fast)
3. FAISS index built from JSON data → saved to `vector_store/faiss_index/`
4. App ready

#### Subsequent runs
1. LLM client re-created (fast)
2. Embedding model loaded from local path (fast)
3. FAISS index loaded from `vector_store/faiss_index/` — **no rebuild**
4. App immediately ready

![Minimal Streamlit Page](static/images/main_app.png)

---

## ✅ Sanity Checks

All sanity scripts live under `sanity_scripts/`.  
Run these **before** launching the main app.

### Check 1 — Library imports (13 libraries)

```bash
python sanity_scripts/sanity_library_imports.py
```

Validates: `requests`, `streamlit`, `llm.endpoint_client`, `langchain`, `langchain_core`, `langchain_community`, `langchain_huggingface`, `langchain_text_splitters`, `faiss`, `sentence_transformers`, `transformers`, `torch`

Expected output:
```
============================================================
Library Import Validation
============================================================
Validating 12 required libraries...
 ✅ SUCCESS: requests import
 ✅ SUCCESS: streamlit import
 ✅ SUCCESS: Endpoint client import
 ✅ SUCCESS: LangChain import
 ✅ SUCCESS: LangChain Core import
 ✅ SUCCESS: LangChain Community import
 ✅ SUCCESS: LangChain HuggingFace import
 ✅ SUCCESS: LangChain Text Splitters import
    Loading faiss.
    Successfully loaded faiss.
 ✅ SUCCESS: FAISS import
 ✅ SUCCESS: Sentence Transformers import
 ✅ SUCCESS: Transformers import
 ✅ SUCCESS: Torch import
 ✅ SUCCESS: All 12 libraries imported successfully
 ✅ SUCCESS: Sanity result PASSED
```

### Check 2 — LLM endpoint inference (most important)

```bash
python sanity_scripts/sanity_llama_model.py
```

Tests full endpoint connectivity and inference. Sends `"What is Machine Learning?"` and measures latency.

Expected output:
```
============================================================
Endpoint Configuration
============================================================
LLM_MODE = 'llama'
Target → Llama HTTP endpoint: http://wiphackq0vcsii.cloudloka.com:8000/v1
         Model: meta-llama/Meta-Llama-3.1-8B-Instruct
Temperature: 0.3 | Max tokens: 512 | Timeout: 120s

STEP 1: Initializing LLM client
LLM_MODE = 'llama'
Initialising Llama HTTP endpoint 'http://wiphackq0vcsii.cloudloka.com:8000/v1' (model=meta-llama/Meta-Llama-3.1-8B-Instruct)
Configured Llama HTTP client — url='http://wiphackq0vcsii.cloudloka.com:8000/v1' model='meta-llama/Meta-Llama-3.1-8B-Instruct'
✅ SUCCESS: Llama LangChain LLM ready — url='http://wiphackq0vcsii.cloudloka.com:8000/v1'
✅ SUCCESS: LLM client initialized
 
STEP 2: Running test inference
[TRACE] LlamaEndpointClient.invoke() — sending to endpoint 'http://wiphackq0vcsii.cloudloka.com:8000/v1'
POST http://wiphackq0vcsii.cloudloka.com:8000/v1/chat/completions
Response received (NN chars)
✅ SUCCESS: Inference completed successfully in XX.XX seconds. Response length: NN characters.
LLM Prompt: What is Machine Learning?
✅ SUCCESS: 📝 LLM Response: <response content>
 
✅ SUCCESS: Sanity check PASSED
```

### Check 3 — Full module and data validation

```bash
python sanity_scripts/sanity_app_components.py
```

Validates (in order): RAG document loading (symptoms/conditions/preventive counts), embedder import, vector store import, retriever import, LLM model loading, UI component imports.

Expected output:
```
============================================================
RAG MODULE VALIDATION
============================================================

STEP 1: Loading RAG documents (symptoms, conditions, preventive)
Reading JSON file: data/symptoms_conditions.json
Loaded 30 symptom documents
Reading JSON file: data/conditions_info.json
Loaded 26 condition documents
Reading JSON file: data/preventive_tips.json
Loaded 10 preventive documents
✅ SUCCESS: Documents loaded — Symptoms: 30, Conditions: 26, Preventive: 10, Total: 66

STEP 2: Validating embedder module
✅ SUCCESS: Embedder module imported successfully

STEP 3: Validating vector store module
✅ SUCCESS: Vector store module imported successfully

STEP 4: Validating retriever module
✅ SUCCESS: Retriever module imported successfully
 
============================================================
LLM MODULE VALIDATION
============================================================

STEP 5: Loading language model
LLM_MODE = 'llama'
Initialising Llama HTTP endpoint 'http://wiphackq0vcsii.cloudloka.com:8000/v1' (model=meta-llama/Meta-Llama-3.1-8B-Instruct)
Configured Llama HTTP client — url='http://wiphackq0vcsii.cloudloka.com:8000/v1' model='meta-llama/Meta-Llama-3.1-8B-Instruct'
✅ SUCCESS: Llama LangChain LLM ready — url='http://wiphackq0vcsii.cloudloka.com:8000/v1'
✅ SUCCESS: Language model loaded — Type: EndpointLLM

============================================================
UI MODULE VALIDATION
============================================================

STEP 6: Validating UI components
✅ SUCCESS: UI components imported successfully

✅ SUCCESS: Sanity check completed successfully
```
### Check 4 — Interactive Streamlit endpoint UI

```bash
python -m streamlit run sanity_scripts/sanity_app.py
```

- Opens a minimal Streamlit page to test the LLM endpoint interactively. 
- User can enter any query, which is sent to the endpoint and the response is displayed. 
- This validates the full request/response cycle in a Streamlit context.

![Minimal Streamlit Page](static/images/minimal_streamlit_page.png)

---

## 🖥️ UI Features

### Main Chat Interface
- Multi-turn conversational chat with persistent session state across reruns
- User messages: `👤` avatar; Assistant responses: `🩺` avatar
- Sidebar prefilled queries auto-submitted on next rerun
- Optional **"📚 View retrieved health knowledge sources"** expander per response (enabled by `SHOW_RETRIEVED_SOURCES=true`)
- Source metadata labels resolved: `condition` → `symptom` → `category` → `source_file`
- Inline error messages for LLM endpoint failures

### Sidebar (400px fixed width, blue-themed buttons)
- **Disclaimer banner** — always visible (`st.error` red box)
- **Quick Symptom Guide** expander:
  - Primary symptom — dynamic dropdown from `symptoms_conditions.json` (falls back to static defaults)
  - Duration — 6 options: Less than 1 day → More than 1 month
  - Severity — slider 1–10, default 5
  - Additional symptoms — free text
  - **"📋 Prepare My Symptom Question"** — validates primary symptom, composes structured query, prefills chat
- **"🗑️ Clear Chat History"** — resets session to initial welcome message
- **"ℹ️ About this app"** — expandable: tech stack, disclaimer, model info

---

## 💬 Example Prompts

```
I have fever, sore throat, and body ache for 2 days. What could this be?
```
```
My primary symptom is Persistent Cough, for 3 weeks, severity 6/10.
What could this be and what should I do?
```
```
I feel tired, very thirsty, and urinate frequently. What conditions are commonly linked to this?
```
```
I have had shortness of breath and chest tightness for 1 week. What should I know?
```
```
What are some general tips to prevent cardiovascular disease?
```
```
I have a severe headache with neck stiffness and sensitivity to light.
```
```
I have been feeling very low in mood and have no energy for the past month.
```

---

## 🐛 Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ConnectionError` / `HTTPError` on startup | LLM endpoint not reachable | Verify `LLM_BASE_URL`, check endpoint is running |
| `ValueError: Could not extract text from response` | Unexpected endpoint response shape | Check endpoint JSON format — both `choices` and `generated_text` shapes are handled |
| `ModuleNotFoundError` | Dependency missing | Run `python -m pip install -r requirements.txt` |
| Embedding model load fails | `models/all-MiniLM-L6-v2/` missing | Check directory exists; or set `embedding_model = "sentence-transformers/all-MiniLM-L6-v2"` in `config.py` for HuggingFace Hub download |
| FAISS index stale or corrupted | Data changed after initial build | Delete `vector_store/faiss_index/` and restart — rebuilds automatically |
| Slow first startup | Building FAISS index | Normal; subsequent runs are fast |
| `streamlit: command not found` | Streamlit not on PATH | Use `python -m streamlit run app.py` |
| Port conflict | 8501 in use | Use `--server.port 8502` |
| Sources never shown in UI | `SHOW_RETRIEVED_SOURCES` not set | `export SHOW_RETRIEVED_SOURCES=true` |
| Response contains diagnostic language | New phrase pattern not covered | Add pattern to `_DIAGNOSIS_PATTERNS` in `utils/safety_filter.py` |
| Sidebar symptom list is empty | JSON parse failure | Check `data/symptoms_conditions.json` schema — fallback static defaults will be used |
| Timeout errors | Slow endpoint | Increase `LLM_TIMEOUT` (default 120s) via env var or `config.py` |
| `ValueError: Unknown LLM_MODE` | Invalid `LLM_MODE` value | Set to `"llama"` or `"sagemaker"` only |

---

## 📁 File Reference

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit entry point. Orchestrates session state, resource caching (`@st.cache_resource`), input domain filtering, RAG invocation, output safety, and chat rendering. |
| `config.py` | `AppConfig` dataclass — single source of truth for all runtime settings. Exported as `cfg` singleton. Every field env-var overridable. Injects `UIText` as `cfg.ui_text`. Logs active overrides at startup. |
| `ui_text_config.py` | `PageText`, `SidebarText`, `ChatText` frozen dataclasses assembled into `UIText`. All user-facing strings live here. No runtime logic. |
| `logger_config.py` | `StructuredFormatter` and helpers (`get_logger`, `log_section`, `log_step`, `log_success`, `log_error`, `log_warning`) for colored, timestamped, emoji-prefixed structured console logging. |
| `llm/endpoint_client.py` | `SageMakerEndpointClient` (boto3) and `LlamaEndpointClient` (requests). Both expose `.invoke(prompt) → EndpointResponse`. Shared `_coerce_prompt()` and `_extract_content()` handle multiple response shapes. |
| `llm/langchain_adapter.py` | `EndpointLLM` — LangChain `LLM` subclass wrapping any endpoint client. Implements `_call()`, `_llm_type`, `_identifying_params`. Pydantic v2 compatible. |
| `llm/model_loader.py` | `get_llm()` with `@lru_cache(maxsize=1)`. Reads `cfg.llm_mode`, constructs `EndpointLLM`. Raises `ValueError` for unknown modes. |
| `llm/prompt_templates.py` | `RAG_PROMPT` — `PromptTemplate` with safety rules, style rules, and structured output format. Inputs: `{context}` and `{question}`. |
| `rag/data_loader.py` | Loads 3 JSON files. Converts each record into structured `LangChain Document` with typed metadata. Functions: `load_symptom_documents`, `load_condition_documents`, `load_preventive_documents`, `load_all_documents`. |
| `rag/embedder.py` | `get_embeddings()` with `@lru_cache(maxsize=1)`. Returns `HuggingFaceEmbeddings` (all-MiniLM-L6-v2, CPU, normalised). |
| `rag/vector_store.py` | `build_or_load_vector_store()` — loads existing FAISS index or builds/saves fresh one. `_split_documents()` — chunking with per-chunk debug logging. |
| `rag/retriever.py` | `build_rag_chain()` — `RetrievalQA` with stuff chain, k=4, `RAG_PROMPT`. `retrieve_sources()` — separate top-k retrieval for UI display (content truncated to 300 chars). |
| `ui/sidebar.py` | `render_sidebar()` — dynamic symptom loading with fallback, structured query builder with validation, disclaimer, clear chat, about section. 400px CSS + blue button theme injected. |
| `ui/chat_interface.py` | `init_chat_state()`, `render_chat_history()`, `add_message()`. Multi-turn session state. Conditional source expanders controlled by `cfg.show_retrieved_sources`. |
| `utils/safety_filter.py` | `is_health_related()` (input gate, 60+ keywords), `get_off_domain_message()`, `sanitize_response()` (10 regex replacements + disclaimer). Each replacement auditably logged. |
| `data/symptoms_conditions.json` | 30 symptom records (WHO/CDC). |
| `data/conditions_info.json` | 26 condition records with full care detail (WHO/CDC). |
| `data/preventive_tips.json` | 10 prevention category records (WHO). |
| `vector_store/faiss_index/` | Auto-generated FAISS index. Safe to delete to trigger a full rebuild on next startup. |
| `models/all-MiniLM-L6-v2/` | Local copy of sentence-transformers embedding model. No internet download at runtime. |
| `sanity_scripts/sanity_library_imports.py` | Validates 13 required library imports. Exits with code 1 on any failure. |
| `sanity_scripts/sanity_llama_model.py` | End-to-end LLM endpoint test with latency measurement. Supports both `llama` and `sagemaker` modes from `cfg`. |
| `sanity_scripts/sanity_app_components.py` | Full module import + data load validation across RAG (documents, embedder, vector store, retriever), LLM, and UI layers. |
| `sanity_scripts/sanity_app.py` | Minimal Streamlit interactive endpoint test UI. |
| `requirements.txt` | Pinned Python 3.11 dependency list. Source of truth for reproducible installs. |

---

## 🏆 Hackathon Notes

- Single configurable **LLM backend** — swap between `llama` (HTTP) and `sagemaker` (AWS) with one env var
- No provider lock-in — HTTP client is generic, not tied to any API provider SDK
- Embeddings and FAISS run fully **locally on CPU** — no external vector database needed
- Data sourced from **WHO / CDC** public health guidance — cited in every document
- **Dual-layer safety**: prompt-level constraints at inference + regex post-processing on output
- **Input domain gate**: non-health queries blocked before any LLM call is made
- All UI copy in `ui_text_config.py` — fully separable from runtime logic
- Keep the **disclaimer banner visible** at all times during judging

---

*Built for the Wipro × Intel Hackathon · March 2026*
