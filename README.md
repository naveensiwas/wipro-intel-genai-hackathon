# 🩺 Healthcare Symptom Information Assistant

> Built for the **Wipro × Intel Hackathon** · March 2026

A conversational health-information assistant that accepts natural-language symptom descriptions, retrieves relevant knowledge from a structured WHO/CDC knowledge base using RAG, and returns safe, non-diagnostic health guidance via a configurable remote LLM endpoint.

---

## ⚠️ Medical Disclaimer

This tool provides **general health information only**.  
It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.  
Always consult a qualified healthcare provider for any medical concerns.  
For severe, sudden, or worsening symptoms — **seek emergency care immediately**.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Project Structure](#-project-structure)
4. [Tech Stack](#-tech-stack)
5. [LLM Endpoint Configuration](#-llm-endpoint-configuration)
6. [All Configuration Parameters](#-all-configuration-parameters)
7. [Healthcare Data](#-healthcare-data)
8. [RAG Pipeline — How It Works](#-rag-pipeline--how-it-works)
9. [Prompt Safety Policy](#-prompt-safety-policy)
10. [Safety Filter](#-safety-filter-post-processing)
11. [Prerequisites](#-prerequisites)
12. [Installation](#-installation)
13. [Running the App](#-running-the-app)
14. [Sanity Checks](#-sanity-checks)
15. [Test Check](#-test-check)
16. [UI Features](#-ui-features)
17. [Example Prompts](#-example-prompts)
18. [Troubleshooting](#-troubleshooting)
19. [File Reference](#-file-reference)

---

## 🎯 Project Overview

The assistant:

- Accepts freeform or structured symptom queries from the user
- Embeds the query using a local `sentence-transformers` model
- Retrieves the top-4 most relevant chunks from a local FAISS index built over WHO/CDC-curated JSON data
- Sends a safety-constrained prompt (retrieved context + user question) to a remote LLM endpoint
- Post-processes the response through a regex-based safety filter
- Displays the answer and expandable source citations in a Streamlit chat UI

The app **never diagnoses**, **never prescribes**, and **always recommends consulting a healthcare professional**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)                 │
│  ┌───────────────┐ ┌─────────────────────────────────┐  │
│  │  sidebar.py   │ │      chat_interface.py          │  │
│  │  - Disclaimer │ │  - Welcome message              │  │
│  │  - Quick guide│ │  - Message history              │  │
│  │  - Clear chat │ │  - Source citation expander     │  │
│  └───────────────┘ └─────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────┘
                           │ User Query
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   RAG Pipeline                          │
│                                                         │
│  1. Embed query         (rag/embedder.py)               │
│     └─ all-MiniLM-L6-v2 via HuggingFaceEmbeddings       │
│                                                         │
│  2. Vector search       (rag/vector_store.py)           │
│     └─ FAISS index  ←── built from JSON data            │
│          top-k=4 similar chunks                         │
│                                                         │
│  3. Build RAG chain     (rag/retriever.py)              │
│     └─ RetrievalQA (LangChain "stuff" chain)            │
│          with safety-constrained PromptTemplate         │
│                                                         │
│  4. Call LLM endpoint   (llm/model_loader.py)           │
│     └─ EndpointChatClient (llm/endpoint_client.py)      │
│          POST {LLM_BASE_URL}/chat/completions           │
│                                                         │
│  5. Safety filter       (utils/safety_filter.py)        │
│     └─ regex replacements + disclaimer append           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
                  Response + Sources displayed
```

---

## 🗂️ Project Structure

```
Wipro_Intel_Hackathon_LLM_EP/
│
├── app.py                         ← Main Streamlit entry point
├── config.py                      ← Centralised configuration (all env-overridable)
├── requirements.txt               ← Pinned Python dependencies
│
├── data/
│   ├── symptoms_conditions.json   ← 30 symptoms mapped to conditions (WHO/CDC)
│   ├── conditions_info.json       ← 26 conditions with full care detail
│   └── preventive_tips.json       ← 10 prevention categories
│
├── llm/
│   ├── __init__.py
│   ├── endpoint_client.py         ← Generic HTTP client for the LLM endpoint
│   ├── model_loader.py            ← Cached singleton factory (get_llm)
│   └── prompt_templates.py        ← Safety-constrained RAG PromptTemplate
│
├── rag/
│   ├── __init__.py
│   ├── data_loader.py             ← Loads JSON → LangChain Documents
│   ├── embedder.py                ← HuggingFaceEmbeddings (cached singleton)
│   ├── vector_store.py            ← FAISS build/load/persist
│   └── retriever.py               ← RetrievalQA chain assembly
│
├── ui/
│   ├── __init__.py
│   ├── sidebar.py                 ← Disclaimer, Quick Symptom Guide, About
│   └── chat_interface.py          ← Chat state, history rendering, add_message
│
├── utils/
│   ├── __init__.py
│   └── safety_filter.py           ← Post-generation safety language filter
│
├── vector_store/
│   └── faiss_index/               ← Auto-generated FAISS index (created on first run)
│       ├── index.faiss
│       └── index.pkl
│
├── sanity_library_imports.py      ← Validates core library imports
├── sanity_llama_model.py          ← End-to-end endpoint inference test
├── sanity_app.py                  ← Streamlit + endpoint UI sanity check
└── test_check.py                  ← Full module import + data load test
```

---

## 🛠️ Tech Stack

| Layer | Library / Tool | Version |
|---|---|---|
| UI | Streamlit | 1.32.2 |
| LLM | Configurable remote LLM endpoint | — |
| HTTP client | `requests` | 2.32.5 |
| RAG orchestration | LangChain | 0.3.0 |
| Embeddings | `langchain-huggingface` + `sentence-transformers` | 0.1.0 / 5.3.0 |
| Vector store | FAISS (CPU) | `faiss-cpu` 1.13.2 |
| Text splitting | `langchain-text-splitters` | 0.3.0 |
| Language | Python | 3.11 |

---

## ⚙️ LLM Endpoint Configuration

All LLM settings live in **one place**: `config.py`.  
Every setting is also overridable via an **environment variable** — no code changes needed to switch endpoints.

### `config.py` defaults

```python
llm_base_url : str   = os.getenv("LLM_BASE_URL",    "http://localhost:8000/v1")
llm_model    : str   = os.getenv("LLM_MODEL",        "llama-3-2-3b-tgi-cpu-endpoint")
llm_api_key  : str   = os.getenv("LLM_API_KEY",      "dummy")
llm_timeout  : int   = os.getenv("LLM_TIMEOUT",      60)
temperature  : float = os.getenv("LLM_TEMPERATURE",  0.3)
max_tokens   : int   = os.getenv("LLM_MAX_TOKENS",   512)
verbose      : bool  = os.getenv("LLM_VERBOSE",      False)
```

### How the endpoint is called

`llm/endpoint_client.py` sends a `POST` to:

```
{LLM_BASE_URL}/chat/completions
```

with the body:

```json
{
  "model": "<llm_model>",
  "messages": [{ "role": "user", "content": "<prompt>" }],
  "temperature": 0.3,
  "max_tokens": 512
}
```

It handles both standard `/choices[0].message.content` and HuggingFace TGI `generated_text` response shapes automatically.

### Setting your endpoint — Option A: Environment variables (recommended)

```bash
export LLM_BASE_URL="http://<your-tgi-host>:<port>/v1"
export LLM_MODEL="llama-3-2-3b-tgi-cpu-endpoint"
export LLM_API_KEY="dummy"          # set a real key if your endpoint requires one
export LLM_TIMEOUT="60"
export LLM_TEMPERATURE="0.3"
export LLM_MAX_TOKENS="512"
```

### Setting your endpoint — Option B: Edit `config.py` directly

Change the default values in `config.py`:

```python
llm_base_url: str = os.getenv("LLM_BASE_URL", "http://<your-tgi-host>:<port>/v1")
llm_model:    str = os.getenv("LLM_MODEL",    "llama-3-2-3b-tgi-cpu-endpoint")
llm_api_key:  str = os.getenv("LLM_API_KEY",  "your-api-key-here")
```

All other modules (`app.py`, `sanity_app.py`, `sanity_llama_model.py`, etc.) read from `cfg` automatically — no other file needs changing.

---

## 📐 All Configuration Parameters

| Parameter | Env Var | Default | Description |
|---|---|---|---|
| `llm_base_url` | `LLM_BASE_URL` | `http://localhost:8000/v1` | Base URL of the LLM endpoint (`/chat/completions` is appended) |
| `llm_model` | `LLM_MODEL` | `llama-3-2-3b-tgi-cpu-endpoint` | Model name sent in the request payload |
| `llm_api_key` | `LLM_API_KEY` | `dummy` | Bearer token sent in `Authorization` header |
| `llm_timeout` | `LLM_TIMEOUT` | `60` | HTTP request timeout in seconds |
| `temperature` | `LLM_TEMPERATURE` | `0.3` | Generation temperature (lower = more deterministic) |
| `max_tokens` | `LLM_MAX_TOKENS` | `512` | Maximum tokens in the LLM response |
| `verbose` | `LLM_VERBOSE` | `false` | Enable verbose logging |
| `embedding_model` | *(hardcoded)* | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `chunk_size` | *(hardcoded)* | `400` | Document chunk size for text splitting |
| `chunk_overlap` | *(hardcoded)* | `60` | Overlap between consecutive text chunks |
| `retriever_k` | *(hardcoded)* | `4` | Number of FAISS chunks retrieved per query |
| `vector_store_dir` | *(hardcoded)* | `vector_store/faiss_index` | Path where the FAISS index is saved/loaded |
| `symptoms_conditions_path` | *(hardcoded)* | `data/symptoms_conditions.json` | Path to symptom data |
| `conditions_info_path` | *(hardcoded)* | `data/conditions_info.json` | Path to condition detail data |
| `preventive_tips_path` | *(hardcoded)* | `data/preventive_tips.json` | Path to prevention tips data |

---

## 🏥 Healthcare Data

All data is curated from **WHO and CDC** public health guidance and stored as structured JSON under `data/`.

### `data/symptoms_conditions.json`
30 entries. Each entry contains:

| Field | Example |
|---|---|
| `symptom` | `"fever"` |
| `description` | `"A temporary increase in body temperature..."` |
| `body_system` | `"Immune / General"` |
| `related_conditions` | `["influenza", "common cold", "COVID-19", ...]` |
| `source` | `"WHO / CDC"` |

### `data/conditions_info.json`
26 entries. Each entry contains:

| Field | Example |
|---|---|
| `name` | `"influenza"` |
| `common_name` | `"Flu"` |
| `description` | `"Influenza is a contagious respiratory illness..."` |
| `common_causes` | `["Influenza A or B virus", ...]` |
| `typical_symptoms` | `["Fever", "Chills", "Cough", ...]` |
| `general_care_tips` | `["Rest and stay hydrated", ...]` |
| `when_to_seek_care` | `"Seek care immediately if you have difficulty breathing..."` |
| `specialist_type` | `"General Practitioner (GP) / Family Medicine"` |
| `prevention` | `["Annual flu vaccine", "Hand hygiene", ...]` |
| `source` | `"CDC"` |

### `data/preventive_tips.json`
10 prevention categories. Each entry contains:

| Field | Example |
|---|---|
| `category` | `"Nutrition"` |
| `tips` | `["Eat a variety of fruits and vegetables...", ...]` |
| `source` | `"WHO"` |

**Categories covered:** Nutrition, Physical Activity, Sleep, Mental Health, Vaccination, Hygiene, Smoking Cessation, Alcohol, Cardiovascular Health, Diabetes Prevention.

**Total documents indexed in FAISS: ~66** (split into ~90+ chunks after text splitting).

---

## 🔄 RAG Pipeline — How It Works

### Step 1 — Data loading (`rag/data_loader.py`)
The three JSON files are loaded and converted into `LangChain Document` objects.  
Each document is a structured text block. Example for a symptom:

```
Symptom: fever
Body System: Immune / General
Description: A temporary increase in body temperature...
Commonly associated conditions: influenza, common cold, COVID-19, ...
Source: WHO / CDC
```

### Step 2 — Embedding + FAISS indexing (`rag/embedder.py`, `rag/vector_store.py`)
- Documents are split into chunks (size=400, overlap=60) by `RecursiveCharacterTextSplitter`
- Each chunk is embedded by `sentence-transformers/all-MiniLM-L6-v2` running on CPU
- FAISS stores the vectors locally at `vector_store/faiss_index/`
- **On first run:** the index is built and saved to disk
- **On subsequent runs:** the persisted index is loaded from disk (fast)

### Step 3 — Query retrieval (`rag/retriever.py`)
- The user query is embedded with the same model
- FAISS retrieves the top-4 most similar document chunks using cosine similarity
- Chunks are passed as `{context}` into the LangChain `PromptTemplate`

### Step 4 — LLM call (`llm/model_loader.py`, `llm/endpoint_client.py`)
- `get_llm()` returns a cached `EndpointChatClient` singleton
- The RAG chain invokes `.invoke()` on the LLM
- `EndpointChatClient` sends a `POST /chat/completions` to the configured endpoint
- The response is extracted from `choices[0].message.content` or `generated_text`

### Step 5 — Safety filtering (`utils/safety_filter.py`)
- Diagnostic phrases (`"You have"`, `"You are diagnosed with"`, etc.) are replaced with cautious language
- A formal medical disclaimer is appended to every response

### Step 6 — Display (`app.py`, `ui/chat_interface.py`)
- The sanitized response is rendered in the Streamlit chat
- Retrieved source chunks are shown in a collapsible expander with metadata labels

---

## 🔐 Prompt Safety Policy

The prompt template in `llm/prompt_templates.py` enforces the following rules for the LLM at inference time:

| Rule | Enforcement |
|---|---|
| Role constraint | LLM is told it is a "health information assistant, not a doctor" |
| Cautious language | Must say "may be associated with", never "you have" |
| Context-only answers | Must answer only from the retrieved context — no hallucination |
| Specialist recommendation | Every response must end with a specialist recommendation |
| Emergency escalation | Severe/sudden symptoms must always be directed to emergency care |
| Off-topic deflection | Non-health questions must be politely declined |
| No medication advice | Must never suggest starting, stopping, or changing medication |

Structured output format enforced in the prompt:
1. What the symptoms may be associated with (from context)
2. General care and wellbeing tips
3. When to seek medical care
4. Recommended type of specialist to consult

---

## 🛡️ Safety Filter (Post-Processing)

`utils/safety_filter.py` applies additional safety on top of the prompt constraints.

**Phrase replacements:**

| Detected phrase | Replaced with |
|---|---|
| `You have` | `This may be associated with` |
| `you have` | `this may be associated with` |
| `You are diagnosed with` | `The symptoms described could be related to` |
| `You definitely have` | `The information suggests a possible association with` |
| `This is clearly` | `This may be` |
| `You suffer from` | `You may be experiencing symptoms associated with` |

**Disclaimer:** A formal disclaimer is appended to every response:

> *This information is for general educational purposes only and does not constitute medical advice, diagnosis, or treatment. Please consult a qualified healthcare professional for personalised medical guidance.*

---

## 📦 Prerequisites

- **Python 3.11** (required — the dependency set is built for 3.11)
- **A running LLM endpoint** accessible from your machine  
  (must support `POST /chat/completions` with a standard chat payload)
- **pip** and optionally `virtualenv` or `venv`
- Internet access on first run only (to download the embedding model ~90 MB from HuggingFace Hub)

---

## 🚀 Installation

### 1. Clone / open the project

```bash
cd "/Users/na20275126/Library/CloudStorage/OneDrive-Wipro/Wipro_Intel_Hackathon_LLM_EP"
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install all dependencies

```bash
python -m pip install -r requirements.txt
```

> ⚠️ `torch==2.10.0` and `sentence-transformers==5.3.0` are large packages. Installation may take several minutes on first run.

### 5. Configure your LLM endpoint

Set environment variables (recommended):

```bash
export LLM_BASE_URL="http://<your-tgi-host>:<port>/v1"
export LLM_MODEL="llama-3-2-3b-tgi-cpu-endpoint"
export LLM_API_KEY="dummy"
```

**Or** edit the defaults directly in `config.py` (lines 11–13):

```python
llm_base_url: str = os.getenv("LLM_BASE_URL", "http://<your-tgi-host>:<port>/v1")
llm_model:    str = os.getenv("LLM_MODEL",    "llama-3-2-3b-tgi-cpu-endpoint")
llm_api_key:  str = os.getenv("LLM_API_KEY",  "your-api-key-here")
```

---

## ▶️ Running the App

### Run sanity checks first (see [Sanity Checks](#-sanity-checks) section)

### Start the main app

```bash
streamlit run app.py
```

Or on a specific port:

```bash
streamlit run app.py --server.port 8502
```

#### What happens on first run

1. Streamlit connects to the configured LLM endpoint (cached for the session)
2. The embedding model (`all-MiniLM-L6-v2`) is downloaded from HuggingFace Hub (~90 MB) and cached locally
3. The FAISS index is built from the three JSON data files and saved to `vector_store/faiss_index/`
4. The app is ready

#### What happens on subsequent runs

1. The LLM client is re-created from config (fast)
2. The embedding model is loaded from local cache (fast)
3. The FAISS index is loaded from `vector_store/faiss_index/` (fast)
4. The app is immediately ready — no rebuild needed

---

## ✅ Sanity Checks

Run these checks **before** launching the main app to confirm each layer is working correctly.

### Check 1 — Library imports

Validates that the core endpoint client and Streamlit are importable.

```bash
python sanity_library_imports.py
```

Expected output:
```
✅ Endpoint client import OK
✅ Streamlit import OK
✅ LLM endpoint adapter import OK
```

---

### Check 2 — Endpoint inference

Calls your configured LLM endpoint with a test prompt and prints the response. This is the **most important** check before running the full app.

```bash
python sanity_llama_model.py
```

Expected output:
```
[ModelLoader] Connecting to LLM endpoint: http://localhost:8000/v1 (model=llama-3-2-3b-tgi-cpu-endpoint)
[ModelLoader] Remote LLM client initialised successfully.
✅ Endpoint response from llama-3-2-3b-tgi-cpu-endpoint: Machine learning is a branch of...
```

If this fails, verify `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` before proceeding.

---

### Check 3 — Streamlit + endpoint UI

A minimal Streamlit page that calls the configured endpoint interactively.

```bash
streamlit run sanity_app.py
```

Open the local URL in your browser, type a prompt, click **Run**, and verify a response appears.

---

## 🧪 Test Check

`test_check.py` performs a comprehensive import and data-load validation across all modules.

```bash
python test_check.py
```

Expected output:
```
Starting test...
Symptoms: 30, Conditions: 26, Preventive: 10, Total: 66
Embedder import OK
VectorStore import OK
Retriever import OK
ModelLoader import OK
UI imports OK
Test complete.
```

Any `ERROR:` line indicates a missing dependency or misconfiguration — fix before running the main app.

---

## 🖥️ UI Features

### Main chat interface
- Conversational chat with persistent session history
- User messages shown with 👤 avatar; assistant responses with 🩺 avatar
- Each assistant response is followed by a collapsible **"📚 View retrieved health knowledge sources"** expander showing the top-4 retrieved chunks and their metadata labels (condition, symptom, category, or source file)

### Sidebar
- **Disclaimer banner** — always visible in red
- **Quick Symptom Guide** — structured symptom builder:
  - Body area affected (12 options)
  - Duration (5 options)
  - Severity slider (1–10)
  - Additional freetext symptoms
  - Click **Build Query** to auto-populate the chat input
- **Clear Chat History** button — resets the session
- **About this app** expandable section with tech stack info

### Error handling
If the LLM endpoint is unreachable or returns an error, the app shows a descriptive error message in the chat and prompts the user to verify endpoint settings.

---

## 💬 Example Prompts

```
I have fever, sore throat, and body ache for 2 days. What could this be associated with?
```
```
I have an itchy skin rash on my arms for a week. What general care steps are recommended?
```
```
I feel tired, very thirsty, and urinate frequently. What conditions are commonly linked to this?
```
```
I have had a persistent cough and shortness of breath for 3 weeks. What should I know?
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
| `ConnectionError` or `HTTPError` on startup | LLM endpoint not reachable | Verify `LLM_BASE_URL` is correct and the endpoint is running |
| `ValueError: LLM endpoint response did not contain readable text content` | Endpoint returned unexpected JSON | Check the endpoint response shape; contact the endpoint operator |
| `ModuleNotFoundError` on any import | Dependencies not installed | Run `python -m pip install -r requirements.txt` |
| Embedding model download fails | No internet on first run | Allow internet access on first run or pre-cache the model manually |
| FAISS index stale / corrupted | Data changed after initial build | Delete `vector_store/faiss_index/` and restart — it will rebuild automatically |
| Slow first startup | Building FAISS index + downloading embeddings | Normal; subsequent runs are fast |
| `streamlit: command not found` | Streamlit not on PATH | Use `python -m streamlit run app.py` instead |
| Streamlit port conflict | Port 8501 already in use | Use `streamlit run app.py --server.port 8502` |
| Wrong model responding | `LLM_MODEL` mismatch | Update `LLM_MODEL` env var or the default in `config.py` |
| Response looks like a raw diagnosis | Safety filter missed a pattern | Check `utils/safety_filter.py` and add new patterns as needed |

---

## 📁 File Reference

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application entry point. Wires all components together. Handles resource caching, chat loop, error display, and footer. |
| `config.py` | `AppConfig` dataclass — single source of truth for all settings. Reads from environment variables with sensible defaults. Exported as `cfg` singleton. |
| `llm/endpoint_client.py` | `EndpointChatClient` class. Sends `POST /chat/completions` to the configured endpoint. Handles prompt coercion, auth header, and response parsing for both `choices` and `generated_text` shapes. `EndpointResponse` dataclass wraps the returned content. |
| `llm/model_loader.py` | `get_llm()` — `@lru_cache` singleton factory. Builds and returns an `EndpointChatClient` from `cfg`. Called once per app session. |
| `llm/prompt_templates.py` | `RAG_PROMPT` — a `PromptTemplate` with a strict system instruction enforcing safe, non-diagnostic, context-only responses. Inputs: `{context}` and `{question}`. |
| `rag/data_loader.py` | Loads the three JSON files and converts each record into a structured `LangChain Document` with `page_content` and `metadata`. Functions: `load_symptom_documents`, `load_condition_documents`, `load_preventive_documents`, `load_all_documents`. |
| `rag/embedder.py` | `get_embeddings()` — `@lru_cache` singleton. Returns a `HuggingFaceEmbeddings` instance for `all-MiniLM-L6-v2`, CPU device, with normalized embeddings. |
| `rag/vector_store.py` | `build_or_load_vector_store()` — Checks for an existing FAISS index at `vector_store/faiss_index/index.faiss`. If found, loads it; otherwise builds, saves, and returns a fresh index from all documents. |
| `rag/retriever.py` | `build_rag_chain(llm, vector_store)` — builds a `RetrievalQA` chain (LangChain "stuff" chain type) with the safety prompt. `retrieve_sources(vector_store, query)` — returns the top-k retrieved chunks for UI display. |
| `ui/sidebar.py` | `render_sidebar()` — renders the disclaimer, Quick Symptom Guide form, Clear Chat button, and About expander using values from `cfg`. |
| `ui/chat_interface.py` | `init_chat_state()` — initialises `st.session_state["messages"]` with a welcome message. `render_chat_history()` — renders all messages. `add_message(role, content, sources)` — appends to session state. |
| `utils/safety_filter.py` | `sanitize_response(text)` — applies regex phrase replacements and appends the formal medical disclaimer. |
| `data/symptoms_conditions.json` | 30 symptom records sourced from WHO/CDC. |
| `data/conditions_info.json` | 26 condition records with full care detail, sourced from WHO/CDC. |
| `data/preventive_tips.json` | 10 prevention category records sourced from WHO. |
| `vector_store/faiss_index/` | Auto-generated FAISS index directory. Created on first run. Safe to delete to trigger a rebuild. |
| `sanity_library_imports.py` | Validates core library imports. Fast import check. |
| `sanity_llama_model.py` | Calls `get_llm().invoke("What is ML.")` and prints the endpoint response. Tests the full LLM call path without Streamlit. |
| `sanity_app.py` | Minimal Streamlit page that calls `get_llm()` and lets you test the endpoint interactively via a text input and button. |
| `test_check.py` | Imports and exercises every module in sequence. Prints per-module OK/ERROR. Run before the main app for a quick full-stack validation. |
| `requirements.txt` | Pinned dependency list for reproducible installs. |

---

## 🏆 Hackathon Notes

- Single configurable **LLM endpoint** — swap model or host with one env var change
- No provider lock-in — the HTTP client is generic, not tied to any API provider SDK
- Embeddings and FAISS run fully **locally on CPU** — no external vector DB needed
- Data sourced from **WHO / CDC** public health guidance — cited in every document
- Prompt engineering enforces safety constraints at the instruction level
- Post-generation safety filter adds a second layer of phrase-level guardrails
- Keep the **disclaimer banner visible** at all times during judging

---

*Built for the Wipro × Intel Hackathon · March 2026*
