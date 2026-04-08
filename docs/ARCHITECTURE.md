# 🏗️ Architecture

> [← Back to README](../README.md)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Web Interface                           │
│  ┌───────────────┐  ┌─────────────────────────┐  ┌────────────────────┐  │
│  │  Chat Window  │  │         Sidebar         │  │ Metrics Dashboard  │  │
│  │  • Messages   │  │  • Navigation (2 views) │  │  • Health Status   │  │
│  │  • Sources    │  │  • Smart Symptom Guide  │  │  • Latency Panels  │  │
│  │  • Avatars    │  │  • Wipro / Intel Logos  │  │  • Request Table   │  │
│  └───────────────┘  │  • About / Clear Chat   │  │  • CSV Export      │  │
│                     └─────────────────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                              ▲ st.session_state ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        Main Application Logic (main.py)                  │
│  • @st.cache_resource — LLM, vector store, RAG chain loaded once         │
│  • Session state: messages, active_requests, current_view                │
│  • Prefilled query support from sidebar structured input                 │
│  • Per-request timing accumulators → metrics entry on completion         │
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

---

## Request Data Flow

```
User types message (or uses Smart Symptom Guide to compose query)
    │
    ▼
┌────────────────────────────────────┐
│  Domain Gate: is_health_related_   │
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
┌────────────────────────────┐
│  rag_chain.invoke()        │
│  RetrievalQA (stuff)       │
│  → RAG_PROMPT filled       │
│    with context + question │
└────────────────────────────┘
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
│  • Appends ⚕️ disclaimer  │
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

## RAG Pipeline — Chunking & Indexing Flow

```
load_all_documents()            →  List[Document]  (raw, full-text)
        │
        ▼
RecursiveCharacterTextSplitter  →  List[Document]  (chunked, metadata preserved)
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

## RAG Pipeline — Retrieval & Generation

```python
# On each user query:
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

chain = RetrievalQA.from_chain_type(
    llm=EndpointLLM(...),
    chain_type="stuff",           # All retrieved docs concatenated into one prompt
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": RAG_PROMPT},
)

result = chain.invoke({"query": user_input})
# result["result"]           → generated text
# result["source_documents"] → List[Document] (used for source attribution)
```

---

## Three-Layer Safety Architecture

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

---

*[← Back to README](../README.md)*
