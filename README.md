# 🩺 Healthcare Symptom Information Assistant

> **Wipro × Intel Hackathon — April 2026**

A safe, trustworthy, and informative conversational AI system that helps users understand health-related symptoms through a **Retrieval-Augmented Generation (RAG)** pipeline. Built with **Streamlit**, powered by configurable LLM backends (AWS SageMaker or Llama HTTP endpoints), and grounded in a curated healthcare knowledge base.

> ⚕️ **Medical Disclaimer:** This assistant provides **general health information only** and explicitly avoids diagnosis, prescription, or personalized medical advice. Always consult a qualified healthcare professional for medical guidance. Sources: WHO, CDC.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [🏗️ Architecture](docs/ARCHITECTURE.md) | System diagrams, request data flow, RAG indexing pipeline, safety layers |
| [📋 Module Breakdown](docs/MODULES.md) | Detailed per-module documentation (UI, RAG, LLM, Safety, Core, Config) |
| [🛠️ Setup & Installation](docs/SETUP.md) | Clone, install, download embedding model, configure, run |
| [🔧 Configuration Reference](docs/CONFIGURATION.md) | All environment variables with defaults, types, and descriptions |
| [📊 Performance Monitoring](docs/MONITORING.md) | Metrics schema, field reference, dashboard panels, thread safety |
| [🧑‍💻 Development & FAQ](docs/DEVELOPMENT.md) | Customization, troubleshooting, dependencies, FAQ |

---

## 🎯 Key Features

| Feature | Detail |
|---------|--------|
| **Conversational Chat** | Multi-turn chat UI with full persistent history via Streamlit session state |
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

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/naveensiwas/wipro-intel-genai-hackathon.git
cd wipro-intel-genai-hackathon

# 2. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Download embedding model
mkdir -p assets/models
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='sentence-transformers/all-MiniLM-L6-v2',
    local_dir='assets/models/all-MiniLM-L6-v2',
    local_dir_use_symlinks=False
)
"

# 4. Run (uses SageMaker by default)
python app_launcher.py
```

> For full setup details, environment variable overrides, and Llama HTTP configuration → [Setup Guide](docs/SETUP.md)

---

## 📁 Project Structure

```
Wipro_Intel_Hackathon_LLM_EP/
├── app_launcher.py              # Root entry point — sets PYTHONPATH, runs Streamlit
├── requirements.txt             # Pinned Python dependencies
├── pyproject.toml               # Build system (setuptools); package root = src/
├── README.md                    # This file
│
├── docs/                        # 📚 Detailed documentation
│   ├── ARCHITECTURE.md          # System diagrams & data flow
│   ├── MODULES.md               # Per-module documentation
│   ├── SETUP.md                 # Installation & run instructions
│   ├── CONFIGURATION.md         # All environment variables
│   ├── MONITORING.md            # Metrics schema & dashboard
│   └── DEVELOPMENT.md           # Customization, troubleshooting & FAQ
│
├── src/
│   └── app/
│       ├── main.py              # Streamlit app entry point
│       ├── config/              # AppConfig dataclass + UIText
│       ├── core/                # Logging, metrics, error handling
│       ├── llm/                 # LLM clients, adapter, prompt templates
│       ├── rag/                 # Data loader, embedder, vector store, retriever
│       ├── safety/              # Domain gate + output sanitizer
│       └── ui/                  # Chat, sidebar, dashboard, styles
│
├── data/
│   ├── seed/                    # Healthcare knowledge base (JSON)
│   └── runtime/                 # FAISS index + metrics (auto-generated)
│
├── assets/
│   ├── images/                  # Wipro & Intel logos
│   └── models/
│       └── all-MiniLM-L6-v2/   # Local embedding model (download required)
│
└── scripts/
    └── sanity/                  # Startup verification scripts
```

---

## 📝 Attribution

**Project:** Healthcare Symptom Information Assistant  
**Event:** Wipro × Intel Hackathon — April 2026  
**Tech:** Streamlit · LangChain · FAISS · all-MiniLM-L6-v2 · Meta Llama 3.1 8B · Intel® CPU  
**Created by:** The Care Coders Team

---

*Built with ❤️ for safe, trustworthy healthcare information delivery — optimized for Intel® CPU inference.*
