# 🛠️ Local Setup & Installation

> [← Back to README](../README.md)

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| RAM | ~4 GB (embeddings + FAISS) |
| Disk | ~2 GB (model weights + index) |
| OS | macOS / Linux / Windows |
| GPU | Not required (CPU-only) |

---

## Step 1: Clone the Repository

```bash
# HTTPS
git clone https://github.com/naveensiwas/wipro-intel-genai-hackathon.git
cd wipro-intel-genai-hackathon

# Or SSH (if you have an SSH key configured)
git clone git@github.com:naveensiwas/wipro-intel-genai-hackathon.git
cd wipro-intel-genai-hackathon
```

---

## Step 2: Create Virtual Environment & Install Dependencies

```bash
# Create a Python virtual environment
python3 -m venv venv

# Activate — macOS/Linux
source venv/bin/activate

# Activate — Windows
# venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Step 3: Download the Embedding Model Locally (Required)

The app uses the `all-MiniLM-L6-v2` sentence-transformer model for all embeddings.
The model must be present at `assets/models/all-MiniLM-L6-v2` before the app starts.

```bash
# Create the models directory
mkdir -p assets/models

# Download model from HuggingFace Hub into local path
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='sentence-transformers/all-MiniLM-L6-v2',
    local_dir='assets/models/all-MiniLM-L6-v2',
    local_dir_use_symlinks=False
)
"
```

Verify the download:

```bash
ls -la assets/models/all-MiniLM-L6-v2
# Expected files:
# config.json
# model.safetensors
# tokenizer.json
# tokenizer_config.json
# modules.json
# sentence_bert_config.json
# 1_Pooling/config.json
```

> **Why local?** Loading from disk avoids HuggingFace Hub network calls at startup and ensures the app works in air-gapped / offline environments.

---

## Step 4: Configure Environment (Optional)

All settings have sensible defaults defined in `src/app/config/settings.py`.  

> ⚠️ **Current Limitation:** `.env` file support is **not yet implemented**. Override settings using shell `export` commands or inline variable assignment only.

### Minimum recommended override

Always set the embedding model path explicitly to avoid ambiguity:

```bash
export EMBEDDING_MODEL=./assets/models/all-MiniLM-L6-v2
```

### Option A: AWS SageMaker backend (default)

```bash
export LLM_MODE=sagemaker
export AWS_PROFILE=my-sso
export AWS_REGION=us-east-2
export SAGEMAKER_ENDPOINT=intel-llama3-1-8b-cpu-tgi-endpoint
export LLM_TEMPERATURE=0.2
export LLM_MAX_TOKENS=512
export LLM_TIMEOUT=120
export EMBEDDING_MODEL=./assets/models/all-MiniLM-L6-v2

python app_launcher.py
```

### Option B: Llama HTTP endpoint (server deployment)

```bash
export LLM_MODE=llama
export LLM_BASE_URL=http://your-llama-server.com:8000/v1
export LLM_API_KEY=your-api-key-if-needed
export LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
export EMBEDDING_MODEL=./assets/models/all-MiniLM-L6-v2
export DOMAIN_FILTER_MODE=hybrid
export SHOW_RETRIEVED_SOURCES=true

python app_launcher.py
```

### Inline one-liner (no export)

```bash
EMBEDDING_MODEL=./assets/models/all-MiniLM-L6-v2 \
LLM_MODE=llama \
LLM_BASE_URL=http://your-server:8000/v1 \
python app_launcher.py
```

For all available environment variables → see [CONFIGURATION.md](CONFIGURATION.md)

---

## Step 5: Verify Data Files

```bash
ls -la data/seed/
# symptoms_conditions.json
# conditions_info.json
# preventive_tips.json
```

---

## Step 6: Run the App

```bash
# Recommended: root launcher (automatically sets PYTHONPATH)
python app_launcher.py

# Alternative: direct Streamlit
streamlit run src/app/main.py

# With inline env vars
EMBEDDING_MODEL=./assets/models/all-MiniLM-L6-v2 \
LLM_MODE=llama \
LLM_BASE_URL=http://your-server:8000/v1 \
python app_launcher.py
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

# Full end-to-end test of the Streamlit app with a sample query
streamlit run scripts/sanity/sanity_app.py
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
4. Click **📋 Prepare Symptom Details** — composes a structured query e.g.:  
   *"My primary symptom is Fever, for 1–3 days, with a severity of 7/10. What could this be and what should I do?"*
5. Query auto-submits on next chat render

---

*[← Back to README](../README.md)*
