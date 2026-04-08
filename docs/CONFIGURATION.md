# 🔧 Configuration Reference

> [← Back to README](../README.md)

---

All settings live in `src/app/config/settings.py` as the `AppConfig` dataclass.  
Environment-configurable fields are resolved at import time via `os.getenv()` with typed defaults.

> ⚠️ **Note:** `.env` file support is **not yet implemented**. Use shell `export` commands or inline variable assignment to override defaults.

---

## LLM Backend

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `LLM_MODE` | `sagemaker` | str | LLM backend: `sagemaker` or `llama` |
| `LLM_TEMPERATURE` | `0.2` | float | Response creativity (0.0–1.0) |
| `LLM_MAX_TOKENS` | `512` | int | Max response tokens |
| `LLM_TIMEOUT` | `120` | int | Request timeout (seconds) |
| `LLM_VERBOSE` | `false` | bool | Log raw request/response payloads |

## AWS SageMaker (used when `LLM_MODE=sagemaker`)

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `AWS_PROFILE` | `my-sso` | str | Named boto3 SSO profile |
| `AWS_REGION` | `us-east-2` | str | AWS region for SageMaker |
| `SAGEMAKER_ENDPOINT` | `intel-llama3-1-8b-cpu-tgi-endpoint` | str | SageMaker endpoint name |

## Llama HTTP Endpoint (used when `LLM_MODE=llama`)

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `LLM_BASE_URL` | *(internal server)* | str | Base URL for Llama HTTP endpoint |
| `LLM_API_KEY` | `""` | str | Bearer token (optional, skipped if empty or `"dummy"`) |
| `LLM_MODEL` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | str | Model identifier sent in payload |

## Embeddings

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `EMBEDDING_MODEL` | `./assets/models/all-MiniLM-L6-v2` | str | Local path or HuggingFace model ID |

> See [SETUP.md](SETUP.md) for instructions on downloading the model locally.

## RAG / Chunking

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `CHUNK_SIZE` | `400` | int | Document chunk size (tokens) |
| `CHUNK_OVERLAP` | `60` | int | Overlap between adjacent chunks |
| `RETRIEVER_K` | `4` | int | Number of documents retrieved per query |

## Vector Store

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `VECTOR_STORE_DIR` | `./data/runtime/vector_store/faiss_index` | str | FAISS index directory |

## Data Paths

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `SYMPTOMS_CONDITIONS_PATH` | `./data/seed/symptoms_conditions.json` | str | Symptoms knowledge base |
| `CONDITIONS_INFO_PATH` | `./data/seed/conditions_info.json` | str | Conditions knowledge base |
| `PREVENTIVE_TIPS_PATH` | `./data/seed/preventive_tips.json` | str | Preventive tips knowledge base |

## Metrics

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `METRICS_FILE_PATH` | `./data/runtime/metrics/performance_metrics.json` | str | Metrics JSON storage path |

## Domain Gate (Safety)

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `DOMAIN_FILTER_MODE` | `hybrid` | str | Domain gate strategy: `keyword`, `semantic`, or `hybrid` |
| `DOMAIN_SIMILARITY_THRESHOLD` | `0.40` | float | Minimum semantic similarity score to accept (0.0–1.0) |
| `DOMAIN_SIMILARITY_K` | `1` | int | Number of docs retrieved for similarity scoring |
| `DOMAIN_USE_KEYWORD_FALLBACK` | `true` | bool | Allow keyword match to pass when semantic score fails |

## UI Behaviour

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `SHOW_RETRIEVED_SOURCES` | `true` | bool | Display retrieved source documents in chat UI |

---

## Example: Override for Llama HTTP Deployment

```bash
export LLM_MODE=llama
export LLM_BASE_URL=http://your-server:8000/v1
export LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
export EMBEDDING_MODEL=./assets/models/all-MiniLM-L6-v2
export DOMAIN_FILTER_MODE=hybrid
export DOMAIN_SIMILARITY_THRESHOLD=0.40
export SHOW_RETRIEVED_SOURCES=true

python app_launcher.py
```

## Example: Stricter vs More Permissive Domain Gate

```bash
# Stricter — fewer false accepts
export DOMAIN_SIMILARITY_THRESHOLD=0.55
export DOMAIN_FILTER_MODE=semantic

# More permissive — fewer false rejects
export DOMAIN_SIMILARITY_THRESHOLD=0.30
export DOMAIN_FILTER_MODE=hybrid
```

---

*[← Back to README](../README.md)*
