# 🧑‍💻 Development, Troubleshooting & FAQ

> [← Back to README](../README.md)

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

Add new fields in `src/app/core/metrics.py` → `build_metrics_entry()`, then render them in `src/app/ui/metrics_dashboard.py`.

### Known Limitation: Multi-Turn Context

Chat history is stored in the UI via Streamlit session state and is re-rendered each turn, but **prior conversation turns are not included in the LLM prompt**. Each query is answered independently from retrieved context only. Follow-up questions that rely on prior answers (e.g., "what about the second symptom I mentioned?") may not be answered correctly unless the user restates the context.

---

## 🔐 Safety & Ethical Design

### Responsible AI Principles

- **No Diagnosis**: The system explicitly cannot and will not diagnose conditions
- **No Prescription**: Medication recommendations are blocked at prompt and output layers
- **Grounded Only**: Responses cite only information present in the retrieved knowledge base
- **Transparent Sources**: Every response optionally shows which knowledge documents were used
- **Urgent Escalation**: Severe symptom descriptions trigger natural-language urgency escalation — without clinical labels
- **Educational Framing**: All information positioned as general health education, not personalized advice

---

## 🚨 Troubleshooting

### FAISS Index Issues

```bash
# Force full rebuild
rm -rf data/runtime/vector_store/faiss_index/
python app_launcher.py
```

### Embedding Model Not Found

```bash
# Re-download model
mkdir -p assets/models
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='sentence-transformers/all-MiniLM-L6-v2',
    local_dir='assets/models/all-MiniLM-L6-v2',
    local_dir_use_symlinks=False
)
"
```

### LLM Endpoint Timeout

```bash
# Test Llama HTTP endpoint
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

### AWS SSO Authentication Failure

```bash
aws sso login --profile my-sso
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

# Verify embedding model loads correctly
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
| `boto3` / `botocore` | 1.42.73 | AWS SageMaker client |
| `pydantic` | 2.12.5 | Config validation (AppConfig, EndpointLLM) |
| `pydantic-settings` | 2.13.1 | Settings management |
| `pandas` | 2.3.3 | Metrics DataFrame processing |
| `requests` | 2.32.5 | Llama HTTP endpoint client |
| `numpy` | 1.26.4 | Numerical operations (FAISS) |
| `safetensors` | 0.7.0 | Model weight loading |
| `huggingface_hub` | 1.7.2 | Model download utility |
| `python-dotenv` | 1.2.2 | Included in deps (`.env` support not yet wired) |

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
A: SageMaker uses AWS for LLM inference via boto3 + a named SSO profile (default for local dev). Llama mode POSTs to an OpenAI-compatible HTTP endpoint (for server deployments).

**Q: Why does the first startup take 1-2 minutes?**  
A: On first run, there is no FAISS index on disk. The app loads all knowledge base documents, splits them into ~400-token chunks, embeds them, and builds the FAISS index. Subsequent startups load the cached index in under 1 second.

**Q: How is TTFT measured for non-streaming responses?**  
A: Since `RetrievalQA.invoke()` is non-streaming, TTFT is approximated as the generation completion timestamp. True TTFT would require streaming token callbacks.

**Q: Can I use a different embedding model?**  
A: Yes — set `EMBEDDING_MODEL` to any HuggingFace model ID or local path. Delete the FAISS index directory afterwards to force a rebuild.

**Q: How do I update the knowledge base?**  
A: Edit or add JSON files in `data/seed/`, register new loaders in `data_loader.py`, then delete the FAISS index directory to trigger a rebuild.

**Q: Is the system thread-safe?**  
A: Yes. Metrics JSON writes are protected by `threading.Lock`. Streamlit isolates session state per browser session. LLM and vector store are initialized once via `@st.cache_resource` and `@lru_cache`.

**Q: What happens when a greeting like "Hi" is sent?**  
A: Greetings are in the keyword allowlist, so they pass the domain gate. The LLM prompt instructs it to respond warmly and invite a health question.

**Q: What does the Health Status indicator mean?**  
A: 🟢 Healthy = ≥80% requests without errors; 🟡 Degraded = 60–79%; 🔴 Unhealthy = <60%.

**Q: Why are token counts approximate?**  
A: Token counts use whitespace splitting (`len(text.split())`) for performance. Accuracy is ~85-90% vs. sub-word tokenization.

**Q: Does the chat remember previous questions?**  
A: The chat UI displays history (session state), but the LLM prompt only includes the current query and retrieved documents — not prior conversation turns. See the Known Limitation note above.

---

## 🤝 Support

1. Check the [Troubleshooting](#-troubleshooting) section first
2. Review terminal logs (color-coded, millisecond-precision)
3. Check the Metrics Dashboard for request-level insights
4. Run sanity scripts in `scripts/sanity/` to isolate component failures
5. Contact the development team

---

*[← Back to README](../README.md)*
