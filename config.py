"""
Centralised configuration for the Healthcare Symptom Checker application.
"""
from dataclasses import dataclass
import os


@dataclass
class AppConfig:
    # ── LLM mode switch ───────────────────────────────────────────────────────
    # Set LLM_MODE=sagemaker  → uses AWS SageMaker endpoint via boto3 + SSO
    # Set LLM_MODE=llama      → uses remote Llama HTTP endpoint directly
    llm_mode: str = os.getenv("LLM_MODE", "sagemaker")   # "sagemaker" | "llama"

    # ── AWS SageMaker settings (used when llm_mode=sagemaker) ─────────────────
    aws_profile:          str   = os.getenv("AWS_PROFILE",       "my-sso")
    aws_region:           str   = os.getenv("AWS_REGION",        "us-east-2")
    sagemaker_endpoint:   str   = os.getenv("SAGEMAKER_ENDPOINT","llama-3-2-3b-tgi-cpu-endpoint")

    # ── Llama HTTP endpoint settings (used when llm_mode=llama) ───────────────
    llm_base_url:         str   = os.getenv("LLM_BASE_URL",      "http://wiphackq0vcsii.cloudloka.com:8000/v1")
    llm_api_key:          str   = os.getenv("LLM_API_KEY",       "")

    # ── Shared LLM generation settings ────────────────────────────────────────
    llm_model:            str   = os.getenv("LLM_MODEL",                "llama-3-2-3b-tgi-cpu-endpoint")
    llm_timeout:          int   = int(os.getenv("LLM_TIMEOUT",          "60"))
    temperature:          float = float(os.getenv("LLM_TEMPERATURE",    "0.3"))
    max_tokens:           int   = int(os.getenv("LLM_MAX_TOKENS",       "512"))
    verbose:              bool  = os.getenv("LLM_VERBOSE", "false").lower() == "true"

    # ── Embeddings ─────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── RAG / chunking ─────────────────────────────────────────────────────
    chunk_size: int = 400
    chunk_overlap: int = 60
    retriever_k: int = 4

    # ── Vector store ───────────────────────────────────────────────────────
    vector_store_dir: str = "vector_store/faiss_index"

    # ── Data paths ─────────────────────────────────────────────────────────
    symptoms_conditions_path: str = "data/symptoms_conditions.json"
    conditions_info_path:     str = "data/conditions_info.json"
    preventive_tips_path:     str = "data/preventive_tips.json"

    # ── UI ─────────────────────────────────────────────────────────────────
    app_title: str = "🩺 Healthcare Symptom Information Assistant"
    disclaimer: str = (
        "⚠️ **Disclaimer:** This tool provides general health information only. "
        "It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider for medical concerns."
    )


# Singleton instance used across all modules
cfg = AppConfig()

# ── Configuration diagnostic info (for debugging) ──────────────────────────────
# Check if environment variables are overriding defaults
_env_overrides = {
    "LLM_MODE": os.getenv("LLM_MODE"),
    "LLM_MAX_TOKENS": os.getenv("LLM_MAX_TOKENS"),
    "LLM_TIMEOUT": os.getenv("LLM_TIMEOUT"),
    "LLM_TEMPERATURE": os.getenv("LLM_TEMPERATURE"),
}
_active_overrides = {k: v for k, v in _env_overrides.items() if v is not None}

if _active_overrides:
    import sys
    print(f"⚙️  Config: Environment variable overrides detected:", file=sys.stderr)
    for key, val in _active_overrides.items():
        print(f"   • {key}={val}", file=sys.stderr)

