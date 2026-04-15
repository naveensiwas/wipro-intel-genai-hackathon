"""
Centralized runtime configuration for the Healthcare Symptom Checker application.
UI text lives in a separate module: ui_text_config.py
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
from app.core.logging import get_logger, log_section
from app.config.ui_text import UIText
logger = get_logger(__name__)


# Resolve repository root from this file location: src/app/config/settings.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class AppConfig:
    # ── LLM mode switch ───────────────────────────────────────────────────────
    # LLM backend selector:
    # - "sagemaker": AWS SageMaker endpoint via boto3
    # - "llama": direct HTTP endpoint (/v1/chat/completions style)

    # AWS SageMaker endpoint OR direct HTTP endpoint (/v1/chat/completions style)
    # llm_mode: str = os.getenv("LLM_MODE", "sagemaker")   # For local testing.
    llm_mode: str = os.getenv("LLM_MODE", "llama")   # For JUMP machine.

    # ── AWS SageMaker settings (used when llm_mode=sagemaker) ─────────────────
    aws_profile:          str   = os.getenv("AWS_PROFILE", "my-sso")
    aws_region:           str   = os.getenv("AWS_REGION", "us-east-2")
    sagemaker_endpoint:   str   = os.getenv("SAGEMAKER_ENDPOINT","intel-llama3-1-8b-cpu-tgi-endpoint")

    # ── Llama HTTP endpoint settings (used when llm_mode=llama) ───────────────
    llm_base_url:         str   = os.getenv("LLM_BASE_URL", "http://wiphackdw23f36.cloudloka.com:8000/v1")
    llm_api_key:          str   = os.getenv("LLM_API_KEY", "")

    # ── Shared LLM generation settings ────────────────────────────────────────
    llm_model:            str   = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    llm_timeout:          int   = int(os.getenv("LLM_TIMEOUT", "120"))  # seconds
    temperature:          float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens:           int   = int(os.getenv("LLM_MAX_TOKENS", "512"))
    verbose:              bool  = os.getenv("LLM_VERBOSE", "false").lower() == "true"

    # ── Embeddings ─────────────────────────────────────────────────────────
    # embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2" # Original HuggingFace model (CPU-friendly)
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        str(_REPO_ROOT / "assets" / "models" / "all-MiniLM-L6-v2"),
    )

    # ── RAG / chunking ─────────────────────────────────────────────────────
    chunk_size: int = 400
    chunk_overlap: int = 60
    retriever_k: int = 4

    # ── Vector store ───────────────────────────────────────────────────────
    vector_store_dir: str = os.getenv(
        "VECTOR_STORE_DIR",
        str(_REPO_ROOT / "data" / "runtime" / "vector_store" / "faiss_index"),
    )

    # ── Metrics path ───────────────────────────────────────────────────────
    metrics_file_path: str = os.getenv(
        "METRICS_FILE_PATH",
        str(_REPO_ROOT / "data" / "runtime" / "metrics" / "performance_metrics.json"),
    )

    # ── Data paths ─────────────────────────────────────────────────────────
    symptoms_conditions_path: str = os.getenv(
        "SYMPTOMS_CONDITIONS_PATH",
        str(_REPO_ROOT / "data" / "seed" / "symptoms_conditions.json"),
    )
    conditions_info_path: str = os.getenv(
        "CONDITIONS_INFO_PATH",
        str(_REPO_ROOT / "data" / "seed" / "conditions_info.json"),
    )
    preventive_tips_path: str = os.getenv(
        "PREVENTIVE_TIPS_PATH",
        str(_REPO_ROOT / "data" / "seed" / "preventive_tips.json"),
    )

    # ── UI behavior ─────────────────────────────────────────────────────────
    show_retrieved_sources: bool = os.getenv("SHOW_RETRIEVED_SOURCES", "true").lower() == "true" # Toggle for displaying retrieved source documents in the UI.
    chat_view = "chat"
    metrics_view = "metrics"

    # ── Domain gate behavior ────────────────────────────────────────────────
    domain_filter_mode: str = os.getenv("DOMAIN_FILTER_MODE", "hybrid")
    domain_similarity_threshold: float = float(os.getenv("DOMAIN_SIMILARITY_THRESHOLD", "0.40"))
    domain_similarity_k: int = int(os.getenv("DOMAIN_SIMILARITY_K", "1"))
    domain_use_keyword_fallback: bool = os.getenv("DOMAIN_USE_KEYWORD_FALLBACK", "true").lower() == "true"

    # ── UI text ─────────────────────────────────────────────────────────────
    ui_text: UIText = field(default_factory=UIText)


# Shared config instance across modules
cfg = AppConfig()

# ── Configuration diagnostic info (for debugging) ──────────────────────────────
# Check if environment variables are overriding defaults
_env_overrides = {
    "LLM_MODE": os.getenv("LLM_MODE"),
    "LLM_MAX_TOKENS": os.getenv("LLM_MAX_TOKENS"),
    "LLM_TIMEOUT": os.getenv("LLM_TIMEOUT"),
    "LLM_TEMPERATURE": os.getenv("LLM_TEMPERATURE"),
    "DOMAIN_FILTER_MODE": os.getenv("DOMAIN_FILTER_MODE"),
    "DOMAIN_SIMILARITY_THRESHOLD": os.getenv("DOMAIN_SIMILARITY_THRESHOLD"),
    "DOMAIN_SIMILARITY_K": os.getenv("DOMAIN_SIMILARITY_K"),
    "DOMAIN_USE_KEYWORD_FALLBACK": os.getenv("DOMAIN_USE_KEYWORD_FALLBACK"),
}
_active_overrides = {k: v for k, v in _env_overrides.items() if v is not None}

if _active_overrides:
    log_section(logger, "Config: Environment variable overrides detected")
    for key, val in _active_overrides.items():
        logger.info(f"• {key}={val}")
