"""
Centralized runtime configuration for the Healthcare Symptom Checker application.
UI text lives in a separate module: ui_text_config.py
"""
from dataclasses import dataclass, field
import os
from logger_config import get_logger, log_section
from ui_text_config import UIText
logger = get_logger(__name__)


@dataclass
class AppConfig:
    # ── LLM mode switch ───────────────────────────────────────────────────────
    # LLM backend selector:
    # - "sagemaker": AWS SageMaker endpoint via boto3
    # - "llama": direct HTTP endpoint (/v1/chat/completions style)

    # AWS SageMaker endpoint OR direct HTTP endpoint (/v1/chat/completions style)
    #llm_mode: str = os.getenv("LLM_MODE", "sagemaker")   # For local testing.
    llm_mode: str = os.getenv("LLM_MODE", "llama")   # For JUMP machine.

    # ── AWS SageMaker settings (used when llm_mode=sagemaker) ─────────────────
    aws_profile:          str   = os.getenv("AWS_PROFILE", "my-sso")
    aws_region:           str   = os.getenv("AWS_REGION", "us-east-2")
    sagemaker_endpoint:   str   = os.getenv("SAGEMAKER_ENDPOINT","llama-3-2-3b-tgi-cpu-endpoint")

    # ── Llama HTTP endpoint settings (used when llm_mode=llama) ───────────────
    llm_base_url:         str   = os.getenv("LLM_BASE_URL", "http://wiphackq0vcsii.cloudloka.com:8000/v1")
    llm_api_key:          str   = os.getenv("LLM_API_KEY", "")

    # ── Shared LLM generation settings ────────────────────────────────────────
    llm_model:            str   = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    llm_timeout:          int   = int(os.getenv("LLM_TIMEOUT", "120"))  # seconds
    temperature:          float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens:           int   = int(os.getenv("LLM_MAX_TOKENS", "512"))
    verbose:              bool  = os.getenv("LLM_VERBOSE", "false").lower() == "true"

    # ── Embeddings ─────────────────────────────────────────────────────────
    # embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2" # Original HuggingFace model (CPU-friendly)
    embedding_model: str = "models/all-MiniLM-L6-v2" # Local path to the same model (for faster loading)

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

    # ── UI behavior ─────────────────────────────────────────────────────────
    show_retrieved_sources: bool = os.getenv("SHOW_RETRIEVED_SOURCES", "false").lower() == "true" # Toggle for displaying retrieved source documents in the UI.

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
}
_active_overrides = {k: v for k, v in _env_overrides.items() if v is not None}

if _active_overrides:
    log_section(logger, "Config: Environment variable overrides detected")
    for key, val in _active_overrides.items():
        logger.info(f"• {key}={val}")
