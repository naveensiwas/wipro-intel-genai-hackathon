"""
Embedding model setup using sentence-transformers (CPU-only).

Provides a cached HuggingFaceEmbeddings instance for use with LangChain + FAISS.
The embedding model is configured via EMBEDDING_MODEL environment variable
and can be either a local model path or a HuggingFace model ID.

The cache ensures only one embeddings instance is created per process,
avoiding redundant model loads across multiple Streamlit reruns.
"""
from functools import lru_cache
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.settings import cfg
from app.core.logging import get_logger, log_success, log_error

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.

    Loads the embedding model specified in EMBEDDING_MODEL config:
    - If it's a local path, loads from disk
    - If it's a model ID, downloads from HuggingFace Hub
    - All embeddings are computed on CPU (device="cpu")
    - Output embeddings are L2-normalized for stable cosine similarity

    Returns:
        HuggingFaceEmbeddings: Cached embeddings instance compatible with FAISS

    Raises:
        FileNotFoundError: If EMBEDDING_MODEL is a local path that doesn't exist
        Exception: If model download/loading fails (network, corrupt model, etc.)

    Note:
        Result is cached via @lru_cache(maxsize=1), so repeated calls return
        the same instance. This is critical for performance since model loading
        is expensive (CPU time, memory allocation).

    Example:
        embeddings = get_embeddings()
        vector = embeddings.embed_query("What is a fever?")  # 384-dim vector
    """
    model_path = Path(cfg.embedding_model)
    logger.info(f"Loading embedding model: {model_path}")
    try:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model path not found: {model_path}. "
                "Set EMBEDDING_MODEL to a valid local path or HuggingFace model id."
            )

        embeddings = HuggingFaceEmbeddings(
            model_name=str(model_path),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        log_success(logger, f"Embedding model loaded: {model_path}")
        return embeddings
    except Exception as exc:
        log_error(logger, f"Failed to load embedding model '{cfg.embedding_model}'", exc)
        raise
