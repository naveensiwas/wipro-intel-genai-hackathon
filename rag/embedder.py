"""
Embedding model setup using sentence-transformers (CPU-only).
Uses HuggingFaceEmbeddings for compatibility with LangChain + FAISS.
"""
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from config import cfg
from logger_config import get_logger, log_success, log_error

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance."""
    logger.info(f"Loading embedding model: {cfg.embedding_model}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        log_success(logger, f"Embedding model loaded: {cfg.embedding_model}")
        return embeddings
    except Exception as exc:
        log_error(logger, f"Failed to load embedding model '{cfg.embedding_model}'", exc)
        raise
