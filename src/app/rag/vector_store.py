"""
FAISS vector store management.
Builds the index on first run, then persists to disk for faster subsequent starts.
"""
import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.data_loader import load_all_documents
from app.rag.embedder import get_embeddings
from app.config.settings import cfg
from app.core.logging import get_logger, log_success, log_error, log_step

logger = get_logger(__name__)


def _split_documents(docs):
    """
    Split documents into semantic chunks using recursive character splitting.

    Splits on natural boundaries (paragraphs, sentences, words) to preserve
    semantic coherence. Chunks may overlap to maintain context across boundaries.

    Args:
        docs: List of LangChain Document objects from data loader

    Returns:
        list[Document]: Split documents with preserved metadata, ready for embedding

    Note:
        Chunk size and overlap are configured via CHUNK_SIZE and CHUNK_OVERLAP.
        First few chunks are logged at debug level for inspection.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.debug(f"Split {len(docs)} documents into {len(chunks)} chunks "
                 f"(size={cfg.chunk_size}, overlap={cfg.chunk_overlap})")

    # Log details of the first few chunks for debugging
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        preview = chunk.page_content.replace("\n", " ").strip()[:200]
        logger.debug(
            f"Chunk {idx}/{len(chunks)} | source={source} | "
            f"length={len(chunk.page_content)} | preview={preview}"
        )

    return chunks


def build_or_load_vector_store() -> FAISS:
    """
    Load an existing FAISS index from disk, or build a new one if it doesn't exist.

    Initialization Flow:
    1. Check if VECTOR_STORE_DIR contains a valid FAISS index (index.faiss + index.pkl)
    2. If found, load it with embeddings → return (fast path, ~seconds)
    3. If not found, load all health documents → chunk them → embed → build FAISS index
       → save to disk → return (slow path, ~minutes first time)

    This pattern ensures:
    - First app startup is slower but builds the index once
    - All subsequent startups are fast (loaded from disk)
    - Index can be rebuilt by deleting VECTOR_STORE_DIR

    Returns:
        FAISS: Vector store ready for semantic search and RAG

    Raises:
        Exception: If document loading fails, or FAISS operations fail

    Note:
        FAISS index is built with cosine similarity metric (via normalized embeddings).
        Index persists in VECTOR_STORE_DIR (default: data/runtime/vector_store/faiss_index).
    """
    embeddings = get_embeddings()
    index_dir  = cfg.vector_store_dir
    index_file = os.path.join(index_dir, "index.faiss")

    if os.path.exists(index_dir) and os.path.exists(index_file):
        log_step(logger, 1, f"Loading existing FAISS index from '{index_dir}'")
        try:
            vector_store = FAISS.load_local(
                index_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            log_success(logger, f"FAISS index loaded from '{index_dir}'")
            return vector_store
        except Exception as exc:
            log_error(logger, f"Failed to load FAISS index from '{index_dir}'", exc)
            raise

    # Build a fresh index
    log_step(logger, 1, "No existing FAISS index found — building new index")
    try:
        docs   = load_all_documents()
        chunks = _split_documents(docs)
        logger.info(f"Indexing {len(chunks)} chunks...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        os.makedirs(index_dir, exist_ok=True)
        vector_store.save_local(index_dir)
        log_success(logger, f"FAISS index built and saved to '{index_dir}' ({len(chunks)} chunks)")
        return vector_store
    except Exception as exc:
        log_error(logger, "Failed to build FAISS index", exc)
        raise
