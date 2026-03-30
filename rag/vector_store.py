"""
FAISS vector store management.
Builds the index on first run, then persists to disk for faster subsequent starts.
"""
import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.data_loader import load_all_documents
from rag.embedder import get_embeddings
from config import cfg
from logger_config import get_logger, log_success, log_error, log_step

logger = get_logger(__name__)


def _split_documents(docs):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
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
    """Load the FAISS index from disk if it exists, otherwise build and save it."""
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
