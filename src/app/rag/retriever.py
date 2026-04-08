"""
RAG chain assembly and document retrieval.

Provides functions to:
1. Build a LangChain RetrievalQA chain (combining LLM + vector store + prompt)
2. Retrieve top-k source documents for a query (for UI display)
3. Compute semantic similarity scores for domain gating

All retrieval operations use the same underlying FAISS vector store.
"""
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from app.llm.prompt_templates import RAG_PROMPT
from app.config.settings import cfg
from app.core.logging import get_logger, log_success, log_error

logger = get_logger(__name__)


def _get_retriever(vector_store: FAISS):
    """
    Internal helper to instantiate a FAISS retriever.

    Consolidates the retriever configuration (search type, k parameter) in one place
    to reduce duplication between build_rag_chain() and retrieve_sources().

    Args:
        vector_store: FAISS vector store instance

    Returns:
        LangChain Retriever configured for similarity search with cfg.retriever_k
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": cfg.retriever_k},
    )


def build_rag_chain(llm, vector_store: FAISS) -> RetrievalQA:
    """
    Assemble a RAG (Retrieval-Augmented Generation) chain.

    Wires together:
    1. A vector store retriever (semantic search via FAISS)
    2. An LLM for grounded text generation
    3. A custom healthcare prompt template with safety rules

    The chain follows the "stuff" prompt pattern: all retrieved documents
    are concatenated into the prompt (suitable for small context windows).

    Args:
        llm: LangChain-compatible LLM instance (EndpointLLM wrapper)
        vector_store: FAISS vector store containing indexed health documents

    Returns:
        RetrievalQA: Executable chain with invoke({"query": str}) interface.
                    Returns dict with "result" (generated text) and
                    "source_documents" (List[Document])

    Raises:
        Exception: If chain assembly fails (invalid LLM, missing vector store, etc.)
    """
    logger.info(f"Building RAG chain (retriever_k={cfg.retriever_k}, chain_type='stuff')")
    try:
        retriever = _get_retriever(vector_store)
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": RAG_PROMPT},
        )
        log_success(logger, "RAG chain assembled")
        return chain
    except Exception as exc:
        log_error(logger, "Failed to build RAG chain", exc)
        raise


def retrieve_sources(vector_store: FAISS, query: str) -> list[dict]:
    """
    Retrieve top-k source documents relevant to a query for UI display.

    Performs semantic search and returns formatted document chunks suitable
    for rendering in the Streamlit UI (e.g., in expanders with metadata labels).

    Args:
        vector_store: FAISS vector store instance
        query: User query string (typically already validated as health-related)

    Returns:
        list[dict]: Each dict contains:
            - "content" (str): Truncated document text (max 300 chars) with ellipsis
            - "metadata" (dict): Document metadata (source_file, condition, symptom, etc.)

    Note:
        Retrieved documents are pre-truncated to improve UI rendering.
        Full documents are still available in the RAG chain output.
    """
    logger.debug(f"Retrieving top-{cfg.retriever_k} sources for query: '{query[:80]}...'")
    retriever = _get_retriever(vector_store)
    docs = retriever.invoke(query)
    results = []
    for doc in docs:
        results.append({
            "content":  doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "metadata": doc.metadata,
        })
    logger.debug(f"Retrieved {len(results)} source chunks")
    return results


def get_top_similarity_score(vector_store: FAISS, query: str, k: int = 1) -> float:
    """
    Compute a bounded semantic similarity score [0.0, 1.0] for domain gating.

    Retrieves the top-k documents and returns the maximum similarity score.
    Handles both raw similarity and distance-based FAISS scoring by normalizing
    to [0.0, 1.0] range for consistent threshold comparisons.

    Score Normalization Logic:
    - If scores are in [0.0, 1.0], they are already similarities → use as-is
    - If scores are > 1.0, they are distances → transform: similarity = 1 / (1 + distance)
    - Final score is always clamped to [0.0, 1.0]

    Args:
        vector_store: FAISS vector store instance
        query: Query string to score
        k: Number of nearest neighbors to consider (default: 1 = top match only)

    Returns:
        float: Normalized similarity score in [0.0, 1.0]
               0.0 = completely dissimilar from health domain
               1.0 = highly similar to health domain

    Note:
        Returns 0.0 if FAISS returns no hits (empty index or query fails).
        Uses raw FAISS scores to avoid relevance-score warnings when
        backend normalization differs by distance metric.
    """
    k = max(1, int(k))
    docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
    if not docs_and_scores:
        logger.debug("Domain gate: no FAISS hits returned for query")
        return 0.0

    raw_scores = [float(score) for _, score in docs_and_scores]

    # If scores are already similarity-like, keep them as-is.
    if all(0.0 <= s <= 1.0 for s in raw_scores):
        top_score = max(raw_scores)
    else:
        # Treat scores as distances and map to (0, 1] conservatively.
        similarities = [1.0 / (1.0 + max(0.0, s)) for s in raw_scores]
        top_score = max(similarities)

    top_score = max(0.0, min(1.0, float(top_score)))
    logger.debug(f"Domain gate: top semantic score={top_score:.4f} (k={k})")
    return top_score
