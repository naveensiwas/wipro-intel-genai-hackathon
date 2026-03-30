"""
RAG chain assembly.
Wires the retriever and LLM into a RetrievalQA chain with a custom healthcare prompt.
"""
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from llm.prompt_templates import RAG_PROMPT
from config import cfg
from logger_config import get_logger, log_success, log_error

logger = get_logger(__name__)


def build_rag_chain(llm, vector_store: FAISS) -> RetrievalQA:
    """Build and return a RetrievalQA chain."""
    logger.info(f"Building RAG chain (retriever_k={cfg.retriever_k}, chain_type='stuff')")
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": cfg.retriever_k},
        )
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
    """Return the top-k retrieved source chunks for a query (for UI display)."""
    logger.debug(f"Retrieving top-{cfg.retriever_k} sources for query: '{query[:80]}...'")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": cfg.retriever_k},
    )
    docs = retriever.invoke(query)
    results = []
    for doc in docs:
        results.append({
            "content":  doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "metadata": doc.metadata,
        })
    logger.debug(f"Retrieved {len(results)} source chunks")
    return results
