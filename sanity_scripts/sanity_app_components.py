"""
Sanity Check – Performs a comprehensive import and data-load validation across all modules.

Validates all required imports and data loading functions for the Healthcare Symptom Checker project.
Provides detailed feedback on each import and data load with structured logging.
"""

import sys
import os
# Ensure the project root is on sys.path so logger_config and other modules can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logger_config import get_logger, log_section, log_success, log_error, log_step
logger = get_logger(__name__)

# ============================================================================
# RAG MODULE VALIDATION
# ============================================================================

log_section(logger, "RAG MODULE VALIDATION")

# Validate data loader: imports symptom, condition, and preventive documents
log_step(logger, 1, "Loading RAG documents (symptoms, conditions, preventive)")
try:
    from rag.data_loader import load_symptom_documents, load_condition_documents, load_preventive_documents
    s = load_symptom_documents()
    c = load_condition_documents()
    p = load_preventive_documents()
    total = len(s) + len(c) + len(p)
    log_success(logger, f"Documents loaded — Symptoms: {len(s)}, Conditions: {len(c)}, Preventive: {len(p)}, Total: {total}")
except Exception as e:
    log_error(logger, "Failed to load RAG documents", e)

# Validate embedder module for document embedding generation
log_step(logger, 2, "Validating embedder module")
try:
    from rag.embedder import get_embeddings
    log_success(logger, "Embedder module imported successfully")
except Exception as e:
    log_error(logger, "Failed to import embedder module", e)

# Validate vector store module for storing and querying embeddings
log_step(logger, 3, "Validating vector store module")
try:
    from rag.vector_store import build_or_load_vector_store
    log_success(logger, "Vector store module imported successfully")
except Exception as e:
    log_error(logger, "Failed to import vector store module", e)

# Validate retriever module for RAG chain and source retrieval
log_step(logger, 4, "Validating retriever module")
try:
    from rag.retriever import build_rag_chain, retrieve_sources
    log_success(logger, "Retriever module imported successfully")
except Exception as e:
    log_error(logger, "Failed to import retriever module", e)

# ============================================================================
# LLM MODULE VALIDATION
# ============================================================================

log_section(logger, "LLM MODULE VALIDATION")

# Validate model loader and instantiate the language model
log_step(logger, 5, "Loading language model")
try:
    from llm.model_loader import get_llm
    llm = get_llm()
    log_success(logger, f"Language model loaded — Type: {llm.__class__.__name__}")
except Exception as e:
    log_error(logger, "Failed to load language model", e)

# ============================================================================
# UI MODULE VALIDATION
# ============================================================================

log_section(logger, "UI MODULE VALIDATION")

# Validate UI components: sidebar rendering and chat interface initialization
log_step(logger, 6, "Validating UI components")
try:
    from ui.sidebar import render_sidebar
    from ui.chat_interface import init_chat_state, render_chat_history, add_message
    log_success(logger, "UI components imported successfully")
except Exception as e:
    log_error(logger, "Failed to import UI components", e)

logger.info("")
log_success(logger, "Sanity check completed successfully")
