# Sanity Check – Comprehensive library import validation
"""
Validates all required libraries for the Healthcare Symptom Checker project.
Provides detailed feedback on each import with structured logging.
"""
from logger_config import get_logger, log_success, log_error, log_section

logger = get_logger(__name__)

checks = [
    ("requests", "requests"),
    ("streamlit", "streamlit"),
    ("Endpoint client", "llm.endpoint_client"),
    ("LangChain", "langchain"),
    ("LangChain Core", "langchain_core"),
    ("LangChain Community", "langchain_community"),
    ("LangChain HuggingFace", "langchain_huggingface"),
    ("LangChain Text Splitters", "langchain_text_splitters"),
    ("FAISS", "faiss"),
    ("Sentence Transformers", "sentence_transformers"),
    ("Transformers", "transformers"),
    ("Torch", "torch"),
]

log_section(logger, "Library Import Validation")
logger.info(f"Validating {len(checks)} required libraries...")
failed = []

for label, module_name in checks:
    try:
        __import__(module_name)
        log_success(logger, f"{label} import")
    except Exception as exc:
        log_error(logger, f"{label} import failed ({exc.__class__.__name__})", exc)
        failed.append((label, module_name, str(exc)))

if failed:
    logger.error(f"❌ Sanity check FAILED: {len(failed)}/{len(checks)} import(s) failed")
    logger.error("Failed imports:")
    for label, module_name, error in failed:
        logger.error(f"  • {label} ({module_name}): {error}")
    raise SystemExit(1)

log_success(logger, f"All {len(checks)} libraries imported successfully")
logger.info("✅ Sanity result: PASSED")
