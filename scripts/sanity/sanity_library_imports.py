"""
Sanity Check – Comprehensive library import validation for Healthcare Symptom Checker project.

Validates all required libraries for the Healthcare Symptom Checker project.
Provides detailed feedback on each import with structured logging.
"""
from app.core.logging import get_logger, log_success, log_error, log_section
logger = get_logger(__name__)

# List of libraries to validate (label, module name)
checks = [
    ("requests", "requests"),
    ("streamlit", "streamlit"),
    ("Endpoint client", "app.llm.endpoint_client"),
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

# Adding logging section header
log_section(logger, "Library Import Validation")
logger.info(f"Validating {len(checks)} required libraries...")

# Attempting imports and logging results
failed = []
for label, module_name in checks:
    try:
        __import__(module_name)
        log_success(logger, f"{label} import")

    # Catching any exception to log details and continue checking other libraries
    except Exception as exc:
        log_error(logger, f"{label} import failed ({exc.__class__.__name__})", exc)
        failed.append((label, module_name, str(exc)))

# If any imports failed, log details and exit with error
if failed:
    log_error(logger,f"Sanity check FAILED: {len(failed)}/{len(checks)} import(s) failed")
    log_error(logger, "Failed imports:")
    for label, module_name, error in failed:
        log_error(logger,f" • {label} ({module_name}): {error}")
    raise SystemExit(1)

# If we reach this point, all imports succeeded
log_success(logger, f"All {len(checks)} libraries imported successfully")
log_success(logger, "Sanity result PASSED")
