"""
Error handling and logging utilities.

Provides decorators and context managers to reduce boilerplate when
handling exceptions and logging them consistently across modules.

This consolidates the repeated try/except + logging pattern found in
model_loader.py, vector_store.py, embedder.py, and other modules.
"""
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional, TypeVar
import logging

T = TypeVar('T')


def handle_with_logging(
    logger: logging.Logger,
    operation_name: str,
    error_prefix: str = "Operation failed"
) -> Callable:
    """
    Decorator to wrap functions with automatic error logging.

    If the decorated function raises an exception, it will be logged
    via the provided logger and re-raised.

    Args:
        logger: Logger instance to use for error logging
        operation_name: Human-readable name of the operation being performed
        error_prefix: Custom prefix for error messages (default: "Operation failed")

    Returns:
        Decorated function that logs exceptions before re-raising

    Example:
        @handle_with_logging(logger, "Loading embeddings")
        def get_embeddings():
            return HuggingFaceEmbeddings(...)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                logger.error(f"{error_prefix}: {operation_name}", exc_info=exc)
                raise
        return wrapper
    return decorator


@contextmanager
def log_operation(
    logger: logging.Logger,
    operation: str,
    success_message: Optional[str] = None,
    error_context: Optional[dict] = None
):
    """
    Context manager for logging the success or failure of an operation.

    Logs entry and exit of an operation with automatic exception handling.

    Args:
        logger: Logger instance to use
        operation: Description of the operation
        success_message: Custom message to log on success (default: "Operation completed")
        error_context: Extra context dict to include in error logs

    Yields:
        None

    Example:
        with log_operation(logger, "Building vector store", success_message="Index created"):
            vector_store = FAISS.from_documents(chunks, embeddings)
    """
    logger.debug(f"Starting: {operation}")
    try:
        yield
        msg = success_message or "Operation completed"
        logger.info(f"✅ {msg}")
    except Exception as exc:
        error_msg = f"❌ {operation} failed"
        if error_context:
            error_msg += f" | Context: {error_context}"
        logger.error(error_msg, exc_info=exc)
        raise


def safe_call(
    func: Callable[..., T],
    logger: logging.Logger,
    operation_name: str,
    default_return: Optional[T] = None,
    reraise: bool = True
) -> Optional[T]:
    """
    Execute a function safely with logging, optionally catching exceptions.

    Useful for non-critical operations where a default value is acceptable
    on failure (e.g., loading optional configuration files).

    Args:
        func: The function to call
        logger: Logger instance
        operation_name: Description of the operation
        default_return: Value to return if an exception occurs (default: None)
        reraise: If True, exception is re-raised after logging; if False, suppressed

    Returns:
        Result of func() on success, or default_return on failure

    Example:
        logo_img = safe_call(
            load_image,
            logger,
            "Loading company logo",
            default_return=None,
            reraise=False
        )
    """
    try:
        return func()
    except Exception as exc:
        logger.error(f"Failed to {operation_name}", exc_info=exc)
        if reraise:
            raise
        return default_return
      
