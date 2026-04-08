"""
Structured logging configuration for the Healthcare Symptom Checker application.

Provides:
- Colored, timestamped console output via StructuredFormatter
- Per-level emoji prefixes for quick visual scanning
- Optional job/document context injected via ContextFilter
- Convenience helpers: get_logger, set_context, log_section, log_step, etc.

Usage:
from app.core.logging import get_logger, log_success
    logger = get_logger(__name__)
    logger.info("Starting RAG pipeline")
    log_success(logger, "Vector store loaded")
"""

import logging
import sys
from datetime import datetime
from typing import Optional


class Colors:
    """ANSI escape codes used to colorize terminal log output."""

    RESET = '\033[0m'
    BOLD  = '\033[1m'

    # Standard foreground colors
    BLACK   = '\033[30m'
    RED     = '\033[31m'
    GREEN   = '\033[32m'
    YELLOW  = '\033[33m'
    BLUE    = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN    = '\033[36m'
    WHITE   = '\033[37m'

    # High-intensity (bright) foreground colors
    BRIGHT_RED     = '\033[91m'
    BRIGHT_GREEN   = '\033[92m'
    BRIGHT_YELLOW  = '\033[93m'
    BRIGHT_BLUE    = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN    = '\033[96m'
    BRIGHT_WHITE   = '\033[97m'


class StructuredFormatter(logging.Formatter):
    """
    Custom log formatter that produces colored, pipe-delimited output:
        <timestamp> | <emoji> <LEVEL> | <logger_name> | <message>
    """

    # ANSI color per log level
    LEVEL_COLORS = {
        'DEBUG':    Colors.BLUE,
        'INFO':     Colors.GREEN,
        'WARNING':  Colors.YELLOW,
        'ERROR':    Colors.RED,
        'CRITICAL': Colors.BRIGHT_RED,
    }

    # Emoji prefix per log level for fast visual scanning
    LEVEL_EMOJIS = {
        'DEBUG':    '🔧',
        'INFO':     'ℹ️ ',
        'WARNING':  '⚠️ ',
        'ERROR':    '❌',
        'CRITICAL': '🚨',
    }

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as a colored, structured string."""
        # Millisecond-precision timestamp
        timestamp   = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level_name  = record.levelname
        level_color = self.LEVEL_COLORS.get(level_name, Colors.WHITE)
        emoji       = self.LEVEL_EMOJIS.get(level_name, '  ')

        return (
            f"{Colors.CYAN}{timestamp}{Colors.RESET} | "
            f"{emoji} {level_color}{level_name:8s}{Colors.RESET} | "
            f"{Colors.MAGENTA}{record.name:25s}{Colors.RESET} | "
            f"{record.getMessage()}"
        )


class ContextFilter(logging.Filter):
    """
    Logging filter that injects job_id and document_id into every log record.

    Context is set once (e.g. at the start of a job) and cleared when done,
    so callers never need to pass IDs on every log call.
    """

    def __init__(self):
        super().__init__()
        self.context: dict = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach current job_id and document_id to the record."""
        record.job_id      = self.context.get('job_id', '')
        record.document_id = self.context.get('document_id', '')
        return True

    def set_context(self, job_id: Optional[str] = None, document_id: Optional[str] = None):
        """Store job/document identifiers for subsequent log records."""
        if job_id:
            self.context['job_id'] = job_id
        if document_id:
            self.context['document_id'] = document_id

    def clear_context(self):
        """Remove all stored context (call after a job completes)."""
        self.context = {}


# Module-level singleton — shared by all loggers created via get_logger()
_context_filter = ContextFilter()


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger configured with StructuredFormatter and ContextFilter.

    Idempotent: calling twice with the same name returns the same logger
    without adding duplicate handlers.

    Args:
        name: Logger name — pass __name__ from the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:   # avoid duplicate handlers on repeated calls
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(StructuredFormatter())
        handler.addFilter(_context_filter)

        logger.addHandler(handler)
        logger.propagate = False   # prevent double-printing via root logger

    return logger


def set_context(job_id: Optional[str] = None, document_id: Optional[str] = None):
    """
    Set job/document context for all subsequent log messages in this process.

    Prints a one-time visual header so the context is visible at the top of
    a log block without repeating it on every line.

    Args:
        job_id:      Identifier for the current processing job.
        document_id: Identifier for the document being processed.
    """
    _context_filter.set_context(job_id, document_id)

    # Build and print a single context banner
    context_parts = []
    if job_id:
        context_parts.append(f"JOB ID: {job_id}")
    if document_id:
        context_parts.append(f"DOCUMENT: {document_id}")

    if context_parts:
        sep = "=" * 80
        print(f"\n{Colors.CYAN}{sep}{Colors.RESET}")
        print(f"{Colors.CYAN}{'  '.join(context_parts)}{Colors.RESET}")
        print(f"{Colors.CYAN}{sep}{Colors.RESET}\n")


def clear_context():
    """Clear job/document context (call after a job or request completes)."""
    _context_filter.clear_context()


def log_section(logger: logging.Logger, title: str):
    """Log a visual section divider — useful for separating pipeline stages."""
    sep = "=" * 60
    logger.info("")
    logger.info(sep)
    logger.info(title)
    logger.info(sep)


def log_step(logger: logging.Logger, step_number: int, description: str):
    """Log a numbered pipeline step (e.g. STEP 1: Loading vector store)."""
    logger.info("")
    logger.info(f"STEP {step_number}: {description}")


def log_success(logger: logging.Logger, message: str):
    """Log a success event at INFO level with a ✅ prefix."""
    logger.info(f"✅ SUCCESS: {message}")


def log_error(logger: logging.Logger, message: str, exception: Optional[Exception] = None):
    """
    Log an error at ERROR level with an ❌ prefix.

    Args:
        message:   Human-readable description of what failed.
        exception: If provided, full traceback is included via exc_info.
    """
    if exception:
        logger.error(f"❌ ERROR: {message}", exc_info=exception)
    else:
        logger.error(f"❌ ERROR: {message}")


def log_warning(logger: logging.Logger, message: str):
    """Log a warning at WARNING level with a ⚠️ prefix."""
    logger.warning(f"⚠️  WARNING: {message}")


# ── Root logger setup ─────────────────────────────────────────────────────────
# Configures the root logger, so that logs from libraries also get structured formatting and context.
root_logger = logging.getLogger()
if not root_logger.handlers:
    root_logger.setLevel(logging.INFO)
    _root_handler = logging.StreamHandler(sys.stdout)
    _root_handler.setFormatter(StructuredFormatter())
    _root_handler.addFilter(_context_filter)
    root_logger.addHandler(_root_handler)
