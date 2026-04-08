"""
Shared utilities for rendering retrieved source documents.

Consolidates source metadata extraction and display logic used across
multiple UI components (main chat interface and history rendering).
"""
from typing import Dict, Any


def format_source_label(metadata: Dict[str, Any], fallback_label: str = "Health Data") -> str:
    """
    Extract and format a human-readable label from source metadata.

    Falls back through: condition → symptom → category → fallback_label.

    Args:
        metadata: Dictionary of source metadata containing optional keys:
                  'condition', 'symptom', 'category', 'source_file'
        fallback_label: Default label if no recognized metadata field is found

    Returns:
        Formatted label string suitable for UI display
    """
    return (
        metadata.get("condition")
        or metadata.get("symptom")
        or metadata.get("category")
        or metadata.get("source_file", fallback_label)
    )


def truncate_source_content(content: str, max_length: int = 300) -> str:
    """
    Truncate source content to a display-friendly length.

    Args:
        content: Raw source document content
        max_length: Maximum number of characters to retain

    Returns:
        Truncated content with ellipsis if original exceeds max_length
    """
    if len(content) > max_length:
        return content[:max_length] + "..."
    return content

