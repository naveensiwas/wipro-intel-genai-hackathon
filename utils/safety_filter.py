"""
Safety filter for post-processing LLM responses.
Softens hard diagnostic language and ensures every response
ends with a reminder to consult a healthcare professional.
"""
import re
from logger_config import get_logger, log_warning

logger = get_logger(__name__)

# Patterns that suggest a firm diagnosis — replace with safer phrasing
_DIAGNOSIS_PATTERNS = [
    (r"\bYou have\b", "This may be associated with"),
    (r"\byou have\b", "this may be associated with"),
    (r"\bYou are diagnosed with\b", "The symptoms described could be related to"),
    (r"\byou are diagnosed with\b", "the symptoms described could be related to"),
    (r"\bYou definitely have\b", "The information suggests a possible association with"),
    (r"\byou definitely have\b", "the information suggests a possible association with"),
    (r"\bThis is clearly\b", "This may be"),
    (r"\bthis is clearly\b", "this may be"),
    (r"\bYou suffer from\b", "You may be experiencing symptoms associated with"),
    (r"\byou suffer from\b", "you may be experiencing symptoms associated with"),
]

_CONSULT_REMINDER = (
    "\n\n---\n"
    "⚕️ *This information is for general educational purposes only and does not constitute "
    "medical advice, diagnosis, or treatment. Please consult a qualified healthcare professional "
    "for personalised medical guidance.*"
)

_CONSULT_KEYWORDS = [
    "consult", "speak with", "see a doctor", "healthcare professional",
    "medical professional", "specialist", "physician", "general practitioner"
]


def sanitize_response(text: str) -> str:
    """
    Apply safety patterns to the LLM response:
    1. Replace hard diagnostic phrases with cautious alternatives.
    2. Append a disclaimer if no consultation reminder is present.
    """
    original_length = len(text)

    # Apply phrase replacement patterns
    replacements_made = 0
    for pattern, replacement in _DIAGNOSIS_PATTERNS:
        new_text = re.sub(pattern, replacement, text)
        if new_text != text:
            replacements_made += 1
            text = new_text

    if replacements_made:
        log_warning(logger, f"Safety filter replaced {replacements_made} diagnostic phrase(s) in response")

    # Always append the formal disclaimer
    text += _CONSULT_REMINDER

    logger.debug(
        f"Safety filter applied — input={original_length} chars, "
        f"output={len(text)} chars, replacements={replacements_made}"
    )

    return text.strip()
