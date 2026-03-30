"""
Safety filter for post-processing LLM responses.
Softens hard diagnostic language and ensures every response
ends with a reminder to consult a healthcare professional.
Also provides an input-side off-domain check.
"""

import re
from logger_config import get_logger, log_warning

logger = get_logger(__name__)

# ── Off-domain input filter ────────────────────────────────────────────────────

# Keywords that indicate a health-related query
_HEALTH_KEYWORDS = [
    "symptom", "symptoms", "pain", "fever", "cough", "headache", "doctor",
    "medicine", "treatment", "diagnosis", "disease", "condition", "health",
    "hospital", "medication", "infection", "injury", "rash", "nausea",
    "fatigue", "breathing", "chest", "blood", "heart", "skin", "stomach",
    "throat", "cold", "flu", "allergy", "allergic", "vomit", "diarrhea",
    "diarrhoea", "dizzy", "dizziness", "swelling", "swollen", "ache",
    "hurts", "hurt", "sore", "tired", "weakness", "weight", "appetite",
    "urination", "discharge", "itching", "itchy", "burn", "burning",
    "numbness", "tingling", "vision", "hearing", "anxiety", "depression",
    "mental", "sleep", "insomnia", "diabetes", "cancer", "stroke", "asthma",
    "hypertension", "pressure", "pulse", "temperature", "clinic", "surgery",
    "specialist", "consult", "prescription", "tablet", "pill", "dose",
    "medical", "diagnose", "examine", "test", "scan", "xray", "x-ray",
    "mri", "blood test", "urine", "stool", "pregnant", "pregnancy",
    "period", "menstrual", "rash", "blister", "wound", "bleeding",
]

# Off-domain response shown to the user
_OFF_DOMAIN_MESSAGE = (
    "I'm a **Healthcare Symptom Information Assistant** and can only help "
    "with health and medical-related questions.\n\n"
    "Please describe your symptoms or ask a health-related question.\n\n"
    "*Examples:*\n"
    "- *I have a persistent cough and fever for 3 days*\n"
    "- *What could cause sharp chest pain?*\n"
    "- *What are the symptoms of diabetes?*"
)

# ── Post-processing safety filter ─────────────────────────────────────────────

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


def is_health_related(query: str) -> bool:
    """
    Return True if the query contains at least one health-related keyword.
    Used to filter off-domain questions before invoking the RAG pipeline.
    """
    q = query.lower()
    matched = [kw for kw in _HEALTH_KEYWORDS if kw in q]
    if matched:
        logger.debug(f"Health query confirmed — matched keywords: {matched[:5]}")
        return True
    log_warning(logger, f"Off-domain query detected — no health keywords matched: '{query[:80]}'")
    return False


def get_off_domain_message() -> str:
    """Return the standard off-domain response message."""
    return _OFF_DOMAIN_MESSAGE


def sanitize_response(text: str) -> str:
    """
    Apply safety patterns to the LLM response:
    1. Replace hard diagnostic phrases with cautious alternatives.
    2. Append a disclaimer at the end.
    """
    # Apply phrase replacement patterns
    replacements_made = 0
    for pattern, replacement in _DIAGNOSIS_PATTERNS:
        new_text, count = re.subn(pattern, replacement, text)
        if count > 0:
            replacements_made += count
            text = new_text
            log_warning(
                logger,
                f"Safety filter replacement: '{pattern}' -> '{replacement}' ({count}x)",
            )

    if replacements_made:
        log_warning(
            logger,
            f"Safety filter replaced {replacements_made} diagnostic phrase(s) in response",
        )

    # Always append the formal disclaimer
    text += _CONSULT_REMINDER

    logger.debug(
        f"Safety filter applied — input={len(text)} chars, "
        f"output={len(text)} chars, replacements={replacements_made}"
    )

    return text.strip()
