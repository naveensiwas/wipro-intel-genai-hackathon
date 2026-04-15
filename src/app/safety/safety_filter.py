"""
Safety filter for post-processing LLM responses.
Softens hard diagnostic language and ensures every response
ends with a reminder to consult a healthcare professional.
Also provides an input-side off-domain check.
"""

import re
import random
from app.core.logging import get_logger, log_warning
from app.config.settings import cfg
from app.rag.retriever import get_top_similarity_score

logger = get_logger(__name__)

# ── Greeting detection (fast-path, runs before domain gate) ───────────────────

# Regex that matches plain greeting inputs — no RAG or LLM call is made for these.
# Covers:
#   • Bare greetings:              "hi", "hello!", "hey there"
#   • Repeated greeting words:     "hi hello", "hello hey", "hey hi there"
#   • Greeting + social filler:    "hello how are you?", "hi how are you doing"
#   • Standalone social phrases:   "how are you?", "how's it going", "what's up"
_GREETING_WORD = r"(?:hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening|greetings)"
_SOCIAL_FILLER = r"(?:how\s+are\s+you(?:\s+doing)?|how(?:'s|\s+is)\s+it\s+going|how\s+do\s+you\s+do|what'?s\s+up|howdy)"
_GREETING_PATTERN = re.compile(
    r"^\s*(?:"
        # Branch 1 — starts with one or more greeting words, optionally followed by a social filler
        + _GREETING_WORD + r"(?:[!.?,\s]+" + _GREETING_WORD + r")*"
        + r"(?:[!.?,\s]+" + _SOCIAL_FILLER + r")?"
    + r"|"
        # Branch 2 — standalone social filler with no greeting word
        + _SOCIAL_FILLER
    + r")[!.?,\s]*$",
    re.IGNORECASE,
)

# Pool of varied greeting replies — one is chosen at random per interaction
# to make the assistant feel more natural and less repetitive.
_GREETING_RESPONSES = [
    "Hey there! 👋 I'm your Healthcare Symptom Assistant. Feel free to share what's been bothering you — I'm here to help.",
    "Hello! 😊 Great to see you. Go ahead and describe your symptoms or ask me anything health-related.",
    "Hi! I'm ready to help with any health questions you have. What's on your mind today?",
    "Hey! 👋 Whether it's a symptom you're unsure about or a condition you'd like to know more about, I've got you covered.",
    "Hello there! I'm here to help you with health and symptom information. What would you like to know?",
    "Hi there! 😊 Tell me what's going on — I can help with symptoms, conditions, or general health questions.",
    "Hey, good to have you here! I specialise in health and symptom information. What can I help you with today?",
    "Hello! I'm your health information assistant. Don't hesitate to share your symptoms or ask about a condition.",
    "I'm doing great, thanks for asking! 😊 I'm here whenever you're ready — feel free to share your symptoms or any health question.",
    "All good here! 👋 I'm your Healthcare Symptom Assistant, ready to help. What health question can I answer for you today?",
]


def is_simple_greeting(text: str) -> bool:
    """Return True if *text* is a plain greeting with no health content."""
    if not text:
        return False
    return bool(_GREETING_PATTERN.match(text.strip()))


def get_greeting_response() -> str:
    """Return a randomly chosen greeting reply (no retrieval / no LLM call needed)."""
    return random.choice(_GREETING_RESPONSES)


# ── Off-domain input filter ────────────────────────────────────────────────────

# Keywords that indicate a health-related query.
# NOTE: Greeting detection is handled by is_simple_greeting() / _GREETING_PATTERN above,
# which runs before the domain gate is ever reached, so greeting words are intentionally excluded here.
_HEALTH_KEYWORDS = [

    # Health symptoms and conditions
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
    "period", "menstrual", "rash", "blister", "wound", "bleeding", "migraine"
]

# Off-domain response shown to the user
_OFF_DOMAIN_MESSAGE = (
    "I’m a **Healthcare Symptom Information Assistant** and can help answer health‑ and medical‑related questions by providing general information.\n\n"
    "Please describe your symptoms or ask a health‑related question to get started.\n\n"
    "*Example questions:*\n"
    "- *I’ve had a persistent cough and fever for the past three days*\n"
    "- *What could cause sharp chest pain?*\n"
    "- *What are common symptoms of diabetes?*"
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


def is_health_related_semantic(query: str, vector_store) -> tuple[bool, float]:
    """
    Semantic domain gate using FAISS relevance score.

    Checks if a query is health-related by computing its semantic similarity
    to the indexed health knowledge base. More robust than keyword matching
    for synonym detection and phrasing variations.

    Args:
        query: User input string to classify
        vector_store: FAISS vector store instance (indexed health documents)

    Returns:
        tuple: (is_health_related, similarity_score)
            - is_health_related (bool): True if score >= domain_similarity_threshold
            - similarity_score (float): Normalized score in [0.0, 1.0] range
              representing semantic similarity to health domain

    Note:
        Returns (False, 0.0) if FAISS returns no results.
        Scores are normalized via distance transformation to ensure consistency
        across different FAISS distance metrics.
    """
    try:
        score = get_top_similarity_score(
            vector_store,
            query,
            k=cfg.domain_similarity_k,
        )
        is_in_domain = score >= cfg.domain_similarity_threshold
        if is_in_domain:
            logger.debug(
                "Semantic domain gate: in-domain "
                f"(score={score:.4f}, threshold={cfg.domain_similarity_threshold:.4f})"
            )
        else:
            log_warning(
                logger,
                "Semantic domain gate: off-domain "
                f"(score={score:.4f}, threshold={cfg.domain_similarity_threshold:.4f})",
            )
        return is_in_domain, score
    except Exception as exc:
        log_warning(logger, f"Semantic domain gate failed; falling back to keyword gate: {exc}")
        return is_health_related(query), 0.0


def is_health_related_with_mode(query: str, vector_store=None) -> tuple[bool, float | None]:
    """
    Domain gate router that classifies queries by health-relatedness.

    Behavior depends on DOMAIN_FILTER_MODE configuration:
    - "keyword": Keyword-only matching (fast, simple, misses synonyms)
    - "semantic": Semantic similarity via FAISS with optional keyword fallback
    - "hybrid": Semantic check OR keyword check (allows both, permissive)

    Keyword fallback is controlled by DOMAIN_USE_KEYWORD_FALLBACK setting
    and provides a safety net when semantic scoring is unreliable.

    Args:
        query: User input to classify
        vector_store: FAISS vector store (required for semantic/hybrid modes,
                     ignored for keyword-only mode)

    Returns:
        tuple: (is_allowed, semantic_score)
            - is_allowed (bool): Whether query passes domain gate and should be processed
            - semantic_score (float | None): Normalized [0.0, 1.0] similarity score
              if available from semantic gate, else None

    Examples:
        # Keyword mode → always returns (result, None)
        allowed, score = is_health_related_with_mode("I have a fever", None)
        # Returns: (True, None)

        # Semantic mode with vector store
        allowed, score = is_health_related_with_mode("How's the weather?", vs)
        # Returns: (False, 0.15) if below threshold
    """
    mode = cfg.domain_filter_mode.strip().lower()

    # For keyword-only mode or if vector store isn't available, rely solely on keyword matching
    if mode == "keyword" or vector_store is None:
        return is_health_related(query), None


    # For semantic or hybrid modes, first attempt the semantic check
    semantic_ok, score = is_health_related_semantic(query, vector_store)

    # For semantic-only mode, the semantic check is definitive (with optional keyword fallback for health-related queries)
    if mode == "semantic":
        if semantic_ok:
            return True, score
        if cfg.domain_use_keyword_fallback and is_health_related(query):
            logger.debug("Semantic gate rejected query, but keyword fallback allowed it")
            return True, score
        return False, score

    # For hybrid mode, allow if either semantic OR keyword check passes (with optional keyword fallback)
    if mode == "hybrid":
        if semantic_ok:
            return True, score
        keyword_ok = is_health_related(query) if cfg.domain_use_keyword_fallback else False
        return keyword_ok, score

    log_warning(logger, f"Unknown DOMAIN_FILTER_MODE='{cfg.domain_filter_mode}', defaulting to keyword")
    return is_health_related(query), score
