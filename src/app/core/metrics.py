"""
Performance metrics collection, computation, and persistence for the
Healthcare Symptom Information Assistant.

This module tracks request-level performance metrics and maintains rolling
summary statistics. All metrics are JSON-persisted and thread-safe.

## Metrics Collected Per Request
─────────────────────────────
  • timestamp_utc          — ISO-8601 timestamp of request completion
  • query_preview          — First 120 chars of user input (for inspection)
  • ttft_ms                — Time To First Token in milliseconds (latency to first output char)
  • total_latency_ms       — Wall-clock duration from user input submission to response ready
  • tokens_per_sec         — Output token throughput (output_tokens / generation_duration)
  • retrieval_latency_ms   — Duration of FAISS semantic search step only
  • end_to_end_latency_ms  — Alias for total_latency_ms (full round-trip)
  • context_size_tokens    — Approximate token count of all retrieved source chunks
  • output_tokens          — Approximate token count of generated response
  • concurrent_users       — Number of active requests at request start time
  • error                  — Error message (string) if request failed, else null

## Rolling Summary (Updated After Every Request)
───────────────────────────────────────────────
  • total_requests             — Count of all requests (successful + failed)
  • successful_requests        — Count of error-free requests
  • avg_total_latency_ms       — Mean of total_latency_ms across successful requests
  • max_concurrent_users_observed — Highest concurrent_users value seen
  • last_updated_utc           — Timestamp of most recent metrics write

## Token Count Approximation
────────────────────────────
Token counts are approximated via whitespace splitting (word count) for efficiency.
This avoids heavyweight tokenizers but sacrifices ~10-15% accuracy vs. sub-word tokenization.

## Thread Safety
───────────────
All writes to the metrics JSON file are protected by a module-level Lock (_METRICS_LOCK).
Safe for concurrent requests from multiple Streamlit sessions.

## Schema Example
─────────────────
{
  "requests": [
    {
      "timestamp_utc": "2025-04-04T15:32:45Z",
      "query_preview": "I have a persistent cough and fever...",
      "ttft_ms": 234.56,
      "total_latency_ms": 2456.78,
      "tokens_per_sec": 12.34,
      "retrieval_latency_ms": 123.45,
      "end_to_end_latency_ms": 2456.78,
      "context_size_tokens": 450,
      "output_tokens": 234,
      "concurrent_users": 1,
      "error": null
    }
  ],
  "summary": {
    "total_requests": 42,
    "successful_requests": 41,
    "avg_total_latency_ms": 1234.56,
    "max_concurrent_users_observed": 3,
    "last_updated_utc": "2025-04-04T15:32:45Z"
  }
}
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from app.config.settings import cfg

# ── Storage ────────────────────────────────────────────────────────────────────
METRICS_FILE_PATH: Path = Path(cfg.metrics_file_path)
_METRICS_LOCK: Lock = Lock()


def _default_payload() -> dict:
    return {
        "requests": [],
        "summary": {
            "total_requests": 0,
            "successful_requests": 0,
            "avg_total_latency_ms": None,
            "max_concurrent_users_observed": 0,
            "last_updated_utc": _utc_now_iso(),
        },
    }


def _ensure_metrics_file_exists() -> None:
    """Ensure the metrics directory and JSON file exist before reading/writing."""
    METRICS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not METRICS_FILE_PATH.exists():
        METRICS_FILE_PATH.write_text(
            json.dumps(_default_payload(), indent=2),
            encoding="utf-8",
        )


def _safe_token_count(text: str) -> int:
    """
    Approximate token count for a string.
    Uses whitespace splitting as a lightweight fallback when a tokenizer is
    not available.  Accuracy is ±10–15 % vs. sub-word tokenizers.
    """
    return len(text.split()) if text else 0


def _to_ms(seconds: float) -> float:
    """Convert a non-negative duration in seconds to rounded milliseconds."""
    return round(max(seconds, 0.0) * 1000.0, 2)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with a trailing 'Z'."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def persist_metrics(entry: dict) -> None:
    """
    Append *entry* to the metrics JSON file and refresh the rolling summary.

    Thread-safe: protected by a module-level ``Lock``.

    Parameters
    ----------
    entry : dict
        A metrics dictionary as produced by :func:`build_metrics_entry`.
    """
    with _METRICS_LOCK:
        _ensure_metrics_file_exists()
        try:
            payload = json.loads(METRICS_FILE_PATH.read_text(encoding="utf-8"))
        except Exception:
            payload = _default_payload()

        requests: list = payload.get("requests", [])
        requests.append(entry)

        successful = [
            r for r in requests
            if not r.get("error") and r.get("total_latency_ms") is not None
        ]
        avg_total_latency = (
            round(sum(r["total_latency_ms"] for r in successful) / len(successful), 2)
            if successful
            else None
        )
        max_concurrency = max(
            (r.get("concurrent_users", 0) for r in requests), default=0
        )

        payload["requests"] = requests
        payload["summary"] = {
            "total_requests": len(requests),
            "successful_requests": len(successful),
            "avg_total_latency_ms": avg_total_latency,
            "max_concurrent_users_observed": max_concurrency,
            "last_updated_utc": _utc_now_iso(),
        }

        METRICS_FILE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_metrics_entry(
    *,
    user_input: str,
    request_started: float,
    request_ended: float,
    retrieval_started: float | None,
    retrieval_ended: float | None,
    generation_started: float | None,
    generation_ended: float | None,
    first_token_ts: float | None,
    sources: list,
    safe_response: str,
    concurrent_users: int,
    error_message: str | None,
) -> dict:
    """
    Compute all performance metrics from raw timing values and return a
    ready-to-persist dictionary.

    All ``*_started`` / ``*_ended`` values are ``time.perf_counter()`` floats.
    Pass ``None`` for any phase that did not execute (e.g. retrieval was
    skipped because the query was off-domain).

    Returns
    -------
    dict
        Keys: timestamp_utc, query_preview, ttft_ms, total_latency_ms,
        tokens_per_sec, retrieval_latency_ms, end_to_end_latency_ms,
        context_size_tokens, output_tokens, concurrent_users, error.
    """
    context_size_tokens = sum(_safe_token_count(src.get("content", "")) for src in sources)
    output_tokens = _safe_token_count(safe_response)

    total_latency_s = request_ended - request_started

    retrieval_latency_s = (
        retrieval_ended - retrieval_started
        if retrieval_started is not None and retrieval_ended is not None
        else None
    )

    ttft_s = (
        first_token_ts - request_started
        if first_token_ts is not None
        else None
    )

    tokens_per_sec = (
        round(
            output_tokens / max((generation_ended - generation_started), 1e-9), 2
        )
        if generation_started is not None
        and generation_ended is not None
        and output_tokens > 0
        else None
    )

    return {
        "timestamp_utc": _utc_now_iso(),
        "query_preview": user_input[:120],
        "ttft_ms": _to_ms(ttft_s) if ttft_s is not None else None,
        "total_latency_ms": _to_ms(total_latency_s),
        "tokens_per_sec": tokens_per_sec,
        "retrieval_latency_ms": _to_ms(retrieval_latency_s) if retrieval_latency_s is not None else None,
        "end_to_end_latency_ms": _to_ms(total_latency_s),   # full round-trip
        "context_size_tokens": context_size_tokens,
        "output_tokens": output_tokens,
        "concurrent_users": concurrent_users,
        "error": error_message,
    }


def reset_metrics() -> None:
    """Clear all persisted metrics and reset summary counters."""
    with _METRICS_LOCK:
        METRICS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_FILE_PATH.write_text(json.dumps(_default_payload(), indent=2), encoding="utf-8")
