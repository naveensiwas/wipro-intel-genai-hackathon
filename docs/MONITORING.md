# 📊 Performance Monitoring

> [← Back to README](../README.md)

---

## Metrics Schema

All per-request metrics are stored in `data/runtime/metrics/performance_metrics.json`:

```json
{
  "requests": [
    {
      "timestamp_utc": "2026-04-08T10:32:45Z",
      "query_preview": "I have a persistent cough and fever...",
      "ttft_ms": 234.56,
      "total_latency_ms": 2456.78,
      "end_to_end_latency_ms": 2456.78,
      "tokens_per_sec": 12.34,
      "retrieval_latency_ms": 123.45,
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
    "last_updated_utc": "2026-04-08T10:32:45Z"
  }
}
```

---

## Metrics Field Reference

| Field | Unit | Description |
|-------|------|-------------|
| `timestamp_utc` | ISO-8601 | UTC request completion time (displayed as IST in dashboard) |
| `query_preview` | String | First 120 chars of user query |
| `ttft_ms` | ms | Time To First Token (approximated as generation end for non-streaming) |
| `total_latency_ms` | ms | Wall-clock time from user input to response ready |
| `end_to_end_latency_ms` | ms | Alias for `total_latency_ms` |
| `tokens_per_sec` | float | Output tokens ÷ generation duration |
| `retrieval_latency_ms` | ms | Duration of FAISS similarity search only |
| `context_size_tokens` | count | Approx. tokens in all retrieved source chunks (word-split) |
| `output_tokens` | count | Approx. tokens in generated response (word-split) |
| `concurrent_users` | count | Active in-flight requests at submission time |
| `error` | string\|null | Exception message if request failed; null otherwise |

> **Token approximation note:** Token counts use `len(text.split())` (whitespace splitting) to avoid loading a heavyweight tokenizer per request. Accuracy is ~85-90% vs. sub-word tokenization.

---

## Rolling Summary Fields

Updated after every request in the `"summary"` block:

| Field | Description |
|-------|-------------|
| `total_requests` | Count of all requests (successful + failed) |
| `successful_requests` | Count of requests with no recorded error |
| `avg_total_latency_ms` | Mean `total_latency_ms` across successful requests |
| `max_concurrent_users_observed` | Highest `concurrent_users` value seen |
| `last_updated_utc` | Timestamp of most recent metrics write |

---

## Metrics Dashboard Panels

Access the dashboard via **📊 Model Latency & Throughput Metrics** in the sidebar:

| Panel | Metrics Shown |
|-------|---------------|
| **Health Status** | 🟢 Healthy (≥80% success) / 🟡 Degraded (≥60%) / 🔴 Unhealthy (<60%) |
| **Summary** | Total requests, successful, avg total latency (sec), max concurrent users |
| **Latency Averages** | End-to-end, retrieval, total, TTFT — all displayed in seconds |
| **Throughput & Context** | Tokens/sec, context size (tokens), output tokens, avg concurrent users |
| **Recent Requests Table** | Last 50 requests; UTC→IST timestamps; styled HTML table with SR No. |
| **Controls** | CSV download (`performance_metrics_requests.csv`) + Reset with confirmation |

---

## Thread Safety

All writes to the metrics JSON file are protected by a module-level `threading.Lock` (`_METRICS_LOCK`).  
This is safe for concurrent requests from multiple Streamlit sessions.

---

## Reset Metrics

From the dashboard:
1. Click **🔄 Reset Metrics** (popover button)
2. Check the confirmation checkbox
3. Click **🔴 Confirm Reset**

This overwrites the metrics file with an empty default payload.

---

*[← Back to README](../README.md)*
