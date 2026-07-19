"""Prometheus instrumentation.

Buckets are chosen for LLM latency, which lives in the 100ms–60s range — the
prometheus_client defaults top out at 10s and would collapse most of the
interesting tail into +Inf.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60)

REQUESTS = Counter(
    "gateway_requests_total",
    "Chat completion requests handled.",
    ["model", "provider", "outcome"],
)

LATENCY = Histogram(
    "gateway_request_duration_seconds",
    "End-to-end request latency, including retries and fallbacks.",
    ["model", "provider"],
    buckets=LATENCY_BUCKETS,
)

CACHE_EVENTS = Counter(
    "gateway_cache_events_total",
    "Cache lookups by tier.",
    ["tier"],  # exact | semantic | miss
)

FALLBACKS = Counter(
    "gateway_fallbacks_total",
    "Times the router advanced to the next candidate after a failure.",
    ["from_provider", "to_provider"],
)

BREAKER_STATE = Gauge(
    "gateway_breaker_state",
    "Circuit breaker state per provider (0=closed, 1=half_open, 2=open).",
    ["provider"],
)

COST = Counter(
    "gateway_cost_usd_total",
    "Cumulative upstream spend in USD.",
    ["tenant", "model"],
)

TOKENS = Counter(
    "gateway_tokens_total",
    "Tokens consumed upstream.",
    ["model", "kind"],  # prompt | completion
)
