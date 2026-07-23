# llm-gateway
OpenAI-compatible LLM inference gateway with policy routing, fallback chains, circuit breakers, semantic caching, rate limiting, metrics, and hard per-tenant budgets.

## Why It Exists
LLM applications often start with a direct call to one provider, then quickly need operational controls: failover when a model is degraded, spend visibility by tenant, request throttling, caching, and Prometheus metrics. `llm-gateway` centralizes those concerns behind an OpenAI-compatible API surface so clients can keep a stable integration while the gateway handles routing and reliability policy.

## Architecture
Request flow:

```text
+--------+      +------------+      +-------+      +--------+      +---------+      +----------+
| client | ---> | rate limit | ---> | cache | ---> | router | ---> | breaker | ---> | provider |
+--------+      +------------+      +-------+      +--------+      +---------+      +----------+
                                      |  ^             |
                                      |  |             v
                                      |  +------ fallback chains
                                      v
                                exact + vector cache
```

Core modules:

- `src/llm_gateway/app.py`: FastAPI application and OpenAI-compatible endpoints.
- `src/llm_gateway/routing/router.py`: policy routing and fallback chains.
- `src/llm_gateway/routing/breaker.py`: provider circuit breaker.
- `src/llm_gateway/cache/semantic.py`: two-tier exact and vector cache.
- `src/llm_gateway/ledger.py`: concurrent cost reservations and per-tenant accounting.
- `src/llm_gateway/ratelimit.py`: token bucket rate limiting.
- `src/llm_gateway/metrics.py`: Prometheus instrumentation.
- `src/llm_gateway/providers/echo.py`: no-network built-in provider for local development and tests.
- `src/llm_gateway/providers/anthropic_provider.py`: Anthropic provider adapter.

## Features
- OpenAI-compatible `/v1/chat/completions` endpoint.
- Policy-based model routing with ordered fallback chains.
- Circuit breaker around provider calls to avoid repeatedly selecting unhealthy providers.
- Two-tier cache with exact lookup and vector similarity matching.
- Per-tenant spend ledger for cost attribution and atomic worst-case reservations
  that prevent concurrent requests or fallbacks from oversubscribing a hard budget.
- Token bucket rate limiting with configurable refill and burst capacity.
- Prometheus metrics endpoint.
- Built-in `echo-fast` provider for local, no-network verification.

## Quickstart
Install for development:

```bash
pip install -e .[dev]
```

Run the gateway:

```bash
uvicorn llm_gateway.app:app --reload
```

Send a chat completion request:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "model": "echo-fast",
    "messages": [
      {"role": "user", "content": "Say hello from the gateway"}
    ]
  }'
```

## Endpoints
| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion request path. |
| `GET` | `/healthz` | Liveness check for the running process. |
| `GET` | `/readyz` | Readiness check for serving traffic. |
| `GET` | `/v1/spend` | Per-tenant spend and usage accounting. |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint. |

## Configuration
Environment variables use the `GATEWAY_` prefix.

| Variable | Description | Example |
| --- | --- | --- |
| `GATEWAY_POLICY` | Routing policy name or serialized policy configuration. | `fallback` |
| `GATEWAY_CACHE_ENABLED` | Enable exact and semantic cache lookup. | `true` |
| `GATEWAY_CACHE_TTL_SECONDS` | Cache entry lifetime in seconds. | `300` |
| `GATEWAY_CACHE_SIMILARITY_THRESHOLD` | Minimum vector similarity score for semantic cache hits. | `0.92` |
| `GATEWAY_BREAKER_FAILURE_THRESHOLD` | Consecutive provider failures before opening a breaker. | `5` |
| `GATEWAY_BREAKER_RECOVERY_SECONDS` | Seconds before an open breaker can probe recovery. | `30` |
| `GATEWAY_RATE_LIMIT_PER_SECOND` | Token bucket refill rate per tenant. | `10` |
| `GATEWAY_BUDGETS_USD` | JSON map of tenant IDs to hard USD ceilings. | `{"acme":25.0}` |
| `GATEWAY_RATE_LIMIT_BURST` | Maximum burst tokens per tenant. | `50` |
| `GATEWAY_PROVIDERS` | Enabled provider identifiers and routing targets. | `echo,anthropic` |
| `GATEWAY_MAX_ATTEMPTS` | Maximum provider attempts across fallback chains. | `3` |

## Testing
Run the test suite:

```bash
pytest -q
```

Run lint checks:

```bash
ruff check .
```

Install both runtime and development dependencies before running CI-equivalent checks:

```bash
pip install -e .[dev]
ruff check .
pytest -q
```

## Design Notes
- The gateway keeps the client-facing API stable while routing and provider policy evolve internally.
- Reliability controls are layered: rate limiting protects the gateway, cache reduces duplicate work, router policy selects candidates, and breakers suppress unhealthy providers.
- The `echo` provider is intentionally no-network so local development, tests, and container health checks do not require external credentials.
- Tenant accounting is handled inside the gateway so spend and usage can be observed consistently across providers.
