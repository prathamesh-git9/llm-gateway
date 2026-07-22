# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-21

### Added

- OpenAI-compatible `POST /v1/chat/completions` endpoint with non-streaming and
  server-sent event streaming responses.
- Policy-based routing for concrete models and logical aliases, with ordered,
  cheapest, and fastest candidate selection.
- Ordered fallback chains across providers and models, including retryable error
  handling and maximum-attempt limits.
- Per-provider circuit breakers with closed, open, and half-open states.
- Two-tier response cache with exact hashing followed by same-namespace vector
  similarity lookup.
- Per-tenant spend ledger with request counts, cached request counts, token
  usage, billed cost, and avoided-cost accounting.
- Token bucket rate limiting keyed by tenant.
- Prometheus metrics for requests, latency, token usage, cost, cache events,
  fallbacks, and breaker state.
- Health, readiness, metrics, and spend endpoints.
- Built-in deterministic `echo` provider for local development, tests, and
  no-network verification.
- Optional Anthropic provider adapter loaded only when configured.
