# Security Policy

## Scope and threat model

`llm-gateway` is a self-hostable inference gateway that sits between tenants and
upstream LLM providers. Its security boundary is the gateway process and the
deployment controls around it: authentication, network exposure, TLS,
configuration, and secret storage remain the operator's responsibility.

The main risks this project is designed to help operators manage are:

- Cross-tenant data exposure through shared gateway state.
- Denial-of-wallet from one tenant consuming another tenant's provider budget.
- Upstream API key disclosure or accidental logging.
- Provider outages or overloads cascading into every request.

## Multi-tenant isolation

Requests carry tenant identity through the gateway-specific `x_tenant` field.
The cost ledger records spend per tenant, and budget checks can reject traffic
once a tenant reaches its configured spend limit. This is primarily a
denial-of-wallet control: one tenant's usage should be observable and limitable
without blending it into another tenant's bill.

Deployments should bind `x_tenant` to an authenticated caller at the edge. Do not
let untrusted clients choose arbitrary tenant IDs without authentication and
authorization checks in front of the gateway.

## Semantic cache isolation

The semantic cache is shared process memory, so cache keys must be scoped by
tenant and model intent. The application namespaces cache entries as
`tenant:model` before exact hashing or semantic lookup. This prevents a prompt
from tenant A from returning tenant B's cached response, even if the prompt text
is identical or semantically similar.

If you change cache keying, embeddings, persistence, or distributed cache
backends, preserve tenant scoping as a hard invariant. Treat unscoped semantic
similarity search as a vulnerability.

## Upstream API keys and secrets

Provider credentials, such as `GATEWAY_ANTHROPIC_API_KEY`, are read from
environment-driven settings and passed only to the provider adapter. Do not put
provider keys in source control, tests, logs, metrics labels, cache entries, or
client-visible response metadata.

Recommended deployment practices:

- Inject secrets through your platform's secret manager or environment system.
- Rotate keys after suspected exposure.
- Scope provider keys to the minimum project/account permissions available.
- Avoid exposing `/metrics` or `/v1/spend` publicly unless protected by your own
  authentication layer.

## Circuit breakers and resilience

Each provider has its own circuit breaker. Repeated retryable failures open that
provider's breaker, routing avoids open providers, and half-open probes allow
recovery checks without immediately sending full traffic back to a degraded
upstream.

Circuit breakers are resilience controls, not access controls. They reduce blast
radius from provider outages and overloads, but they do not replace rate limits,
tenant budgets, authentication, or provider-side quota controls.

## Reporting a vulnerability

Please report vulnerabilities privately using a GitHub private security advisory
for this repository. Do not open a public issue for suspected security problems.

Include:

- Affected version or commit.
- Steps to reproduce or a proof of concept.
- Expected and observed impact.
- Any relevant deployment assumptions.

We will triage reports privately and coordinate disclosure once a fix or
mitigation is available.
