"""Policy-based routing with fallback chains.

A request names either a concrete model or a logical alias ("cheap", "balanced",
"smart"). Aliases expand to an ordered candidate list under a policy; concrete
models expand to themselves plus any configured siblings. The router then walks
that list, skipping tripped providers, and stops at the first success.

Only `retryable` failures advance the chain. A 400 means the payload is wrong,
and burning three more providers on it just multiplies the latency of a request
that was always going to fail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from llm_gateway.errors import GatewayError, NoRouteAvailable
from llm_gateway.metrics import BREAKER_STATE, FALLBACKS
from llm_gateway.providers.base import ModelSpec, Provider
from llm_gateway.routing.breaker import CircuitBreaker, State
from llm_gateway.schemas import ChatRequest, ChatResponse


class Policy(StrEnum):
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    ORDERED = "ordered"  # honour the configured order verbatim


@dataclass
class Candidate:
    provider: Provider
    spec: ModelSpec


@dataclass
class RouteResult:
    response: ChatResponse
    provider: str
    model: str
    attempts: int
    fallback_used: bool
    latency_ms: float


@dataclass
class _LatencyTracker:
    """Exponentially weighted moving average of observed latency.

    Static latency hints go stale the moment a provider degrades; this lets
    `Policy.FASTEST` react to reality instead of to a constant in a table.
    """

    alpha: float = 0.3
    _values: dict[str, float] = field(default_factory=dict)

    def observe(self, key: str, ms: float) -> None:
        prev = self._values.get(key)
        self._values[key] = ms if prev is None else (
            self.alpha * ms + (1 - self.alpha) * prev
        )

    def get(self, key: str, default: float) -> float:
        return self._values.get(key, default)


class Router:
    def __init__(
        self,
        providers: list[Provider],
        *,
        aliases: dict[str, list[str]] | None = None,
        policy: Policy = Policy.ORDERED,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        max_attempts: int = 3,
    ) -> None:
        self._providers = providers
        self._aliases = aliases or {}
        self._policy = policy
        self._max_attempts = max_attempts
        self._breakers: dict[str, CircuitBreaker] = {
            p.name: CircuitBreaker(failure_threshold, recovery_seconds)
            for p in providers
        }
        self.latency = _LatencyTracker()

    # -- candidate selection -------------------------------------------------

    def _resolve(self, model: str) -> list[str]:
        return self._aliases.get(model, [model])

    def candidates(self, model: str) -> list[Candidate]:
        out: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        # Model-major: the alias expresses model preference, so exhaust every
        # provider serving the preferred model before demoting to the next one.
        # Listing *all* matching providers (rather than the first) is what makes
        # provider-level redundancy possible — the same model reachable through
        # two vendors is the most common real fallback pair.
        for name in self._resolve(model):
            for provider in self._providers:
                if not provider.supports(name):
                    continue
                key = (provider.name, name)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Candidate(provider, provider.spec(name)))

        if self._policy is Policy.CHEAPEST:
            # Rank on blended cost. Output tokens are weighted 3x because a
            # completion-heavy workload is what actually moves the bill.
            out.sort(
                key=lambda c: c.spec.pricing.input_per_mtok
                + 3 * c.spec.pricing.output_per_mtok
            )
        elif self._policy is Policy.FASTEST:
            out.sort(
                key=lambda c: self.latency.get(
                    f"{c.spec.provider}:{c.spec.name}", float(c.spec.latency_hint_ms)
                )
            )
        return out

    # -- execution -----------------------------------------------------------

    def _sync_breaker_gauge(self, name: str) -> None:
        value = {State.CLOSED: 0, State.HALF_OPEN: 1, State.OPEN: 2}[
            self._breakers[name].state
        ]
        BREAKER_STATE.labels(provider=name).set(value)

    async def route(self, req: ChatRequest) -> RouteResult:
        chain = self.candidates(req.model)
        if not chain:
            raise NoRouteAvailable(f"no provider serves model '{req.model}'")

        started = time.monotonic()
        attempts = 0
        last_error: Exception | None = None
        previous_provider: str | None = None

        for candidate in chain[: self._max_attempts]:
            name = candidate.provider.name
            breaker = self._breakers[name]
            self._sync_breaker_gauge(name)

            if not breaker.allow():
                last_error = last_error or NoRouteAvailable(
                    f"circuit open for provider '{name}'"
                )
                continue

            attempts += 1
            if previous_provider is not None:
                FALLBACKS.labels(
                    from_provider=previous_provider, to_provider=name
                ).inc()

            attempt_started = time.monotonic()
            # Route on the candidate's own model name — an alias must not be
            # forwarded upstream, and a fallback may be a different model.
            upstream = req.model_copy(update={"model": candidate.spec.name})
            try:
                response = await candidate.provider.complete(upstream)
            except GatewayError as exc:
                breaker.record_failure()
                self._sync_breaker_gauge(name)
                last_error = exc
                previous_provider = name
                if not exc.retryable:
                    raise
                continue

            elapsed_ms = (time.monotonic() - attempt_started) * 1000
            breaker.record_success()
            self._sync_breaker_gauge(name)
            self.latency.observe(f"{name}:{candidate.spec.name}", elapsed_ms)

            return RouteResult(
                response=response,
                provider=name,
                model=candidate.spec.name,
                attempts=attempts,
                fallback_used=attempts > 1 or candidate is not chain[0],
                latency_ms=(time.monotonic() - started) * 1000,
            )

        raise NoRouteAvailable(
            f"all {len(chain)} candidate(s) for '{req.model}' failed: {last_error}"
        ) from last_error
