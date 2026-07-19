"""Provider-agnostic error taxonomy.

The router only needs to know one thing about a failure: is it worth trying
somewhere else? `retryable` encodes exactly that, so routing logic never has to
pattern-match on provider-specific status codes.
"""

from __future__ import annotations


class GatewayError(Exception):
    retryable = False
    status_code = 500

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider


class ProviderTimeout(GatewayError):
    retryable = True
    status_code = 504


class ProviderOverloaded(GatewayError):
    """429 / 529 — the provider is up but shedding load. Try a sibling."""

    retryable = True
    status_code = 429


class ProviderUnavailable(GatewayError):
    """5xx or transport failure."""

    retryable = True
    status_code = 502


class BadRequest(GatewayError):
    """4xx that a different provider would reject identically. Never retried."""

    retryable = False
    status_code = 400


class NoRouteAvailable(GatewayError):
    """Every candidate was tripped, rate limited, or exhausted."""

    retryable = False
    status_code = 503


class RateLimited(GatewayError):
    retryable = False
    status_code = 429
