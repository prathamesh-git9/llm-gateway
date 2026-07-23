"""Environment-driven configuration."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_gateway.routing.router import Policy


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_", env_file=".env", extra="ignore"
    )

    policy: Policy = Policy.ORDERED
    max_attempts: int = 3

    cache_enabled: bool = True
    cache_max_entries: int = 1024
    cache_ttl_seconds: float = 900.0
    cache_similarity_threshold: float = 0.95

    breaker_failure_threshold: int = 5
    breaker_recovery_seconds: float = 30.0

    rate_limit_per_second: float = 20.0
    rate_limit_burst: int = 40

    # JSON environment value, e.g. GATEWAY_BUDGETS_USD='{"acme": 25.0}'.
    # Reservations make this a concurrent hard ceiling, not an alert after spend.
    budgets_usd: dict[str, float] = Field(default_factory=dict)

    # Enabled providers. 'echo' is the built-in no-network provider and is the
    # default so a fresh clone boots and serves traffic with zero credentials.
    providers: list[str] = Field(default_factory=lambda: ["echo"])

    anthropic_api_key: str | None = None
    request_timeout_seconds: float = 60.0
