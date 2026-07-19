"""Provider protocol and shared pricing/token accounting."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from llm_gateway.schemas import ChatRequest, ChatResponse


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens, matching how vendors publish rates."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_mtok
            + completion_tokens * self.output_per_mtok
        ) / 1_000_000


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    pricing: ModelPricing
    context_window: int
    # Rough p50 in ms. Used for latency-weighted routing, refreshed from
    # observed latency at runtime — the static value is only a cold-start prior.
    latency_hint_ms: int = 1500


@runtime_checkable
class Provider(Protocol):
    name: str

    def supports(self, model: str) -> bool: ...

    def spec(self, model: str) -> ModelSpec: ...

    async def complete(self, req: ChatRequest) -> ChatResponse: ...

    async def stream(self, req: ChatRequest) -> AsyncIterator[str]: ...


def estimate_tokens(text: str) -> int:
    """Cheap character-based estimate.

    Deliberately not a real tokenizer: this is only used for pre-flight budget
    checks and for providers that don't return usage. Anything billed against
    it would be wrong, so `cost` always prefers provider-reported usage.
    """
    return max(1, len(text) // 4)
