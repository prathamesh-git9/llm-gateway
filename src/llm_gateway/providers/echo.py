"""Deterministic in-process provider.

Exists so the whole gateway — routing, breakers, cache, ledger, streaming — can
be exercised end to end in CI with no API keys and no network. Every test in
this repo runs against it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from llm_gateway.errors import ProviderOverloaded, ProviderTimeout, ProviderUnavailable
from llm_gateway.providers.base import ModelPricing, ModelSpec, estimate_tokens
from llm_gateway.schemas import ChatRequest, ChatResponse, Choice, Message, Usage

_SPECS = {
    "echo-fast": ModelSpec(
        name="echo-fast",
        provider="echo",
        pricing=ModelPricing(0.25, 1.25),
        context_window=128_000,
        latency_hint_ms=200,
    ),
    "echo-smart": ModelSpec(
        name="echo-smart",
        provider="echo",
        pricing=ModelPricing(3.0, 15.0),
        context_window=200_000,
        latency_hint_ms=900,
    ),
}

# Prompt prefixes that force a failure mode, so tests can drive the router's
# error paths without monkeypatching internals.
_FAULTS = {
    "__timeout__": ProviderTimeout,
    "__overloaded__": ProviderOverloaded,
    "__unavailable__": ProviderUnavailable,
}


class EchoProvider:
    name = "echo"

    def __init__(self, *, latency_ms: int = 0, fail_times: int = 0) -> None:
        self._latency_ms = latency_ms
        self._fail_times = fail_times
        self.calls = 0

    def supports(self, model: str) -> bool:
        return model in _SPECS

    def spec(self, model: str) -> ModelSpec:
        return _SPECS[model]

    def _maybe_fail(self, req: ChatRequest) -> None:
        self.calls += 1
        if self._fail_times > 0:
            self._fail_times -= 1
            raise ProviderUnavailable("injected failure", provider=self.name)
        last = req.messages[-1].content
        for marker, exc in _FAULTS.items():
            if last.startswith(marker):
                raise exc(f"injected {marker}", provider=self.name)

    def _reply(self, req: ChatRequest) -> str:
        return f"echo({req.model}): {req.messages[-1].content}"

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self._maybe_fail(req)
        if self._latency_ms:
            await asyncio.sleep(self._latency_ms / 1000)
        text = self._reply(req)
        prompt_tokens = estimate_tokens(req.prompt_text())
        completion_tokens = estimate_tokens(text)
        spec = self.spec(req.model)
        return ChatResponse(
            model=req.model,
            choices=[Choice(message=Message(role="assistant", content=text))],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=spec.pricing.cost(prompt_tokens, completion_tokens),
            ),
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[str]:
        self._maybe_fail(req)
        for token in self._reply(req).split(" "):
            if self._latency_ms:
                await asyncio.sleep(self._latency_ms / 1000)
            yield token + " "
