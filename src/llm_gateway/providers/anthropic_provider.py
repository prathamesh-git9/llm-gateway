"""Anthropic provider.

The SDK is an optional extra (`pip install llm-gateway[anthropic]`) so the core
gateway and its test suite stay installable without vendor packages. The import
happens in `__init__` rather than at module scope for the same reason: importing
this module must never be what breaks a deployment that only routes to OpenAI.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from llm_gateway.errors import (
    BadRequest,
    ProviderOverloaded,
    ProviderTimeout,
    ProviderUnavailable,
)
from llm_gateway.providers.base import ModelPricing, ModelSpec
from llm_gateway.schemas import ChatRequest, ChatResponse, Choice, Message, Usage

# USD per 1M tokens.
_SPECS = {
    "claude-opus-4-8": ModelSpec(
        "claude-opus-4-8", "anthropic", ModelPricing(5.0, 25.0), 1_000_000, 2400
    ),
    "claude-sonnet-5": ModelSpec(
        "claude-sonnet-5", "anthropic", ModelPricing(3.0, 15.0), 1_000_000, 1400
    ),
    "claude-haiku-4-5": ModelSpec(
        "claude-haiku-4-5", "anthropic", ModelPricing(1.0, 5.0), 200_000, 600
    ),
}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, timeout: float = 60.0) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "AnthropicProvider requires the 'anthropic' extra: "
                "pip install llm-gateway[anthropic]"
            ) from exc

        self._sdk = anthropic
        # Zero-arg construction lets the SDK resolve credentials itself
        # (ANTHROPIC_API_KEY, or an `ant auth login` profile).
        self._client = (
            anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)
            if api_key
            else anthropic.AsyncAnthropic(timeout=timeout)
        )

    def supports(self, model: str) -> bool:
        return model in _SPECS

    def spec(self, model: str) -> ModelSpec:
        return _SPECS[model]

    def _split(self, req: ChatRequest) -> tuple[str | None, list[dict]]:
        """Anthropic takes the system prompt as a top-level field, not a turn."""
        system = "\n".join(m.content for m in req.messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content}
            for m in req.messages
            if m.role != "system"
        ]
        return (system or None), turns

    def _translate(self, exc: Exception) -> Exception:
        sdk = self._sdk
        if isinstance(exc, sdk.RateLimitError):
            return ProviderOverloaded(str(exc), provider=self.name)
        if isinstance(exc, sdk.APITimeoutError):
            return ProviderTimeout(str(exc), provider=self.name)
        if isinstance(exc, sdk.APIStatusError):
            # 5xx and 529 (overloaded) are worth a sibling; 4xx is not — another
            # provider would reject the same payload the same way.
            if exc.status_code >= 500:
                return ProviderUnavailable(str(exc), provider=self.name)
            return BadRequest(str(exc), provider=self.name)
        if isinstance(exc, sdk.APIConnectionError):
            return ProviderUnavailable(str(exc), provider=self.name)
        return exc

    async def complete(self, req: ChatRequest) -> ChatResponse:
        system, turns = self._split(req)
        kwargs = {"model": req.model, "max_tokens": req.max_tokens, "messages": turns}
        if system:
            kwargs["system"] = system
        try:
            msg = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise self._translate(exc) from exc

        text = "".join(b.text for b in msg.content if b.type == "text")
        pricing = self.spec(req.model).pricing
        prompt_tokens = msg.usage.input_tokens
        completion_tokens = msg.usage.output_tokens
        return ChatResponse(
            model=msg.model,
            choices=[
                Choice(
                    message=Message(role="assistant", content=text),
                    finish_reason=msg.stop_reason or "stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=pricing.cost(prompt_tokens, completion_tokens),
            ),
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[str]:
        system, turns = self._split(req)
        kwargs = {"model": req.model, "max_tokens": req.max_tokens, "messages": turns}
        if system:
            kwargs["system"] = system
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for chunk in stream.text_stream:
                    yield chunk
        except Exception as exc:
            raise self._translate(exc) from exc
