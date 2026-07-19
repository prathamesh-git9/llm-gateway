"""FastAPI surface.

Routes:
    POST /v1/chat/completions   OpenAI-compatible, streaming or not
    GET  /healthz               liveness
    GET  /readyz                readiness (fails when every provider is tripped)
    GET  /v1/spend              per-tenant cost ledger
    GET  /metrics               Prometheus exposition
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from llm_gateway.cache.semantic import SemanticCache
from llm_gateway.config import Settings
from llm_gateway.errors import GatewayError, RateLimited
from llm_gateway.ledger import CostLedger
from llm_gateway.metrics import CACHE_EVENTS, COST, LATENCY, REQUESTS, TOKENS
from llm_gateway.providers.echo import EchoProvider
from llm_gateway.ratelimit import TokenBucketLimiter
from llm_gateway.routing.breaker import State
from llm_gateway.routing.router import Router
from llm_gateway.schemas import ChatRequest, ChatResponse

# Logical aliases. Callers ask for an intent; the gateway owns which model
# serves it, so a model swap is a config change rather than a client release.
DEFAULT_ALIASES = {
    "cheap": ["echo-fast"],
    "balanced": ["echo-smart", "echo-fast"],
    "smart": ["echo-smart", "echo-fast"],
}


def build_providers(settings: Settings) -> list:
    providers: list = []
    for name in settings.providers:
        if name == "echo":
            providers.append(EchoProvider())
        elif name == "anthropic":
            from llm_gateway.providers.anthropic_provider import AnthropicProvider

            providers.append(
                AnthropicProvider(
                    api_key=settings.anthropic_api_key,
                    timeout=settings.request_timeout_seconds,
                )
            )
        else:
            raise ValueError(f"unknown provider '{name}'")
    return providers


def create_app(settings: Settings | None = None, **overrides) -> FastAPI:
    settings = settings or Settings(**overrides)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.router = Router(
            build_providers(settings),
            aliases=DEFAULT_ALIASES,
            policy=settings.policy,
            failure_threshold=settings.breaker_failure_threshold,
            recovery_seconds=settings.breaker_recovery_seconds,
            max_attempts=settings.max_attempts,
        )
        app.state.cache = SemanticCache(
            max_entries=settings.cache_max_entries,
            ttl_seconds=settings.cache_ttl_seconds,
            threshold=settings.cache_similarity_threshold,
        )
        app.state.ledger = CostLedger()
        app.state.limiter = TokenBucketLimiter(
            settings.rate_limit_per_second, settings.rate_limit_burst
        )
        yield

    app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"type": type(exc).__name__, "message": exc.message}},
        )

    def get_state(request: Request):
        return request.app.state

    app_state = Depends(get_state)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(state=app_state) -> JSONResponse:
        breakers = state.router._breakers
        states = {name: b.state.value for name, b in breakers.items()}
        # Ready means at least one provider can still take traffic. All-open is
        # a real outage and should pull the instance from the load balancer.
        ready = any(s != State.OPEN.value for s in states.values())
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"ready": ready, "providers": states},
        )

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/spend")
    async def spend(state=app_state) -> dict:
        return {
            tenant: {
                "requests": s.requests,
                "cached_requests": s.cached_requests,
                "prompt_tokens": s.prompt_tokens,
                "completion_tokens": s.completion_tokens,
                "cost_usd": round(s.cost_usd, 6),
                "saved_usd": round(s.saved_usd, 6),
            }
            for tenant, s in state.ledger.all_spend().items()
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest, state=app_state):
        if not state.limiter.allow(req.x_tenant):
            raise RateLimited(f"rate limit exceeded for tenant '{req.x_tenant}'")
        state.ledger.check_budget(req.x_tenant)

        if req.stream:
            return await _stream(req, state)
        return await _complete(req, state)

    async def _complete(req: ChatRequest, state) -> ChatResponse:
        namespace = f"{req.x_tenant}:{req.model}"
        use_cache = state.settings.cache_enabled and req.x_cache

        if use_cache:
            hit = state.cache.get(namespace, req.prompt_text())
            if hit is not None:
                cached, tier = hit
                CACHE_EVENTS.labels(tier=tier).inc()
                REQUESTS.labels(
                    model=req.model, provider="cache", outcome="hit"
                ).inc()
                state.ledger.record(
                    req.x_tenant,
                    prompt_tokens=cached.usage.prompt_tokens,
                    completion_tokens=cached.usage.completion_tokens,
                    cost_usd=cached.usage.cost_usd,
                    cached=True,
                )
                # Never hand back the stored object: callers mutating the
                # gateway block would corrupt every later hit on this entry.
                return cached.model_copy(
                    update={
                        "gateway": {
                            "cache": tier,
                            "cost_saved_usd": cached.usage.cost_usd,
                        }
                    }
                )
            CACHE_EVENTS.labels(tier="miss").inc()

        try:
            result = await state.router.route(req)
        except GatewayError:
            REQUESTS.labels(
                model=req.model, provider="none", outcome="error"
            ).inc()
            raise

        usage = result.response.usage
        LATENCY.labels(model=result.model, provider=result.provider).observe(
            result.latency_ms / 1000
        )
        REQUESTS.labels(
            model=req.model, provider=result.provider, outcome="success"
        ).inc()
        COST.labels(tenant=req.x_tenant, model=result.model).inc(usage.cost_usd)
        TOKENS.labels(model=result.model, kind="prompt").inc(usage.prompt_tokens)
        TOKENS.labels(model=result.model, kind="completion").inc(
            usage.completion_tokens
        )
        state.ledger.record(
            req.x_tenant,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            cached=False,
        )

        response = result.response.model_copy(
            update={
                "gateway": {
                    "cache": "miss",
                    "provider": result.provider,
                    "model": result.model,
                    "attempts": result.attempts,
                    "fallback_used": result.fallback_used,
                    "latency_ms": round(result.latency_ms, 2),
                }
            }
        )
        if use_cache:
            state.cache.put(f"{req.x_tenant}:{req.model}", req.prompt_text(), response)
        return response

    async def _stream(req: ChatRequest, state) -> StreamingResponse:
        chain = state.router.candidates(req.model)
        if not chain:
            raise GatewayError(f"no provider serves model '{req.model}'")
        candidate = chain[0]
        upstream = req.model_copy(update={"model": candidate.spec.name})

        async def sse():
            # Streaming intentionally does not fall back mid-response: once the
            # first token is on the wire, switching providers would splice two
            # different completions together. Failures surface as an SSE error
            # event and the client decides whether to retry.
            try:
                async for chunk in candidate.provider.stream(upstream):
                    payload = {
                        "choices": [{"index": 0, "delta": {"content": chunk}}],
                        "model": candidate.spec.name,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            except GatewayError as exc:
                yield f"data: {json.dumps({'error': exc.message})}\n\n"
            yield "data: [DONE]\n\n"

        REQUESTS.labels(
            model=req.model, provider=candidate.provider.name, outcome="stream"
        ).inc()
        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


app = create_app()
