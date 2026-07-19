import pytest

from llm_gateway.errors import BadRequest, NoRouteAvailable, ProviderUnavailable
from llm_gateway.providers.base import ModelPricing, ModelSpec
from llm_gateway.providers.echo import EchoProvider
from llm_gateway.routing.router import Policy, Router
from llm_gateway.schemas import ChatRequest, Message

ALIASES = {"balanced": ["echo-smart", "echo-fast"]}


def request(model="echo-fast", text="hello"):
    return ChatRequest(model=model, messages=[Message(role="user", content=text)])


class AlwaysFails(EchoProvider):
    name = "flaky"

    async def complete(self, req):
        raise ProviderUnavailable("down", provider=self.name)


class RejectsPayload(EchoProvider):
    name = "strict"

    async def complete(self, req):
        raise BadRequest("bad payload", provider=self.name)


async def test_happy_path():
    router = Router([EchoProvider()], aliases=ALIASES)
    result = await router.route(request())
    assert result.provider == "echo"
    assert result.attempts == 1
    assert not result.fallback_used
    assert "hello" in result.response.choices[0].message.content


async def test_unknown_model_has_no_route():
    router = Router([EchoProvider()])
    with pytest.raises(NoRouteAvailable):
        await router.route(request(model="gpt-nonexistent"))


async def test_falls_back_to_second_candidate():
    healthy = EchoProvider()
    router = Router([AlwaysFails(), healthy], aliases=ALIASES)
    result = await router.route(request(model="balanced"))
    assert result.provider == "echo"
    assert result.attempts == 2
    assert result.fallback_used


async def test_same_model_is_reachable_through_a_second_provider():
    # Provider-level redundancy: one vendor is down, but another serves the
    # identical model. A router that let the first matching provider own a
    # model outright could never express this, which is the most common real
    # fallback pair (e.g. a vendor's own API plus the same model on a cloud).
    healthy = EchoProvider()
    router = Router([AlwaysFails(), healthy], aliases={"only": ["echo-fast"]})
    result = await router.route(request(model="only"))
    assert result.provider == "echo"
    assert result.model == "echo-fast"
    assert result.fallback_used


async def test_candidates_are_ordered_model_major():
    router = Router(
        [AlwaysFails(), EchoProvider()],
        aliases={"balanced": ["echo-smart", "echo-fast"]},
    )
    chain = [(c.provider.name, c.spec.name) for c in router.candidates("balanced")]
    # Exhaust every provider for the preferred model before demoting the model.
    assert chain == [
        ("flaky", "echo-smart"),
        ("echo", "echo-smart"),
        ("flaky", "echo-fast"),
        ("echo", "echo-fast"),
    ]


async def test_non_retryable_error_stops_the_chain():
    healthy = EchoProvider()
    router = Router([RejectsPayload(), healthy], aliases=ALIASES)
    with pytest.raises(BadRequest):
        await router.route(request(model="balanced"))
    # The healthy provider must never have been called: a 400 is deterministic
    # and retrying it elsewhere only multiplies latency.
    assert healthy.calls == 0


async def test_breaker_opens_and_stops_dialling_a_dead_provider():
    flaky = AlwaysFails()
    router = Router([flaky], failure_threshold=2, recovery_seconds=999)
    for _ in range(2):
        with pytest.raises(NoRouteAvailable):
            await router.route(request())
    calls_after_trip = flaky.calls

    with pytest.raises(NoRouteAvailable):
        await router.route(request())
    assert flaky.calls == calls_after_trip  # short-circuited, no upstream call


async def test_cheapest_policy_orders_by_blended_price():
    router = Router(
        [EchoProvider()],
        aliases={"any": ["echo-smart", "echo-fast"]},
        policy=Policy.CHEAPEST,
    )
    chain = router.candidates("any")
    assert [c.spec.name for c in chain] == ["echo-fast", "echo-smart"]


async def test_fastest_policy_uses_observed_latency_over_the_static_hint():
    router = Router(
        [EchoProvider()],
        aliases={"any": ["echo-fast", "echo-smart"]},
        policy=Policy.FASTEST,
    )
    assert router.candidates("any")[0].spec.name == "echo-fast"

    # echo-fast degrades in production; the router must notice and reorder.
    for _ in range(10):
        router.latency.observe("echo:echo-fast", 5000)
    assert router.candidates("any")[0].spec.name == "echo-smart"


def test_pricing_math():
    pricing = ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)
    assert pricing.cost(1_000_000, 0) == pytest.approx(3.0)
    assert pricing.cost(0, 1_000_000) == pytest.approx(15.0)
    assert pricing.cost(1000, 500) == pytest.approx(0.0105)


def test_model_spec_is_hashable_config():
    spec = ModelSpec("m", "p", ModelPricing(1, 2), 1000)
    assert spec.latency_hint_ms == 1500
