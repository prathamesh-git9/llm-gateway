import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Settings


@pytest.fixture
def client():
    settings = Settings(
        providers=["echo"], rate_limit_per_second=1000, rate_limit_burst=1000
    )
    with TestClient(create_app(settings)) as c:
        yield c


def post(client, **kwargs):
    body = {"model": "echo-fast", "messages": [{"role": "user", "content": "hi"}]}
    body.update(kwargs)
    return client.post("/v1/chat/completions", json=body)


def test_health_and_readiness(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True


def test_completion_shape_is_openai_compatible(client):
    body = post(client).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["total_tokens"] > 0
    assert body["gateway"]["provider"] == "echo"
    assert body["gateway"]["cache"] == "miss"


def test_second_identical_request_is_served_from_cache(client):
    post(client, messages=[{"role": "user", "content": "cache me please"}])
    second = post(client, messages=[{"role": "user", "content": "cache me please"}])
    assert second.json()["gateway"]["cache"] == "exact"


def test_cache_can_be_bypassed_per_request(client):
    payload = [{"role": "user", "content": "no cache for me"}]
    post(client, messages=payload)
    second = post(client, messages=payload, x_cache=False)
    assert second.json()["gateway"]["cache"] == "miss"


def test_alias_resolves_to_a_concrete_model(client):
    body = post(client, model="balanced").json()
    assert body["gateway"]["model"] == "echo-smart"


def test_validation_error_is_rejected(client):
    resp = client.post(
        "/v1/chat/completions", json={"model": "echo-fast", "messages": []}
    )
    assert resp.status_code == 422


def test_upstream_failure_surfaces_as_5xx(client):
    resp = post(client, messages=[{"role": "user", "content": "__unavailable__ boom"}])
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "NoRouteAvailable"
    # A failed provider call releases its reservation and records no billed tenant.
    assert "default" not in client.get("/v1/spend").json()


def test_rate_limit_returns_429():
    settings = Settings(providers=["echo"], rate_limit_per_second=1, rate_limit_burst=2)
    with TestClient(create_app(settings)) as c:
        for _ in range(2):
            post(c, x_cache=False)
        resp = post(c, x_cache=False)
        assert resp.status_code == 429


def test_hard_budget_rejects_before_provider_call():
    settings = Settings(
        providers=["echo"],
        rate_limit_per_second=1000,
        rate_limit_burst=1000,
        budgets_usd={"small": 0.000001},
    )
    with TestClient(create_app(settings)) as c:
        response = post(c, x_tenant="small", x_cache=False)
        assert response.status_code == 429
        spend = c.get("/v1/spend").json()["small"]
        assert spend["cost_usd"] == 0
        assert spend["reserved_usd"] == 0


def test_spend_ledger_tracks_cost_and_savings(client):
    payload = [{"role": "user", "content": "ledger check"}]
    post(client, messages=payload)
    post(client, messages=payload)  # cache hit

    spend = client.get("/v1/spend").json()["default"]
    assert spend["requests"] == 2
    assert spend["cached_requests"] == 1
    assert spend["cost_usd"] > 0
    assert spend["saved_usd"] > 0


def test_streaming_emits_sse_and_terminates(client):
    resp = post(client, stream=True)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body = resp.text
    assert body.startswith("data: ")
    assert body.rstrip().endswith("data: [DONE]")
    spend = client.get("/v1/spend").json()["default"]
    assert spend["cost_usd"] > 0
    assert spend["reserved_usd"] == 0


def test_metrics_are_exposed(client):
    post(client)
    metrics = client.get("/metrics").text
    assert "gateway_requests_total" in metrics
    assert "gateway_cost_usd_total" in metrics
