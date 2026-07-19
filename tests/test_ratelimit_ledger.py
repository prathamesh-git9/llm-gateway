import pytest

from llm_gateway.errors import RateLimited
from llm_gateway.ledger import CostLedger
from llm_gateway.ratelimit import TokenBucketLimiter


def test_burst_then_throttle(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr("llm_gateway.ratelimit.time.monotonic", lambda: clock["now"])
    limiter = TokenBucketLimiter(rate_per_second=1, burst=3)

    assert all(limiter.allow("t") for _ in range(3))
    assert not limiter.allow("t")

    clock["now"] += 2
    assert limiter.allow("t")
    assert limiter.allow("t")
    assert not limiter.allow("t")


def test_refill_is_capped_at_burst(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr("llm_gateway.ratelimit.time.monotonic", lambda: clock["now"])
    limiter = TokenBucketLimiter(rate_per_second=10, burst=5)
    limiter.allow("t")

    clock["now"] += 3600  # idle for an hour
    assert limiter.tokens_remaining("t") <= 5


def test_tenants_have_independent_buckets():
    limiter = TokenBucketLimiter(rate_per_second=1, burst=1)
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        TokenBucketLimiter(rate_per_second=0, burst=1)


def test_budget_enforcement():
    ledger = CostLedger(budgets_usd={"acme": 0.01})
    ledger.check_budget("acme")  # under budget, no raise

    ledger.record(
        "acme", prompt_tokens=100, completion_tokens=100, cost_usd=0.02, cached=False
    )
    with pytest.raises(RateLimited):
        ledger.check_budget("acme")


def test_cached_requests_accrue_savings_not_spend():
    ledger = CostLedger()
    ledger.record("t", prompt_tokens=10, completion_tokens=10, cost_usd=0.5, cached=True)
    spend = ledger.spend("t")
    assert spend.cost_usd == 0
    assert spend.saved_usd == 0.5
    assert spend.cached_requests == 1
