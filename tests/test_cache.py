import pytest

from llm_gateway.cache.semantic import SemanticCache, cosine, hashed_embedding
from llm_gateway.schemas import ChatResponse, Choice, Message


def make_response(text: str = "hi") -> ChatResponse:
    return ChatResponse(
        model="echo-fast",
        choices=[Choice(message=Message(role="assistant", content=text))],
    )


def test_exact_hit():
    cache = SemanticCache()
    cache.put("t:m", "what is the capital of france", make_response("Paris"))
    hit = cache.get("t:m", "what is the capital of france")
    assert hit is not None
    response, tier = hit
    assert tier == "exact"
    assert response.choices[0].message.content == "Paris"
    assert cache.stats.exact_hits == 1


def test_semantic_hit_on_reordered_prompt():
    cache = SemanticCache(threshold=0.9)
    cache.put("t:m", "deploy the staging cluster now", make_response("ok"))
    hit = cache.get("t:m", "now deploy the staging cluster")
    assert hit is not None
    assert hit[1] == "semantic"


def test_namespace_isolation():
    # A near-identical prompt from another tenant must never hit. This is the
    # cache's security boundary, not just a correctness detail.
    cache = SemanticCache(threshold=0.5)
    cache.put("tenant-a:m", "summarise the contract", make_response("secret"))
    assert cache.get("tenant-b:m", "summarise the contract") is None


def test_miss_below_threshold():
    cache = SemanticCache(threshold=0.99)
    cache.put("t:m", "how do i reset my password", make_response("a"))
    assert cache.get("t:m", "what is quantum tunnelling") is None
    assert cache.stats.misses == 1


def test_lru_eviction_respects_capacity():
    cache = SemanticCache(max_entries=2, threshold=1.1)  # threshold disables fuzzy hits
    for i in range(5):
        cache.put("t:m", f"prompt number {i}", make_response(str(i)))
    assert len(cache) == 2


def test_ttl_expiry(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        "llm_gateway.cache.semantic.time.monotonic", lambda: clock["now"]
    )
    cache = SemanticCache(ttl_seconds=60, threshold=1.1)
    cache.put("t:m", "hello there", make_response())
    assert cache.get("t:m", "hello there") is not None

    clock["now"] += 61
    assert cache.get("t:m", "hello there") is None


def test_role_prefix_does_not_fuse_into_the_first_token():
    # Regression: `user:what ...` made the role part of the first word, so a
    # prompt and its own reordering lost two terms and fell under threshold.
    from llm_gateway.schemas import ChatRequest

    a = ChatRequest(
        model="m", messages=[Message(role="user", content="what is a circuit breaker")]
    )
    b = ChatRequest(
        model="m", messages=[Message(role="user", content="circuit breaker a is what")]
    )
    assert cosine(
        hashed_embedding(a.prompt_text()), hashed_embedding(b.prompt_text())
    ) == pytest.approx(1.0)


def test_embedding_is_normalised():
    vec = hashed_embedding("alpha beta gamma")
    assert abs(cosine(vec, vec) - 1.0) < 1e-9
