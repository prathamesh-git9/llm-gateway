"""Two-tier response cache: exact hash, then vector similarity.

The exact tier is checked first because it is O(1) and catches the dominant
real-world case (identical retried prompts). The semantic tier only runs on an
exact miss, and only within the same model + tenant — a near-duplicate prompt
answered by a different model is not a valid cache hit.

The default embedder is a hashed bag-of-words, not a learned model. That is a
deliberate tradeoff: it is dependency-free and deterministic, which keeps CI
honest, and it catches reordering and whitespace-level variation. For production
semantic recall, inject a real embedder via the `embed` argument — the interface
is a single `str -> list[float]` callable.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from llm_gateway.schemas import ChatResponse

Embedder = Callable[[str], list[float]]

_DIM = 256


def hashed_embedding(text: str, dim: int = _DIM) -> list[float]:
    """L2-normalised hashed bag-of-words."""
    vec = [0.0] * dim
    for token in text.lower().split():
        idx = int.from_bytes(
            hashlib.blake2b(token.encode(), digest_size=4).digest(), "big"
        )
        vec[idx % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


@dataclass
class CacheEntry:
    key: str
    namespace: str
    embedding: list[float]
    response: ChatResponse
    expires_at: float


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.exact_hits + self.semantic_hits + self.misses
        return 0.0 if total == 0 else (self.exact_hits + self.semantic_hits) / total


class SemanticCache:
    def __init__(
        self,
        *,
        max_entries: int = 1024,
        ttl_seconds: float = 900.0,
        threshold: float = 0.95,
        embed: Embedder | None = None,
    ) -> None:
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_seconds
        self._threshold = threshold
        self._embed = embed or hashed_embedding
        self.stats = CacheStats()

    @staticmethod
    def _key(namespace: str, prompt: str) -> str:
        return hashlib.sha256(f"{namespace}\x00{prompt}".encode()).hexdigest()

    def _evict_expired(self, now: float) -> None:
        for k in [k for k, e in self._entries.items() if e.expires_at <= now]:
            del self._entries[k]

    def get(self, namespace: str, prompt: str) -> tuple[ChatResponse, str] | None:
        """Returns (response, tier) where tier is 'exact' or 'semantic'."""
        now = time.monotonic()
        self._evict_expired(now)

        key = self._key(namespace, prompt)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            self.stats.exact_hits += 1
            return entry.response, "exact"

        query = self._embed(prompt)
        best: CacheEntry | None = None
        best_score = self._threshold
        for candidate in self._entries.values():
            if candidate.namespace != namespace:
                continue
            score = cosine(query, candidate.embedding)
            if score >= best_score:
                best, best_score = candidate, score

        if best is not None:
            self._entries.move_to_end(best.key)
            self.stats.semantic_hits += 1
            return best.response, "semantic"

        self.stats.misses += 1
        return None

    def put(self, namespace: str, prompt: str, response: ChatResponse) -> None:
        now = time.monotonic()
        key = self._key(namespace, prompt)
        self._entries[key] = CacheEntry(
            key=key,
            namespace=namespace,
            embedding=self._embed(prompt),
            response=response,
            expires_at=now + self._ttl,
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
