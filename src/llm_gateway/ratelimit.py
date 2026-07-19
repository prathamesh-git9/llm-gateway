"""Token-bucket rate limiter, keyed per tenant.

Monotonic-clock lazy refill: no background task, no drift from wall-clock jumps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Bucket:
    tokens: float
    updated_at: float


class TokenBucketLimiter:
    def __init__(self, rate_per_second: float, burst: int) -> None:
        if rate_per_second <= 0 or burst <= 0:
            raise ValueError("rate_per_second and burst must be positive")
        self._rate = rate_per_second
        self._burst = float(burst)
        self._buckets: dict[str, Bucket] = {}

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = Bucket(tokens=self._burst, updated_at=now)
            self._buckets[key] = bucket

        elapsed = now - bucket.updated_at
        bucket.tokens = min(self._burst, bucket.tokens + elapsed * self._rate)
        bucket.updated_at = now

        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True
        return False

    def tokens_remaining(self, key: str) -> float:
        bucket = self._buckets.get(key)
        return self._burst if bucket is None else bucket.tokens
