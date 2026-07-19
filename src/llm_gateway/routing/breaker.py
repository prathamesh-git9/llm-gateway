"""Per-provider circuit breaker.

Standard three-state breaker. The half-open state admits exactly one probe so a
recovering provider isn't immediately re-buried by the traffic that tripped it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_seconds: float = 30.0
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _state: State = field(default=State.CLOSED, init=False)
    _probe_in_flight: bool = field(default=False, init=False)

    @property
    def state(self) -> State:
        if self._state is State.OPEN and self._elapsed() >= self.recovery_seconds:
            self._state = State.HALF_OPEN
            self._probe_in_flight = False
        return self._state

    def _elapsed(self) -> float:
        return time.monotonic() - self._opened_at

    def allow(self) -> bool:
        state = self.state
        if state is State.CLOSED:
            return True
        if state is State.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._state = State.CLOSED
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self._probe_in_flight = False
        if self._state is State.HALF_OPEN:
            # The probe failed — go straight back to open, don't count up again.
            self._trip()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = State.OPEN
        self._opened_at = time.monotonic()
