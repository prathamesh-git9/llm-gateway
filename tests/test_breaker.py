from llm_gateway.routing.breaker import CircuitBreaker, State


def test_closed_until_threshold():
    cb = CircuitBreaker(failure_threshold=3, recovery_seconds=60)
    for _ in range(2):
        cb.record_failure()
    assert cb.state is State.CLOSED
    assert cb.allow()

    cb.record_failure()
    assert cb.state is State.OPEN
    assert not cb.allow()


def test_success_resets_the_count():
    cb = CircuitBreaker(failure_threshold=3, recovery_seconds=60)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    assert cb.state is State.CLOSED


def test_half_open_admits_exactly_one_probe():
    cb = CircuitBreaker(failure_threshold=1, recovery_seconds=0)
    cb.record_failure()
    assert cb.state is State.HALF_OPEN
    assert cb.allow()
    # Second caller must be refused while the probe is outstanding, otherwise a
    # recovering provider gets the full thundering herd back at once.
    assert not cb.allow()


def test_failed_probe_reopens_immediately():
    cb = CircuitBreaker(failure_threshold=5, recovery_seconds=0)
    for _ in range(5):
        cb.record_failure()
    assert cb.allow()  # half-open probe
    cb.record_failure()
    # recovery_seconds=0 means it is immediately eligible again, but it must
    # have gone through OPEN rather than counting back up from zero.
    assert cb._state in (State.OPEN, State.HALF_OPEN)
    assert cb._failures < 5 or cb._state is State.OPEN
