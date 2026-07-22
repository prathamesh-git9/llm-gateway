# Contributing

## Development setup

This project supports Python 3.11 and newer.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .[dev]
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

On Unix-like shells, use `.venv/bin/python` instead of
`.venv/Scripts/python.exe`.

## Adding a routing policy

Routing policy lives in `src/llm_gateway/routing/router.py`.

1. Add the policy value to the `Policy` enum.
2. Implement candidate ordering in `Router.candidates`.
3. Keep fallback behavior in `Router.route` unchanged unless the new policy
   explicitly requires different retry semantics.
4. Add focused tests in `tests/test_router.py` for ordering, fallback, and
   circuit-breaker interaction.
5. Document any new `GATEWAY_` setting in `README.md`.

Policies should choose among `Candidate` objects. They should not call providers
or mutate breaker state during selection.

## Adding an upstream provider

Provider adapters implement the protocol in
`src/llm_gateway/providers/base.py`.

1. Create a provider module under `src/llm_gateway/providers/`.
2. Define supported `ModelSpec` entries with pricing, context window, and
   latency hint.
3. Implement `supports`, `spec`, `complete`, and `stream`.
4. Translate vendor errors into `GatewayError` subclasses so the router can
   distinguish retryable upstream failures from bad requests.
5. Wire the provider name in `build_providers` in `src/llm_gateway/app.py`.
6. Add any provider SDK as an optional extra in `pyproject.toml` unless it is
   required for the core gateway.
7. Add tests that do not require real network calls or live API keys.

Do not log, expose, or store upstream API keys. Prefer SDK environment-variable
resolution when it keeps credentials out of application state.

## House style

- Use `from __future__ import annotations` in Python modules.
- Keep docstrings WHY-focused: explain intent, invariants, tradeoffs, or
  operational consequences.
- Prefer existing local patterns over new abstractions.
- Keep behavior changes small and covered by tests.
- Line length is controlled by `pyproject.toml`; currently Ruff is configured
  for 90 columns.
- Run `pytest -q` and `ruff check .` before submitting changes.
