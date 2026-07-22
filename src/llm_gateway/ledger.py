"""Per-tenant cost accounting and budget enforcement."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from llm_gateway.errors import RateLimited


@dataclass
class TenantSpend:
    requests: int = 0
    cached_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    # What the uncached path would have cost. cost_usd + saved_usd is the
    # counterfactual bill, which is the number worth putting on a dashboard.
    saved_usd: float = 0.0


class CostLedger:
    def __init__(self, budgets_usd: dict[str, float] | None = None) -> None:
        self._spend: dict[str, TenantSpend] = defaultdict(TenantSpend)
        self._budgets = budgets_usd or {}

    def check_budget(self, tenant: str) -> None:
        budget = self._budgets.get(tenant)
        if budget is not None and self._spend[tenant].cost_usd >= budget:
            # The comparison is against billed spend only; cached hits accrue
            # saved_usd so dashboards show avoided cost without burning budget.
            raise RateLimited(
                f"tenant '{tenant}' exhausted its ${budget:.2f} budget"
            )

    def record(
        self,
        tenant: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        cached: bool,
    ) -> None:
        spend = self._spend[tenant]
        spend.requests += 1
        if cached:
            # A cache hit bills nothing but still represents avoided spend.
            spend.cached_requests += 1
            spend.saved_usd += cost_usd
            return
        spend.prompt_tokens += prompt_tokens
        spend.completion_tokens += completion_tokens
        spend.cost_usd += cost_usd

    def spend(self, tenant: str) -> TenantSpend:
        return self._spend[tenant]

    def all_spend(self) -> dict[str, TenantSpend]:
        return dict(self._spend)
