"""Per-tenant cost accounting and budget enforcement."""

from __future__ import annotations

import threading
import uuid
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


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    tenant: str
    amount_usd: float


class CostLedger:
    def __init__(self, budgets_usd: dict[str, float] | None = None) -> None:
        self._spend: dict[str, TenantSpend] = defaultdict(TenantSpend)
        self._budgets = budgets_usd or {}
        self._reserved: dict[str, BudgetReservation] = {}
        self._reserved_by_tenant: dict[str, float] = defaultdict(float)
        self._lock = threading.RLock()

    def check_budget(self, tenant: str) -> None:
        with self._lock:
            budget = self._budgets.get(tenant)
            committed = self._spend[tenant].cost_usd
            reserved = self._reserved_by_tenant[tenant]
            if budget is not None and committed + reserved >= budget:
                raise RateLimited(f"tenant '{tenant}' exhausted its ${budget:.6f} budget")

    def reserve(self, tenant: str, amount_usd: float) -> BudgetReservation:
        """Atomically reserve worst-case request cost before provider I/O."""
        if amount_usd < 0:
            raise ValueError("reservation amount must be non-negative")
        with self._lock:
            budget = self._budgets.get(tenant)
            available = (
                None
                if budget is None
                else max(
                    0.0,
                    budget
                    - self._spend[tenant].cost_usd
                    - self._reserved_by_tenant[tenant],
                )
            )
            if available is not None and amount_usd > available + 1e-12:
                raise RateLimited(
                    f"tenant '{tenant}' request could spend ${amount_usd:.6f}; "
                    f"only ${available:.6f} remains"
                )
            reservation = BudgetReservation(
                reservation_id=f"budget_{uuid.uuid4().hex}",
                tenant=tenant,
                amount_usd=amount_usd,
            )
            self._reserved[reservation.reservation_id] = reservation
            self._reserved_by_tenant[tenant] += amount_usd
            return reservation

    def release(self, reservation: BudgetReservation) -> None:
        """Release a reservation after a request fails before billable success."""
        with self._lock:
            current = self._reserved.pop(reservation.reservation_id, None)
            if current is None:
                return
            self._reserved_by_tenant[current.tenant] = max(
                0.0,
                self._reserved_by_tenant[current.tenant] - current.amount_usd,
            )

    def settle(
        self,
        reservation: BudgetReservation,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        """Convert one reservation to immutable billed spend exactly once."""
        with self._lock:
            current = self._reserved.get(reservation.reservation_id)
            if current is None:
                raise ValueError("budget reservation is unknown or already settled")
            if cost_usd > current.amount_usd + 1e-12:
                raise ValueError(
                    "provider-reported cost exceeded the preflight reservation; "
                    "the model price or token bound is inconsistent"
                )
            self.release(current)
            self.record(
                current.tenant,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                cached=False,
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
        with self._lock:
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
        with self._lock:
            return self._spend[tenant]

    def all_spend(self) -> dict[str, TenantSpend]:
        with self._lock:
            return dict(self._spend)

    def budget_status(self, tenant: str) -> dict[str, float | None]:
        with self._lock:
            budget = self._budgets.get(tenant)
            spent = self._spend[tenant].cost_usd
            reserved = self._reserved_by_tenant[tenant]
            return {
                "budget_usd": budget,
                "reserved_usd": reserved,
                "remaining_usd": (
                    None if budget is None else max(0.0, budget - spent - reserved)
                ),
            }
