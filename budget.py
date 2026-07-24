"""Fail-closed admission control for metered intelligence lanes."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class QuotaWindow:
    name: str
    allowance: float
    used: float
    resets_at: float

    def available(self, reserve: float, now: float) -> float:
        if self.resets_at <= now:
            return self.allowance * (1 - reserve)
        return max(0.0, self.allowance * (1 - reserve) - self.used)


@dataclass(frozen=True)
class Route:
    name: str
    quality: int
    cost: float
    metered: bool = True


class BudgetRouter:
    """Select the cheapest sufficient route that is safe in every time window.

    Cost is deliberately provider-neutral. A telemetry adapter may supply tokens,
    credits, or normalized subscription units, but it must use one unit system
    consistently for every window and route.
    """

    def __init__(
        self,
        routes: list[Route],
        *,
        reserve: float = 0.20,
        telemetry_ttl: float = 60.0,
        clock=time.time,
    ) -> None:
        if not 0 <= reserve < 1:
            raise ValueError("reserve must be between zero and one")
        self.routes = tuple(sorted(routes, key=lambda route: (route.cost, route.name)))
        self.reserve = reserve
        self.telemetry_ttl = telemetry_ttl
        self.clock = clock
        self.windows: tuple[QuotaWindow, ...] = ()
        self.observed_at: float | None = None
        self.reserved = 0.0

    def observe(self, windows: list[QuotaWindow]) -> None:
        if not windows:
            raise ValueError("at least one quota window is required")
        self.windows = tuple(windows)
        self.observed_at = self.clock()
        self.reserved = 0.0

    def _safe(self, cost: float) -> bool:
        now = self.clock()
        if self.observed_at is None or now - self.observed_at > self.telemetry_ttl:
            return False
        return all(
            window.available(self.reserve, now) >= self.reserved + cost
            for window in self.windows
        )

    def route(self, minimum_quality: int) -> Route:
        candidates = [
            route for route in self.routes if route.quality >= minimum_quality
        ]
        for route in candidates:
            if not route.metered or self._safe(route.cost):
                if route.metered:
                    self.reserved += route.cost
                return route
        raise RuntimeError("no route satisfies quality and quota reserves")

    def settle(self, reserved_cost: float, actual_cost: float) -> None:
        if reserved_cost < 0 or actual_cost < 0:
            raise ValueError("costs cannot be negative")
        self.reserved = max(0.0, self.reserved - reserved_cost)
        if actual_cost:
            self.windows = tuple(
                QuotaWindow(
                    window.name,
                    window.allowance,
                    window.used + actual_cost,
                    window.resets_at,
                )
                for window in self.windows
            )

