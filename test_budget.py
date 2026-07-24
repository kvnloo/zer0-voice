import unittest

from budget import BudgetRouter, QuotaWindow, Route


class BudgetRouterTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_000.0
        self.routes = [
            Route("local", quality=1, cost=0, metered=False),
            Route("luna", quality=2, cost=1),
            Route("terra", quality=3, cost=3),
            Route("sol", quality=4, cost=8),
            Route("pro", quality=5, cost=20),
        ]

    def router(self):
        return BudgetRouter(
            self.routes, reserve=0.20, telemetry_ttl=60, clock=lambda: self.now
        )

    def test_uses_cheapest_route_that_meets_quality(self):
        router = self.router()
        router.observe([QuotaWindow("five-hour", 100, 0, self.now + 300)])
        self.assertEqual(router.route(3).name, "terra")

    def test_every_window_must_retain_reserve(self):
        router = self.router()
        router.observe(
            [
                QuotaWindow("five-hour", 100, 50, self.now + 300),
                QuotaWindow("weekly", 100, 77, self.now + 30_000),
            ]
        )
        self.assertEqual(router.route(2).name, "luna")
        with self.assertRaisesRegex(RuntimeError, "quota reserves"):
            router.route(3)

    def test_stale_or_missing_telemetry_fails_closed_for_metered_routes(self):
        router = self.router()
        self.assertEqual(router.route(1).name, "local")
        with self.assertRaisesRegex(RuntimeError, "quota reserves"):
            router.route(2)
        router.observe([QuotaWindow("weekly", 100, 0, self.now + 30_000)])
        self.now += 61
        with self.assertRaisesRegex(RuntimeError, "quota reserves"):
            router.route(2)

    def test_concurrent_reservations_prevent_oversubscription(self):
        router = self.router()
        router.observe([QuotaWindow("five-hour", 10, 0, self.now + 300)])
        self.assertEqual(router.route(4).name, "sol")
        with self.assertRaisesRegex(RuntimeError, "quota reserves"):
            router.route(2)

    def test_settlement_releases_reservation_and_accounts_actual_cost(self):
        router = self.router()
        router.observe([QuotaWindow("weekly", 100, 0, self.now + 30_000)])
        route = router.route(3)
        router.settle(route.cost, 2)
        self.assertEqual(router.reserved, 0)
        self.assertEqual(router.windows[0].used, 2)


if __name__ == "__main__":
    unittest.main()
