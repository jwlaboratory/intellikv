"""Tests for the IntelliKV custom-policy patch (not part of upstream)."""
from __future__ import annotations

import json
import random
import unittest

from kvcache_sim.plan import build_execution_plan
from kvcache_sim.policies import simulate_ceiling, simulate_policy
from kvcache_sim.trace import parse_trace_lines


def hot_and_scan_lines(requests: int = 400) -> list[str]:
    rng = random.Random(7)
    system = [f"sys-{i}" for i in range(4)]
    hot = {c: [f"hot-{c}-{i}" for i in range(8)] for c in range(10)}
    lines = []
    scans = 0
    for _ in range(requests):
        if rng.random() < 0.5:
            ids = system + hot[rng.randrange(10)]
        else:
            ids = system + [f"scan-{scans}-{i}" for i in range(12)]
            scans += 1
        lines.append(json.dumps({"block_size": 16, "hash_ids": ids, "input_length": 16 * len(ids)}))
    return lines


class CustomPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_execution_plan(parse_trace_lines(hot_and_scan_lines()))
        cls.ceiling = simulate_ceiling(cls.plan)

    def test_custom_runs_and_stays_below_ceiling(self) -> None:
        result = simulate_policy(self.plan, "custom", 120)
        self.assertEqual(result.policy, "custom")
        self.assertEqual(result.measurementMode, "fixed_window")
        self.assertGreater(result.hitRate, 0.0)
        self.assertLessEqual(result.hitRate, self.ceiling.hitRate + 1e-9)

    def test_optimal_dominates_custom(self) -> None:
        custom = simulate_policy(self.plan, "custom", 120)
        optimal = simulate_policy(self.plan, "optimal", 120)
        self.assertGreaterEqual(optimal.hitRate, custom.hitRate - 1e-9)

    def test_shipped_lfu_beats_lru_on_scan_pollution(self) -> None:
        custom = simulate_policy(self.plan, "custom", 120)
        lru = simulate_policy(self.plan, "lru", 120)
        self.assertGreater(custom.hitRate, lru.hitRate)

    def test_oversized_capacity_reports_underfilled(self) -> None:
        result = simulate_policy(self.plan, "custom", self.plan.unique_blocks + 1000)
        self.assertEqual(result.measurementMode, "underfilled_at_window")

    def test_custom_rejected_on_cpp_backend(self) -> None:
        from kvcache_sim.simulator import CPP_POLICIES, KNOWN_POLICIES

        self.assertIn("custom", KNOWN_POLICIES)
        self.assertNotIn("custom", CPP_POLICIES)


if __name__ == "__main__":
    unittest.main()
