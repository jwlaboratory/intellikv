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


def conversation_lines(requests: int = 600) -> list[str]:
    """Synthetic chat workload with per-request meta.

    Sticky conversations (short think times, many turns) keep extending a
    shared prefix; one-shot conversations never return. A content-aware
    policy should learn to protect sticky conversations' blocks.
    """
    rng = random.Random(11)
    lines = []
    ts = 0
    sticky: list[dict] = []
    one_shot = 0
    for _ in range(requests):
        ts += rng.randrange(200, 2000)
        extend = sticky and rng.random() < 0.6
        if extend:
            conv = rng.choice(sticky)
            conv["turns"] += 1
            conv["blocks"].append(f"c{conv['id']}-{conv['turns']}")
            ids = list(conv["blocks"])
            meta = {"ts": ts, "sph": "app-sticky", "nmsg": 1 + 2 * conv["turns"], "out": 30}
        elif rng.random() < 0.5:
            conv = {"id": len(sticky), "turns": 1, "blocks": [f"sys-s-{i}" for i in range(2)]}
            conv["blocks"].append(f"c{conv['id']}-1")
            sticky.append(conv)
            if len(sticky) > 12:
                sticky.pop(0)
            ids = list(conv["blocks"])
            meta = {"ts": ts, "sph": "app-sticky", "nmsg": 3, "out": 30}
        else:
            one_shot += 1
            ids = [f"sys-o-{i}" for i in range(2)] + [f"once-{one_shot}-{i}" for i in range(6)]
            meta = {"ts": ts, "sph": "app-oneshot", "nmsg": 2, "out": 900}
        lines.append(json.dumps({"block_size": 16, "hash_ids": ids, "input_length": 16 * len(ids), "meta": meta}))
    return lines


class MetaPolicyTests(unittest.TestCase):
    def test_meta_plumbing(self) -> None:
        plan = build_execution_plan(parse_trace_lines(conversation_lines(50)))
        self.assertIsNotNone(plan.request_meta)
        self.assertEqual(len(plan.request_meta), 50)
        self.assertIn("sph", plan.request_meta[0])

    def test_custom_beats_lru_on_conversational_workload(self) -> None:
        plan = build_execution_plan(parse_trace_lines(conversation_lines()))
        custom = simulate_policy(plan, "custom", 24)
        lru = simulate_policy(plan, "lru", 24)
        optimal = simulate_policy(plan, "optimal", 24)
        self.assertGreater(custom.hitRate, lru.hitRate)
        self.assertGreaterEqual(optimal.hitRate, custom.hitRate - 1e-9)


if __name__ == "__main__":
    unittest.main()
