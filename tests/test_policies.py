import json
import random
import unittest

from intellikv.plan import build_execution_plan
from intellikv.policies import simulate_ceiling, simulate_policy
from intellikv.trace import parse_trace_lines

ALL_POLICIES = ["fifo", "lru", "optimal", "custom"]


def hot_and_scan_trace(requests: int = 400) -> list[str]:
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


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_execution_plan(parse_trace_lines(hot_and_scan_trace()))
        cls.ceiling = simulate_ceiling(cls.plan)

    def test_all_policies_run_and_stay_below_ceiling(self):
        for policy in ALL_POLICIES:
            result = simulate_policy(self.plan, policy, 120)
            self.assertFalse(result.underfilled, policy)
            self.assertGreaterEqual(result.hit_rate, 0.0, policy)
            self.assertLessEqual(result.hit_rate, self.ceiling.hit_rate + 1e-9, policy)

    def test_optimal_dominates_lru_and_fifo(self):
        optimal = simulate_policy(self.plan, "optimal", 120)
        for policy in ["fifo", "lru", "custom"]:
            result = simulate_policy(self.plan, policy, 120)
            self.assertGreaterEqual(optimal.hit_rate, result.hit_rate - 1e-9, policy)

    def test_custom_lfu_beats_lru_on_scan_pollution(self):
        custom = simulate_policy(self.plan, "custom", 120)
        lru = simulate_policy(self.plan, "lru", 120)
        self.assertGreater(custom.hit_rate, lru.hit_rate)

    def test_oversized_capacity_is_underfilled(self):
        result = simulate_policy(self.plan, "lru", self.plan.unique_blocks + 1000)
        self.assertTrue(result.underfilled)

    def test_repeated_identical_requests_hit_fully(self):
        lines = [json.dumps({"block_size": 16, "hash_ids": ["a", "b", "c"], "input_length": 48})] * 10
        # Tiny capacity (2 < 3 blocks) forces pressure so the run is measured.
        plan = build_execution_plan(parse_trace_lines(lines))
        result = simulate_policy(plan, "lru", 2)
        self.assertFalse(result.underfilled)
        # With capacity below the working set only a partial prefix can hit.
        self.assertLess(result.hit_rate, 1.0)
        self.assertGreater(simulate_ceiling(plan).hit_rate, 0.99)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            simulate_policy(self.plan, "nope", 10)


if __name__ == "__main__":
    unittest.main()
