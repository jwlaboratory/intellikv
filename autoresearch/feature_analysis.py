"""Offline predictiveness analysis: which causal block features predict
time-to-next-touch beyond idle time itself?

For every touch interval (node touched at request r, next touch at r+gap) we
snapshot causal features at r. For each idle checkpoint e we consider all
intervals with gap > e (block survived to idle e) and score the binary label
"returns within the next DELTA requests" (gap <= e + DELTA). Feature-bucket
rates are fit on the first half of intervals and AUC is measured on the
second half, so this mirrors online learning + later use.

Run: python3.11 autoresearch/feature_analysis.py <trace> [max_requests]
"""
from __future__ import annotations

import sys

from kvcache_sim.plan import build_execution_plan
from kvcache_sim.trace import parse_trace_file

DELTA = 32
IDLE_CHECKPOINTS = (0, 4, 16, 64, 128, 256)
NEVER = 1 << 60


def depth_bucket(depth: int) -> int:
    for i, edge in enumerate((2, 8, 32, 128)):
        if depth <= edge:
            return i
    return 4


def freq_bucket(uses: int) -> int:
    for i, edge in enumerate((1, 2, 4, 8, 16)):
        if uses <= edge:
            return i
    return 5


def gap_bucket(gap: int) -> int:
    if gap <= 0:
        return 0
    for i, edge in enumerate((5, 20, 80)):
        if gap <= edge:
            return i + 1
    return 4


def auc(pairs: list[tuple[float, int]]) -> float:
    """AUC of score vs binary label via rank statistic."""
    pos = sum(label for _score, label in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    pairs = sorted(pairs, key=lambda p: p[0])
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum += avg_rank
        rank += j - i
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def main() -> None:
    trace_path = sys.argv[1] if len(sys.argv) > 1 else "art_chat_20k.jsonl.gz"
    max_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    trace = parse_trace_file(trace_path, max_records=max_requests)
    plan = build_execution_plan(trace)
    n = len(plan.parent)

    depth = [0] * n
    last_seen = [-1] * n
    uses = [0] * n
    prev_gap = [0] * n

    # intervals: (gap, depth_b, freq_b, own_gap_b, parent_gap_b, start_request)
    intervals: list[tuple[int, int, int, int, int, int]] = []
    open_interval: dict[int, tuple[int, int, int, int]] = {}  # node -> feature snapshot at last touch

    for r in range(plan.request_count):
        start = plan.request_starts[r]
        end = plan.request_starts[r + 1]
        for i in range(start, end):
            node = plan.node_for_event[i]
            if depth[node] == 0 and node != 0:
                depth[node] = depth[plan.parent[node]] + 1
            seen = last_seen[node]
            if seen >= 0:
                gap = r - seen
                if gap > 0:
                    snap = open_interval.get(node)
                    if snap is not None:
                        intervals.append((gap,) + snap + (seen,))
                    prev_gap[node] = gap
            uses[node] += 1
            last_seen[node] = r
            parent = plan.parent[node]
            open_interval[node] = (
                depth_bucket(depth[node]),
                freq_bucket(uses[node]),
                gap_bucket(prev_gap[node]),
                gap_bucket(prev_gap[parent]) if parent > 0 else 0,
            )
    # close never-returned intervals with gap = NEVER
    for node, snap in open_interval.items():
        intervals.append((NEVER,) + snap + (last_seen[node],))

    print(f"trace={trace_path} requests={plan.request_count} intervals={len(intervals)}")

    # temporal split by interval start
    intervals.sort(key=lambda t: t[5])
    half = len(intervals) // 2
    train, test = intervals[:half], intervals[half:]

    feature_defs = {
        "depth": lambda t: t[1],
        "freq": lambda t: t[2],
        "own_gap": lambda t: t[3],
        "parent_gap": lambda t: t[4],
        "own+parent_gap": lambda t: t[3] * 5 + t[4],
        "freq+own_gap": lambda t: t[2] * 5 + t[3],
        "depth+freq": lambda t: t[1] * 6 + t[2],
        "all": lambda t: ((t[1] * 6 + t[2]) * 5 + t[3]) * 5 + t[4],
    }

    print(f"\nAUC of P(return within {DELTA}) among survivors at idle e (train->test):")
    header = "idle_e  survivors  base_rate  " + "  ".join(f"{name:>15s}" for name in feature_defs)
    print(header)
    for e in IDLE_CHECKPOINTS:
        surv_train = [t for t in train if t[0] > e]
        surv_test = [t for t in test if t[0] > e]
        if len(surv_test) < 100:
            continue
        labels_test = [1 if t[0] <= e + DELTA else 0 for t in surv_test]
        base = sum(labels_test) / len(labels_test)
        row = f"{e:6d}  {len(surv_test):9d}  {base:9.3f}  "
        cells = []
        for name, fn in feature_defs.items():
            rate: dict[int, list[int]] = {}
            for t in surv_train:
                b = fn(t)
                s = rate.setdefault(b, [0, 0])
                s[0] += 1 if t[0] <= e + DELTA else 0
                s[1] += 1
            global_rate = sum(v[0] for v in rate.values()) / max(1, sum(v[1] for v in rate.values()))
            pairs = []
            for t, label in zip(surv_test, labels_test):
                s = rate.get(fn(t))
                score = (s[0] + 3 * global_rate) / (s[1] + 3) if s else global_rate
                pairs.append((score, label))
            cells.append(f"{auc(pairs):15.3f}")
        print(row + "  ".join(cells))


if __name__ == "__main__":
    main()
