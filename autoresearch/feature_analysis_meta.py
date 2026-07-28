"""Predictiveness analysis for request-metadata features.

Same interval framework as feature_analysis.py (touch intervals per trie node,
AUC of "returns within DELTA requests" among survivors at idle e, first-half
train -> second-half test), but features come from the request `meta`:

    nmsg   messages in the conversation so far (turn depth)
    out    output_length of the request (response size)
    think  real ms since this node's previous touch (think time)
    tenant system_prompt_hash arrival popularity (recent-window count)
    lulen/lq  last user message length / contains '?'

Results split by node depth (shallow <=8 blocks vs deep) since shared-prefix
blocks and conversation-tail blocks behave differently.

Run: python3.11 autoresearch/feature_analysis_meta.py traces/day1_100k.jsonl.gz [max_requests]
"""
from __future__ import annotations

import collections
import sys

from kvcache_sim.plan import build_execution_plan
from kvcache_sim.trace import parse_trace_file

DELTA = 32
IDLE_CHECKPOINTS = (0, 4, 16, 64, 128)
NEVER = 1 << 60


def bucket(value: float, edges: tuple[float, ...]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def freq_bucket(uses: int) -> int:
    return bucket(uses, (1, 2, 4, 8, 16))


def auc(pairs: list[tuple[float, int]]) -> float:
    pos = sum(label for _s, label in pairs)
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
        avg = (2 * rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum += avg
        rank += j - i
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def main() -> None:
    trace_path = sys.argv[1]
    max_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    trace = parse_trace_file(trace_path, max_records=max_requests)
    plan = build_execution_plan(trace)
    metas = plan.request_meta
    if metas is None:
        raise SystemExit("trace has no meta field")
    n = len(plan.parent)

    depth = [0] * n
    last_seen = [-1] * n
    last_seen_ts = [0] * n
    uses = [0] * n

    tenant_window: collections.deque[str] = collections.deque()
    tenant_count: collections.Counter[str] = collections.Counter()

    # interval: (gap, start_request, depth, feature dict-free tuple)
    intervals: list[tuple] = []
    open_snap: dict[int, tuple] = {}

    for r in range(plan.request_count):
        meta = metas[r] or {}
        ts = int(meta.get("ts") or 0)
        sph = str(meta.get("sph") or "")
        nmsg_b = bucket(int(meta.get("nmsg") or 0), (2, 4, 8, 16, 32))
        out_b = bucket(int(meta.get("out") or 0), (16, 64, 256, 1024))
        lulen_b = bucket(int(meta.get("lulen") or 0), (50, 200, 1000))
        lq = 1 if meta.get("lq") else 0
        tenant_b = bucket(tenant_count.get(sph, 0), (1, 5, 20, 100))
        tenant_window.append(sph)
        tenant_count[sph] += 1
        if len(tenant_window) > 1000:
            old = tenant_window.popleft()
            tenant_count[old] -= 1
            if not tenant_count[old]:
                del tenant_count[old]

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
                    snap = open_snap.get(node)
                    if snap is not None:
                        intervals.append((gap, seen, depth[node]) + snap)
            think_ms = (ts - last_seen_ts[node]) if seen >= 0 else -1
            think_b = 0 if think_ms < 0 else 1 + bucket(think_ms, (5_000, 30_000, 120_000, 600_000, 3_600_000))
            uses[node] += 1
            last_seen[node] = r
            last_seen_ts[node] = ts
            open_snap[node] = (nmsg_b, out_b, think_b, tenant_b, lulen_b, lq, freq_bucket(uses[node]))
    for node, snap in open_snap.items():
        intervals.append((NEVER, last_seen[node], depth[node]) + snap)

    print(f"trace={trace_path} requests={plan.request_count} intervals={len(intervals)}")

    F = {  # index into snapshot part of interval tuple (offset 3)
        "nmsg": lambda t: t[3],
        "out": lambda t: t[4],
        "think": lambda t: t[5],
        "tenant": lambda t: t[6],
        "lulen": lambda t: t[7],
        "lq": lambda t: t[8],
        "freq": lambda t: t[9],
        "nmsg+think": lambda t: t[3] * 8 + t[5],
        "nmsg+out": lambda t: t[3] * 6 + t[4],
        "out+think": lambda t: t[4] * 8 + t[5],
        "nm+out+think": lambda t: (t[3] * 6 + t[4]) * 8 + t[5],
        "nm+th+freq": lambda t: (t[3] * 8 + t[5]) * 6 + t[9],
    }

    intervals.sort(key=lambda t: t[1])
    half = len(intervals) // 2
    train, test = intervals[:half], intervals[half:]

    for depth_class, pred in (("shallow(d<=8)", lambda d: d <= 8), ("deep(d>8)", lambda d: d > 8)):
        print(f"\n=== {depth_class} ===")
        print("idle_e  survivors  base  " + "  ".join(f"{k:>13s}" for k in F))
        for e in IDLE_CHECKPOINTS:
            surv_tr = [t for t in train if t[0] > e and pred(t[2])]
            surv_te = [t for t in test if t[0] > e and pred(t[2])]
            if len(surv_te) < 200:
                continue
            labels = [1 if t[0] <= e + DELTA else 0 for t in surv_te]
            base = sum(labels) / len(labels)
            cells = []
            for name, fn in F.items():
                rate: dict[int, list[int]] = {}
                for t in surv_tr:
                    s = rate.setdefault(fn(t), [0, 0])
                    s[0] += 1 if t[0] <= e + DELTA else 0
                    s[1] += 1
                tot_pos = sum(v[0] for v in rate.values())
                tot = sum(v[1] for v in rate.values())
                g = tot_pos / tot if tot else 0.0
                pairs = []
                for t, label in zip(surv_te, labels):
                    s = rate.get(fn(t))
                    score = (s[0] + 3 * g) / (s[1] + 3) if s else g
                    pairs.append((score, label))
                cells.append(f"{auc(pairs):13.3f}")
            print(f"{e:6d}  {len(surv_te):9d}  {base:.3f}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
