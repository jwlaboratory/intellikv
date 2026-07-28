"""Which eviction ORDERING best matches actual time-to-next-touch?

Simulates eviction moments: every SAMPLE_EVERY requests, take the set of live
touch intervals (block touched before r, next touch after r) with idle below a
window W (a proxy for what a cache of that size still holds), subsample
candidates, and measure pairwise concordance: how often does each scorer rank
a block with larger actual remaining wait as MORE evictable?

Scorers (higher = evict first):
    lru        logical idle (requests since last touch)  == LRU baseline
    real       real-time idle (ms since last touch)
    overdue    real idle / node's own mean past real gap (cadence-normalized)
    hazard     learned E[min(remaining,H) | think+nmsg+freq bucket, idle bucket]
               (fit online, causally, like the v1 policy but with meta features)

Run: python3.11 autoresearch/ranking_analysis.py <trace> [max_requests]
"""
from __future__ import annotations

import random
import sys

from kvcache_sim.plan import build_execution_plan
from kvcache_sim.trace import parse_trace_file

SAMPLE_EVERY = 500
CAND_PER_SNAPSHOT = 250
WINDOWS = (64, 256, 1024)
NEVER = 1 << 60
IDLE_EDGES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
HORIZON = 256


def bucket(value: float, edges: tuple[float, ...]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def main() -> None:
    trace_path = sys.argv[1]
    max_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    trace = parse_trace_file(trace_path, max_records=max_requests)
    plan = build_execution_plan(trace)
    metas = plan.request_meta or [None] * plan.request_count
    n = len(plan.parent)
    rng = random.Random(999)

    depth = [0] * n
    last_seen = [-1] * n
    last_seen_ts = [0] * n
    uses = [0] * n
    gap_sum_ms = [0.0] * n
    gap_cnt = [0] * n
    feat_state = [0] * n

    n_idle = len(IDLE_EDGES)
    n_feat = 7 * 6 * 6  # think_b x nmsg_b x freq_b
    haz_cnt = [0] * (n_feat * n_idle)
    haz_sum = [0.0] * (n_feat * n_idle)
    glob_cnt = [0] * n_idle
    glob_sum = [0.0] * n_idle

    req_ts = [0] * plan.request_count

    # Pass 1: record per-node touch times (logical + real) and fit hazard online.
    # We also snapshot, for every touch interval, what we need to score it later:
    # (node, start_r, start_ts, end_r, feat_at_start, mean_gap_at_start, uses_at_start)
    intervals: list[tuple[int, int, int, int, int, float]] = []
    open_idx: dict[int, int] = {}

    def record(fb: int, gap: int) -> None:
        base = fb * n_idle
        for j in range(n_idle):
            edge = IDLE_EDGES[j]
            if edge >= gap or edge > HORIZON:
                break
            rem = gap - edge
            if rem > HORIZON:
                rem = HORIZON
            haz_cnt[base + j] += 1
            haz_sum[base + j] += rem
            glob_cnt[j] += 1
            glob_sum[j] += rem

    for r in range(plan.request_count):
        meta = metas[r] or {}
        ts = int(meta.get("ts") or 0)
        req_ts[r] = ts
        nmsg_b = bucket(int(meta.get("nmsg") or 0), (2, 4, 8, 16, 32))
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
                    idx = open_idx.get(node)
                    if idx is not None:
                        rec = intervals[idx]
                        intervals[idx] = rec[:3] + (r,) + rec[4:]
                    record(feat_state[node], gap)
                    gap_ms = ts - last_seen_ts[node]
                    if gap_ms > 0:
                        gap_sum_ms[node] += gap_ms
                        gap_cnt[node] += 1
            think_ms = (ts - last_seen_ts[node]) if seen >= 0 else -1
            think_b = 0 if think_ms < 0 else 1 + bucket(think_ms, (5_000, 30_000, 120_000, 600_000, 3_600_000))
            uses[node] += 1
            fb = (think_b * 6 + nmsg_b) * 6 + bucket(uses[node], (1, 2, 4, 8, 16))
            last_seen[node] = r
            last_seen_ts[node] = ts
            feat_state[node] = fb
            mean_gap = gap_sum_ms[node] / gap_cnt[node] if gap_cnt[node] else -1.0
            open_idx[node] = len(intervals)
            intervals.append((node, r, ts, NEVER, fb, mean_gap))

    print(f"trace={trace_path} requests={plan.request_count} intervals={len(intervals)}")

    def hazard_score(fb: int, idle: int) -> float:
        ib = bucket(idle, IDLE_EDGES[1:])  # index of highest edge <= idle
        # align with edges: bucket() returns count of edges < ... adjust:
        ib = 0
        for j in range(n_idle - 1, -1, -1):
            if IDLE_EDGES[j] <= idle:
                ib = j
                break
        slot = fb * n_idle + ib
        if haz_cnt[slot] >= 5:
            return haz_sum[slot] / haz_cnt[slot]
        if glob_cnt[ib] >= 5:
            return glob_sum[ib] / glob_cnt[ib]
        return float(idle)

    # Pass 2: sweep eviction snapshots.
    intervals.sort(key=lambda t: t[1])
    results = {w: {k: [0, 0] for k in ("lru", "real", "overdue", "hazard")} for w in WINDOWS}
    live: list[tuple[int, int, int, int, int, float]] = []
    idx = 0
    warmup = plan.request_count // 2
    for r in range(warmup, plan.request_count, SAMPLE_EVERY):
        while idx < len(intervals) and intervals[idx][1] < r:
            live.append(intervals[idx])
            idx += 1
        live = [t for t in live if t[3] > r]
        ts_now = req_ts[r]
        for w in WINDOWS:
            cands = [t for t in live if r - t[1] <= w]
            if len(cands) > CAND_PER_SNAPSHOT:
                cands = rng.sample(cands, CAND_PER_SNAPSHOT)
            if len(cands) < 10:
                continue
            scored = []
            for node, sr, sts, er, fb, mean_gap in cands:
                idle_log = r - sr
                idle_real = max(0, ts_now - sts)
                overdue = idle_real / mean_gap if mean_gap > 0 else idle_real / 60_000.0
                remaining = er - r if er != NEVER else NEVER
                scored.append((idle_log, idle_real, overdue, hazard_score(fb, idle_log), remaining))
            for a in range(0, len(scored) - 1, 2):
                s1, s2 = scored[a], scored[a + 1]
                if s1[4] == s2[4]:
                    continue
                worse_is_1 = s1[4] > s2[4]  # 1 has longer remaining => should be evicted first
                for ki, key in enumerate(("lru", "real", "overdue", "hazard")):
                    v1, v2 = s1[ki], s2[ki]
                    if v1 == v2:
                        continue
                    correct = (v1 > v2) == worse_is_1
                    results[w][key][0] += 1 if correct else 0
                    results[w][key][1] += 1
    for w in WINDOWS:
        row = "  ".join(
            f"{k}={results[w][k][0] / results[w][k][1]:.3f}({results[w][k][1]})" if results[w][k][1] else f"{k}=n/a"
            for k in results[w]
        )
        print(f"window<={w:5d}: {row}")


if __name__ == "__main__":
    main()
