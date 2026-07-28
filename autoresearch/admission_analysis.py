"""Admission-time reuse prediction: can we read the request and predict
whether its prefix will be used again in the future?

Framing (differs from feature_analysis_meta.py, which scores blocks at
EVICTION time among survivors): here the decision point is the moment a
request finishes and we must decide whether/which of its blocks to save.

Per-request label: is the DEEPEST block of request r touched by any later
request (within a horizon)?  Deepest-block reuse == the whole prefix was
replayed == the conversation continued, so this is exactly the
"conversation continuation" prediction task (same task as LPC, NeurIPS'25,
but from request-log metadata instead of a 118M text-embedding model).

Features available at decision time (end of request r):
    tenant  system_prompt_hash popularity (recent-1000-request window)
    nmsg    messages in conversation so far
    lulen   last user message length
    lq      last user message contains '?'
    out     output length of the request
    nblk    number of blocks in the request's path
    think   ms since the conversation's previous turn (0 = new conversation)

Models:
  - per-feature bucketed rate tables (shrinkage m=3), first-half train ->
    second-half test, AUC  (same protocol as feature_analysis_meta.py)
  - combined logistic regression on one-hot buckets (pure python SGD)

Also reports the "save point" analysis: reuse rate of newly-created blocks
by relative position in the prompt, split by whether the conversation
continued -- i.e. if we predict "won't continue", what do we lose by not
saving the tail vs the shared head?

Run: python3 autoresearch/admission_analysis.py traces/day5_15k.jsonl.gz [max_requests]
"""
from __future__ import annotations

import bisect
import collections
import math
import sys

from kvcache_sim.plan import build_execution_plan
from kvcache_sim.trace import parse_trace_file

HORIZONS = (64, 1 << 60)  # within-64-requests (small-cache relevant), and "ever"


def bucket(value: float, edges: tuple[float, ...]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


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


FEATURE_SIZES = {
    "tenant": 5,
    "nmsg": 6,
    "lulen": 4,
    "lq": 2,
    "out": 5,
    "nblk": 6,
    "think": 7,
}
FEATURE_ORDER = list(FEATURE_SIZES)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--real-only"]
    real_only = "--real-only" in sys.argv[1:]
    trace_path = args[0]
    max_requests = int(args[1]) if len(args) > 1 else 0
    trace = parse_trace_file(trace_path, max_records=max_requests)
    plan = build_execution_plan(trace)
    metas = plan.request_meta
    if metas is None:
        raise SystemExit("trace has no meta field")
    n_nodes = len(plan.parent)
    R = plan.request_count

    last_seen_ts = [0] * n_nodes
    seen = [False] * n_nodes
    touches: list[list[int]] = [[] for _ in range(n_nodes)]

    # request-level snapshots: feature bucket tuple, deepest node, new-node info
    feats: list[tuple[int, ...]] = []
    deepest: list[int] = []
    # per-request: list of (relative_position_bucket, node) for NEW nodes
    new_blocks: list[list[tuple[int, int]]] = []

    tenant_window: collections.deque[str] = collections.deque()
    tenant_count: collections.Counter[str] = collections.Counter()

    for r in range(R):
        meta = metas[r] or {}
        ts = int(meta.get("ts") or 0)
        sph = str(meta.get("sph") or "")
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
        nblk = end - start
        think_ms = -1
        news: list[tuple[int, int]] = []
        node = 0
        for i in range(start, end):
            node = plan.node_for_event[i]
            if seen[node]:
                think_ms = ts - last_seen_ts[node]  # deepest reused node wins
            else:
                seen[node] = True
                pos_b = bucket((i - start) / max(1, nblk - 1) if nblk > 1 else 0.0,
                               (0.25, 0.5, 0.75, 0.95))
                news.append((pos_b, node))
            touches[node].append(r)
            last_seen_ts[node] = ts

        think_b = 0 if think_ms < 0 else 1 + bucket(
            think_ms, (5_000, 30_000, 120_000, 600_000, 3_600_000))
        fb = (
            tenant_b,
            bucket(int(meta.get("nmsg") or 0), (2, 4, 8, 16, 32)),
            bucket(int(meta.get("lulen") or 0), (50, 200, 1000)),
            1 if meta.get("lq") else 0,
            bucket(int(meta.get("out") or 0), (16, 64, 256, 1024)),
            bucket(nblk, (4, 8, 16, 32, 64)),
            think_b,
        )
        # reorder to FEATURE_ORDER: tenant,nmsg,lulen,lq,out,nblk,think
        feats.append(fb)
        deepest.append(node)
        new_blocks.append(news)

    def touched_within(node: int, r: int, horizon: int) -> int:
        lst = touches[node]
        k = bisect.bisect_right(lst, r)
        return 1 if k < len(lst) and lst[k] - r <= horizon else 0

    # --real-only: restrict labels/eval to genuine conversations (non-empty
    # system-prompt hash, >1 block).  Single-block empty-sph requests have a
    # globally shared block whose "reuse" is not conversation continuation.
    eligible = list(range(R))
    if real_only:
        eligible = [
            r for r in range(R)
            if str((metas[r] or {}).get("sph") or "")
            and plan.request_starts[r + 1] - plan.request_starts[r] > 1
        ]
    print(f"trace={trace_path} requests={R} nodes={n_nodes} "
          f"eval_requests={len(eligible)}{' (real-only)' if real_only else ''}")

    # ---- save-point analysis: new-block reuse rate by relative position ----
    labels_ever = [touched_within(deepest[r], r, 1 << 60) for r in range(R)]
    pos_stats: dict[tuple[int, int], list[int]] = {}
    for r in eligible:
        for pos_b, node in new_blocks[r]:
            s = pos_stats.setdefault((labels_ever[r], pos_b), [0, 0])
            s[0] += touched_within(node, r, 1 << 60)
            s[1] += 1
    print("\nnew-block reuse rate by relative position in prompt "
          "(rows: did conversation continue?)")
    print("           pos<=.25   .25-.5    .5-.75   .75-.95     tail")
    for lab, name in ((1, "continue"), (0, "no-cont ")):
        row = []
        for pb in range(5):
            s = pos_stats.get((lab, pb))
            row.append(f"{s[0]/s[1]:8.3f}" if s and s[1] else "     n/a")
        print(f"  {name}  " + "  ".join(row))

    # ---- prediction ----
    half = R // 2
    for horizon in HORIZONS:
        hname = "ever" if horizon >= 1 << 59 else f"<={horizon}req"
        labels = (labels_ever if horizon >= 1 << 59
                  else [touched_within(deepest[r], r, horizon) for r in range(R)])
        tr = [r for r in eligible if r < half]
        te = [r for r in eligible if r >= half]
        base = sum(labels[r] for r in te) / len(te)
        print(f"\n=== horizon {hname}:  test base rate {base:.3f}  "
              f"(train n={len(tr)}, test n={len(te)}) ===")

        # single-feature rate tables
        print(f"{'feature':>10s}  {'AUC':>6s}")
        for fi, name in enumerate(FEATURE_ORDER):
            rate: dict[int, list[int]] = {}
            for r in tr:
                s = rate.setdefault(feats[r][fi], [0, 0])
                s[0] += labels[r]
                s[1] += 1
            g = sum(v[0] for v in rate.values()) / max(1, sum(v[1] for v in rate.values()))
            pairs = []
            for r in te:
                s = rate.get(feats[r][fi])
                score = (s[0] + 3 * g) / (s[1] + 3) if s else g
                pairs.append((score, labels[r]))
            print(f"{name:>10s}  {auc(pairs):6.3f}")

        # combined logistic regression on one-hot buckets
        offs = []
        off = 0
        for name in FEATURE_ORDER:
            offs.append(off)
            off += FEATURE_SIZES[name]
        dim = off

        def onehot(r: int) -> list[int]:
            return [offs[fi] + feats[r][fi] for fi in range(len(FEATURE_ORDER))]

        w = [0.0] * dim
        b0 = 0.0
        lr = 0.1
        for _epoch in range(30):
            for r in tr:
                z = b0 + sum(w[j] for j in onehot(r))
                p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
                g_ = labels[r] - p
                b0 += lr * g_
                for j in onehot(r):
                    w[j] += lr * g_ - lr * 1e-4 * w[j]
            lr *= 0.93

        pairs = []
        for r in te:
            z = b0 + sum(w[j] for j in onehot(r))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            pairs.append((p, labels[r]))
        a = auc(pairs)

        # operating points: accuracy at best-train-threshold, P/R at keep rates
        thr_pairs = []
        for r in tr:
            z = b0 + sum(w[j] for j in onehot(r))
            thr_pairs.append((1.0 / (1.0 + math.exp(-max(-30, min(30, z)))), labels[r]))
        best_thr, best_acc = 0.5, 0.0
        for t10 in range(1, 100):
            t = t10 / 100.0
            acc = sum(1 for p, lab in thr_pairs if (p >= t) == bool(lab)) / len(thr_pairs)
            if acc > best_acc:
                best_acc, best_thr = acc, t
        acc_te = sum(1 for p, lab in pairs if (p >= best_thr) == bool(lab)) / len(pairs)
        majority = max(base, 1 - base)
        print(f"\n  combined logistic: AUC {a:.3f}  "
              f"accuracy {acc_te:.3f} @thr={best_thr:.2f} "
              f"(majority-class baseline {majority:.3f})")
        ranked = sorted(pairs, reverse=True)
        for keep in (0.3, 0.5, 0.7):
            k = int(len(ranked) * keep)
            kept = ranked[:k]
            tp = sum(lab for _p, lab in kept)
            pos = sum(lab for _p, lab in ranked)
            print(f"  save top {int(keep*100)}% -> precision {tp/max(1,k):.3f}  "
                  f"recall {tp/max(1,pos):.3f}")


if __name__ == "__main__":
    main()
