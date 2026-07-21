"""CLI: replay a JSONL trace against eviction policies at given capacities.

    python3 -m intellikv --trace trace.jsonl --capacities 256,1024,4096
"""
from __future__ import annotations

import argparse

from .plan import build_execution_plan
from .policies import POLICIES, simulate_ceiling, simulate_policy
from .trace import parse_trace_file


def _speedup(hit_rate: float) -> float:
    return min(1000.0, 1.0 / max(1.0 - hit_rate, 0.001))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="intellikv", description="Prefix-aware KV cache eviction policy simulator.")
    parser.add_argument("--trace", required=True, help="Trace path (.jsonl or .jsonl.gz)")
    parser.add_argument("--capacities", required=True, help="Comma-separated cache capacities in blocks")
    parser.add_argument("--policies", default=",".join(POLICIES), help=f"Comma-separated policies (default: {','.join(POLICIES)})")
    parser.add_argument("--block-size", type=int, default=None, help="Fallback block size for records that omit block_size")
    parser.add_argument("--warmup-fraction", type=float, default=0.5, help="Fraction of requests used as warmup (default: 0.5)")
    args = parser.parse_args(argv)

    capacities = sorted({int(part) for part in args.capacities.split(",") if part.strip()})
    policies = [part.strip().lower() for part in args.policies.split(",") if part.strip()]

    trace = parse_trace_file(args.trace, block_size=args.block_size)
    plan = build_execution_plan(trace, warmup_fraction=args.warmup_fraction)
    ceiling = simulate_ceiling(plan)

    print(f"Trace: {trace.request_count:,} requests, {len(trace.ids):,} block events, "
          f"{plan.unique_blocks:,} unique prefix blocks, block size {trace.block_size} tokens")
    print(f"Measurement: last {1 - args.warmup_fraction:.0%} of requests; "
          f"hit rate ceiling {ceiling.hit_rate:.1%} (infinite capacity)")
    print()

    width = 22
    header = f"{'Capacity (blocks)':>18}  " + "".join(f"{p:>{width}}" for p in policies)
    print(header)
    print("-" * len(header))
    for capacity in capacities:
        cells = []
        for policy in policies:
            result = simulate_policy(plan, policy, capacity)
            if result.underfilled:
                cells.append(f"{'underfilled':>{width}}")
            else:
                cells.append(f"{result.hit_rate:>{width - 8}.1%} ({_speedup(result.hit_rate):>4.1f}x)")
        print(f"{capacity:>18,}  " + "".join(cells))
    print()
    print("Speedup is the ideal prefill-only bound 1/(1-hit); underfilled = cache never "
          "filled before the measurement window (not under memory pressure).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
