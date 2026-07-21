"""Generate a synthetic trace: hot conversations plus one-off scan traffic.

Every request shares an 8-block system prompt. Hot conversations add 16
stable blocks each and are revisited often; scan requests add 30 blocks that
are never seen again (cache pollution). Recency-based policies suffer here;
frequency-based ones should not — useful for sanity-checking a custom policy.

    python3 examples/make_trace.py [output.jsonl]
"""
import json
import random
import sys

BLOCK = 64
SYSTEM_BLOCKS = [f"sys-{i}" for i in range(8)]
N_HOT = 60
HOT_BLOCKS = {c: [f"hot-{c}-{i}" for i in range(16)] for c in range(N_HOT)}
N_REQUESTS = 4000
P_HOT = 0.5

out = sys.argv[1] if len(sys.argv) > 1 else "trace.jsonl"
rng = random.Random(42)
scans = 0

with open(out, "w") as f:
    for _ in range(N_REQUESTS):
        if rng.random() < P_HOT:
            ids = SYSTEM_BLOCKS + HOT_BLOCKS[rng.randrange(N_HOT)]
        else:
            ids = SYSTEM_BLOCKS + [f"scan-{scans}-{i}" for i in range(30)]
            scans += 1
        f.write(json.dumps({"block_size": BLOCK, "hash_ids": ids, "input_length": BLOCK * len(ids)}) + "\n")

print(f"wrote {out}: {N_REQUESTS} requests ({scans} scans)")
