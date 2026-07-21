# IntelliKV

Minimal KV-cache eviction policy simulator. Replay a JSONL request trace
against a prefix-aware block cache and compare hit rates across eviction
policies — including your own.

Core semantics (matching the [KVCache.AI hit-rate simulator](https://kvcache.ai/tools/kv-cache-hit-rate-simulator/),
whose open-source package this is adapted from — see `THIRD_PARTY_NOTICES.md`):

- The cache is a **prefix trie**: a block only counts as a hit while its entire
  prefix is cached, so eviction always removes leaves.
- Hit tokens count only the longest continuous cached prefix of each request.
- Hit rate is measured over the last 50% of requests; a capacity that never
  fills before that window is reported as `underfilled`.

No dependencies; Python 3.10+.

## Quick start

```bash
python3 examples/make_trace.py           # writes trace.jsonl
python3 -m intellikv --trace trace.jsonl --capacities 256,512,1024,2048,4096
```

## Writing your own eviction policy

Edit the **SCORING** section of `simulate_custom` in `intellikv/policies.py`.
The cached leaf with the smallest score is evicted first; you control the
score via three hooks (`score_on_insert`, `score_on_hit`,
`score_on_becomes_leaf`). All trie bookkeeping, prefix hit accounting, and
warmup semantics are handled for you. As shipped, `custom` implements LFU as
a working example.

Useful signals inside the hooks:

- `plan.tokens[event_index]` — token weight of the block
- `plan.next_request_for_event[event_index]` — clairvoyant next-reuse (what `optimal` uses)
- `plan.parent[node]` — walk toward the root for depth-based scores

To add more policies side by side, write another `simulate_*` function and
register it in the `POLICIES` dict at the bottom of `policies.py`.

```bash
python3 -m unittest discover -s tests    # sanity checks, incl. custom vs lru
```

## Trace format

JSONL, one request per line, replayed in file order:

```json
{"block_size": 64, "hash_ids": [2001, 2002], "input_length": 128}
```

- `hash_ids`: cache block identities (ints or strings) in prefix order
- `input_length`: prefill token count for the request
- `block_size`: tokens per block (or pass `--block-size` as a fallback)

Real production traces in this format are linked from the KVCache.AI presets
(Mooncake FAST'25, Qwen Bailian, BurstGPT, and others).

## Capacity in blocks vs. GiB

Capacity is specified directly in blocks. To convert a memory budget for a
real model: `blocks = GiB × 2³⁰ / (bytes_per_token × block_size)`, where for
a standard GQA model `bytes_per_token = 2 × layers × kv_heads × head_dim ×
bytes_per_element` (e.g. Llama 3.1 8B at FP8 is 64 KiB/token, so 1 GiB holds
256 blocks of 64 tokens).
