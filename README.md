# IntelliKV

A fork of the [KVCache.AI hit-rate simulator](https://kvcache.ai/tools/kv-cache-hit-rate-simulator/)
(`kvcache-simulator`, vendored from
[kvcache-ai/kvcache-blog](https://github.com/kvcache-ai/kvcache-blog/tree/main/packages/kvcache-simulator),
Apache-2.0 — see `LICENSE.md`) with two additions:

1. **Custom eviction policies** — plug in your own algorithm and benchmark it
   against FIFO / LRU / Optimal.
2. **ART-Chat-2.5M integration** — build traces from the
   [alessiotoniolo/ART-Chat-2.5M](https://huggingface.co/datasets/alessiotoniolo/ART-Chat-2.5M)
   production chatbot trace (2.5M requests, ~18k avg input tokens, high prefix reuse).

## Setup

```bash
pip install -e .        # no dependencies; Python 3.10+
```

## Getting a trace

`scripts/art_chat_trace.py` streams the dataset's daily JSONL files from
HuggingFace and strips each record to what the simulator needs
(`hash_ids`, `input_length`, `block_size: 256`), sorted by timestamp — the
full 148 GB is never downloaded or stored.

```bash
# Quick sample: first 20k requests of day 1
python3 scripts/art_chat_trace.py --days 20260401 --max-requests 20000 -o art_chat_20k.jsonl.gz

# One full day / the whole week
python3 scripts/art_chat_trace.py --days 20260401 -o art_chat_day1.jsonl.gz
python3 scripts/art_chat_trace.py --days all -o art_chat_week.jsonl.gz
```

## Running the simulator

```bash
kvcache-simulator run \
  --trace art_chat_20k.jsonl.gz \
  --model llama-3.1-8b --kv-precision fp8_int8 \
  --backend python \
  --policies fifo,lru,custom,optimal \
  --no-progress
```

`kvcache-simulator list-models` shows the model catalog. All upstream
features (GiB budget sweeps, model/precision accounting, `--format json`,
the fast C++ backend for the built-in policies) work unchanged — but
`custom` requires `--backend python`, since the bundled C++ core only
implements fifo/lru/optimal.

## Writing your own eviction policy

Edit the **SCORING** section of `simulate_custom` in
`src/kvcache_sim/policies.py`. The cache is a prefix trie — a block only hits
while its whole prefix is cached — so eviction always removes leaves; the
cached leaf with the smallest score is evicted first. You control the score
via three hooks:

- `score_on_insert(node, event_index)` — block enters the cache
- `score_on_hit(node, event_index)` — cached block reused on a live prefix
- `score_on_becomes_leaf(node)` — node became evictable again

Useful signals: `plan.tokens[event_index]` (block token weight),
`plan.next_request_for_event[event_index]` (clairvoyant next reuse — what
Optimal uses), `plan.parent[node]` (walk toward the root for depth-based
scores). Everything else — leaf-only eviction, longest-cached-prefix hit
accounting, warmup and underfilled semantics — is handled for you.

As shipped, `custom` implements LFU purely as a plumbing example — on
ART-Chat it loses badly to LRU (7.6% vs 60.9% hit rate at 64 GiB on a 20k
sample), since conversations keep extending their prefix and pure frequency
hoards shallow system-prompt blocks. The gap to beat is LRU → Optimal (60.9%
→ 78.3% on that same sample).

```bash
python3 -m unittest discover -s tests    # includes sanity tests for the patch
```

## Changes vs. upstream

- `src/kvcache_sim/policies.py`: added `simulate_custom()` and the `custom`
  policy name.
- `src/kvcache_sim/simulator.py` / `cli.py`: accept `custom` in `--policies`
  (Python backend only).
- `src/kvcache_sim/resources/`: bundled `models.yaml` + C++ core, which
  upstream injects at package build time (needed to run from a checkout).
- `scripts/art_chat_trace.py` and `tests/test_custom_policy.py` are new.
