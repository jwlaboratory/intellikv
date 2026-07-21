# IntelliKV

The [KVCache.AI hit-rate simulator](https://kvcache.ai/tools/kv-cache-hit-rate-simulator/)
(`kvcache-simulator`, vendored from
[kvcache-ai/kvcache-blog](https://github.com/kvcache-ai/kvcache-blog/tree/main/packages/kvcache-simulator),
Apache-2.0 — see `LICENSE.md`), patched to support **custom eviction
policies** so you can plug in your own algorithm and benchmark it against
FIFO / LRU / Optimal on real or synthetic traces.

## Changes vs. upstream

- `src/kvcache_sim/policies.py`: new `simulate_custom()` — a policy template
  with three clearly-marked SCORING hooks (ships as LFU as a working example),
  registered under the policy name `custom`.
- `src/kvcache_sim/simulator.py` / `cli.py`: accept `custom` in `--policies`
  (Python backend only; the bundled C++ core still knows just fifo/lru/optimal).
- `src/kvcache_sim/resources/`: bundled `models.yaml` + C++ core, which
  upstream syncs from the blog at package build time (needed to run from a
  plain checkout).
- `examples/make_trace.py`: synthetic hot-conversations + scan-pollution trace
  generator; `tests/test_custom_policy.py`: sanity tests for the patch.

## Quick start

```bash
pip install -e .                      # or: PYTHONPATH=src python3 -m kvcache_sim ...
python3 examples/make_trace.py        # writes trace.jsonl

kvcache-simulator run \
  --trace trace.jsonl \
  --model llama-3.1-8b --kv-precision fp8_int8 \
  --backend python \
  --policies fifo,lru,custom,optimal \
  --budgets-gib 1,2,4,8,16,32 --no-progress
```

`kvcache-simulator list-models` shows the model catalog. All upstream options
(GiB budget sweeps, model/precision accounting, JSON output, real-trace
formats) work unchanged; see `src/kvcache_sim/` or the upstream README.

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

Run `--policies custom` with `--backend python` (the C++ backend will tell
you if you forget). Test with:

```bash
python3 -m unittest discover -s tests
```
