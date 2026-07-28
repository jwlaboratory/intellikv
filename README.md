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

# With per-request content metadata (recommended — the custom policy uses it)
python3 scripts/art_chat_trace_meta.py --days 20260401 --max-requests 100000 -o traces/day1_100k.jsonl.gz
```

`art_chat_trace_meta.py` additionally keeps a compact `meta` object per
request (output length, tenant hash, timestamp, turn count, last-user-message
stats — no raw text), which the `custom` policy consumes via
`plan.request_meta`.

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

## The IntelliKV policy (`custom`)

`simulate_custom` in `src/kvcache_sim/policies.py` implements a learned,
content-aware eviction policy — an LRU backbone reshaped by an online
censored survival model. It is causal (never reads the clairvoyant
`next_request_for_event` signal outside env-gated `oracle_*` diagnostic
modes), tiny (~90k integer counters ≈ a few hundred KB), and cheap
(O(1) table updates per block event, K=24 sampled leaves per eviction).

How it works:

1. **Observation** (cache-independent): every request's block path updates
   per-node state — depth, use count, logical gap, real think-time gap
   (`meta.ts`), plus per-request features: tenant popularity
   (`system_prompt_hash` rolling window), turn count (`nmsg`).
2. **Survival tables**: for each (feature bucket × idle-age bucket) it keeps
   a log-binned histogram of "requests until next touch", with multi-stage
   censoring so blocks that never return are counted at every idle age
   (no survivor bias). Feature planes: shallow blocks (depth ≤ 8):
   tenant × think-time × frequency; deep blocks: think-time × tenant ×
   frequency; traces without `meta` fall back to depth × freq × gap.
3. **Scoring**: eviction candidates are ranked by
   `score = (idle + 1) × ((1 − p_fb + ε) / (1 − p_g + ε))^γ`, where `p` is
   the (hierarchically shrunk) probability of returning within the current
   cache-residency window. Blocks likelier than typical to return soon get
   protected; likely-dead blocks get demoted. The clip (±24×) bounds the
   worst case near LRU.
4. **Residency adaptation**: γ scales with estimated residency
   (capacity / first-touch insert rate): small caches make near-term
   predictions decisive (γ=5), huge caches soften toward LRU (γ=2).

Results (hit-rate %, llama-3.1-8b fp8_int8; tuned only on day 1, days 2–7
held out; full sweeps in `autoresearch/results/v9final_*.json`):

| trace | budget | LRU | IntelliKV | Optimal |
|-------|--------|------|-----------|---------|
| day 5 (holdout, 15k) | 16 GiB | 8.6 | **19.7** | 34.5 |
| day 6 (holdout, 20k) | 32 GiB | 21.8 | **32.3** | 50.2 |
| day 7 (holdout, 100k) | 64 GiB | 25.7 | **32.9** | 54.3 |
| day 1 (train, 100k) | 64 GiB | 57.0 | **59.3** | 78.2 |

Across the full 7-day × 6-budget matrix (8–256 GiB), IntelliKV beats LRU in
41/42 cells (worst cell: −0.24 pts at the warmest/largest setting),
capturing 10–35% of the LRU→Optimal gap on held-out days, and roughly
doubling LRU's hit rate at small budgets on cold days. Robustness checks:
stable across seeds (±0.05), warmup fractions 0.3–0.7, budget extremes
2–1024 GiB, and shallow/deep threshold choices.

Env knobs (defaults are the validated config): `INTELLIKV_GAMMA` (`auto`),
`INTELLIKV_CLIP` (24), `INTELLIKV_DEEP_FEAT` (`tenant`), `INTELLIKV_SAMPLE`
(24), `INTELLIKV_SEED` (12345), `INTELLIKV_SHALLOW_DEPTH` (8),
`INTELLIKV_NO_META` (unset; set to force the structural-only model),
`INTELLIKV_CUSTOM_MODE` (`hazard`; `lru` for a sampled-LRU A/B, `oracle_*`
for non-deployable information-value diagnostics), `INTELLIKV_ADMIT` (0 =
off; ~6 refuses inserting predicted-dead blocks — helps cold workloads
slightly, hurts warm ones), and
`INTELLIKV_SAVE_TABLES` / `INTELLIKV_PRIOR` / `INTELLIKV_PRIOR_SHRINK`
(save a run's learned tables — ~400 KB JSON — and warm-start another run
from them; helps cold-start on short traces).

The research journal behind this policy — baselines, feature analyses,
oracle decompositions, failed attempts and all — is in
`autoresearch/log.md`.

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
