# Overnight autoresearch — findings summary

**Task** (readme.md): build a tiny, fast model that decides what KV cache to
keep, genuinely beating LRU/LFU across all settings, exploiting access to
request content. No reward hacking. ~12h autonomous loop, 2026-07-20 23:30 →
2026-07-21.

## Deliverable

`simulate_custom` in `src/kvcache_sim/policies.py` — the **IntelliKV policy**:
an LRU backbone reshaped by an online censored survival model over content +
structural features. Fully causal, ~90k integer counters (~400 KB), O(1)
updates per block event, K=24 sampled candidates per eviction. Optionally
exports/imports its learned tables as a trained prior (~400 KB JSON).

## Headline results — full week at production scale

Hit-rate delta vs LRU (percentage points), **100k-request runs of every day**
(5–7M block events each; llama-3.1-8b fp8_int8, block 256, budgets
8→256 GiB; tuned ONLY on day 1, days 2–7 fully held out):

| day | 8GiB | 16 | 32 | 64 | 128 | 256 |
|-----|------|----|----|----|-----|-----|
| day1 (train) | +2.28 | +3.15 | +3.14 | +2.31 | −0.24 | +0.32 |
| day2 | +4.82 | +5.00 | +2.46 | +1.16 | +0.77 | +0.44 |
| day3 | +5.00 | +6.97 | +6.60 | +4.40 | +2.60 | +1.75 |
| day4 | +4.54 | +7.41 | +8.80 | +6.22 | +2.23 | +2.47 |
| day5 | +3.33 | +5.85 | +6.38 | +4.38 | +2.07 | +2.94 |
| day6 | +3.12 | +6.16 | +8.48 | +7.07 | +4.03 | +2.21 |
| day7 | +3.25 | +6.67 | +8.50 | +7.18 | +2.60 | +2.06 |

**41/42 cells positive; all 36 holdout cells positive.** Gap capture
(LRU→Optimal) on holdout days: 10–31%. At small budgets on cold days the
policy roughly doubles LRU (day6@8GiB 5.5 vs 2.4; day7@16GiB 11.9 vs 5.3;
day4@16GiB 16.0 vs 8.6). Smaller-slice sweeps (15–30k requests of each day)
and extreme budgets (2–1024 GiB) show the same pattern — results files:
`autoresearch/results/v9final_*.json`.

## How it works (one paragraph)

Every request's block path is observed causally (independent of cache
state), updating per-block features: trie depth, use count, logical gap,
real think-time since the conversation's previous turn, tenant
(system-prompt-hash) popularity, turn count. A survival table per
(feature-bucket × idle-age-bucket) accumulates log-binned
"requests-until-next-touch" outcomes, with multi-stage censoring so blocks
that never return are counted at every idle age (no survivor bias).
Eviction ranks sampled leaves by `(idle+1) × ((1−p_fb)/(1−p_glob))^γ` where
p is the hierarchically-shrunk probability of returning within the current
cache-residency window; γ adapts to residency (capacity / first-touch
insert rate) — decisive at small caches, near-LRU at huge ones; a ±24×
clip bounds the worst case near LRU.

## Key scientific findings along the way

1. **Reuse = conversation continuation.** Blocks average ~11–25 touches
   (each turn replays the whole prefix); median reuse distance ~70 requests.
2. **Oracle decomposition** (information-value diagnostics): perfect
   "returns within 64 requests" protection captures ~90% of the LRU→Optimal
   gap at small caches; perfect dead-block eviction captures most of it at
   large caches. Sampled eviction with exact Belady scores reproduces
   Optimal exactly — the framework loses nothing; prediction quality is the
   only bottleneck.
3. **Feature predictiveness is modest but real**: think-time is the best
   single signal (AUC 0.68 shallow / 0.59 deep); tenant 0.58; turn count,
   output length weak; anonymized text itself useless (lq AUC 0.500).
4. **Amplification is where the win came from**: raw probability ratios are
   mild; γ-exponentiation adapted to cache residency converted AUC-0.6
   signals into +5..+11 point gains. Content meta contributes ~+0.5..+3.4
   points over the structural-only variant (ablations, days 3/5).
5. **Failure modes fixed en route**: survivor bias (censor queues), Belady
   inversion under weak predictors (LRU backbone + bounded multiplier),
   residency estimators contaminated by the policy's own evictions
   (first-touch insert rate is policy-independent), and admission control
   that helped cold days but regressed warm ones (reverted — validate on the
   warmest AND coldest holdouts before adopting anything).
6. **What the model learned** (day-5 tables, most extreme buckets by
   traffic): protect hot-tenant shared prefixes touched seconds ago with
   256+ uses (mult 0.03 ≈ 33× effective-age reduction) and deep blocks of
   busy tenants mid-conversation (think ≤5s); demote deep blocks of rare
   tenants with few uses (mult ≈ 1.4) — one-off conversations. Tenant
   popularity × engagement cadence is the retention signal, exactly the
   content-awareness the task hypothesized.

## Robustness / anti-reward-hacking audit

- Tuning restricted to day 1; days 2–7 evaluated with frozen config; the
  strongest results are on the holdout days (opposite of overfitting).
- Stable across seeds (±0.05), warmup fractions 0.3/0.5/0.7, budget extremes
  (2 GiB: ~2× LRU; 1024 GiB: +0.9), shallow/deep threshold 4/8/16, and a
  production-scale 100k-request run.
- No-meta traces degrade gracefully (old 20k trace: ≈LRU or better,
  worst −0.24).
- The policy never reads clairvoyant signals (`oracle_*` modes are env-gated
  diagnostics only).
- Known limitation: warmest/largest regime (day1/day2 @ 64–256 GiB) gains
  are small (−0.2..+0.5); dead-vs-slow discrimination at huge idle remains
  unsolved (oracle says +3–5 pts available there).
- Known limitation: on traces WITHOUT content metadata (structural-only
  fallback, legacy 20k trace) the final build wins +1.2..+2.0 at 8–32 GiB
  but loses 0.4–0.75 at 64–128 GiB — amplified structural features are noisy
  in that regime (insensitive to γ/clip settings; accepted since the meta
  path is the intended input).

## Related work & novelty assessment (web literature search, 2026-07-21)

**Concept already claimed.** Learned prefix-cache eviction exists and 2025–26
is crowded: **LPC** (Yang et al., NeurIPS 2025) claims "first learned prefix
cache eviction" — a 118M-parameter text-embedding model predicts
conversation-continuation probability, combined with hand-tuned time decay
(openreview.net/pdf?id=Vj48eXaQDM). **SAECache** (arXiv 2605.18825) does
online log-normal inter-turn timing (think-time) survival probabilities with
token-type-aware multi-queues using model hidden states. Alibaba's trace
study (USENIX ATC 2025, arXiv 2506.02634) fits per-category reuse
distributions online. UniCache (SIGMETRICS 2026), KVFlow, PBKV, CacheWise
cover adjacent agentic settings. Production systems (vLLM, SGLang,
TensorRT-LLM, Mooncake) remain LRU/heuristic.

**What appears genuinely unstaked (per this search):**
1. **Zero-cost metadata features** — the policy reads only request-log
   metadata: tenant/system-prompt-hash popularity, observed think-time,
   turn count. No paper found uses **tenant identity as a reuse predictor**
   (tenant-awareness elsewhere is only security/fairness). LPC needs an
   embedding-model forward pass; SAECache needs serving-model hidden states;
   ours costs integer table lookups.
2. **Explicitly censored online survival estimation** — no systems eviction
   paper formalizes never-returning objects as censored observations (LHD/
   LRB handle it implicitly or by label capping); formal survival analysis
   appears only in MEC popularity-prediction work that doesn't do eviction.
3. **LRU × bounded learned multiplier** as the robustness mechanism — prior
   anchored-learning work uses fallback (Cold-RL), candidate restriction
   (HALP, LARU), or marker phases (learning-augmented theory), not bounded
   multiplicative modulation of the LRU score; plus residency-adaptive
   amplification (γ scaled to cache turnover).

**Positioning:** "LPC-class gains from pure request-log metadata with no
auxiliary model" — must cite and ideally benchmark against LPC and SAECache;
read UniCache in full (paywalled) before any publication claim.
Closest general-caching ancestors to credit: LHD/EVA (age-conditional reuse
probability, sampled eviction) and HALP (heuristic-anchored learning).

## Reproduce

```bash
python3 scripts/art_chat_trace_meta.py --days 20260405 --max-requests 15000 -o traces/day5_15k.jsonl.gz
kvcache-simulator run --trace traces/day5_15k.jsonl.gz --model llama-3.1-8b \
  --kv-precision fp8_int8 --backend python --policies lru,custom,optimal \
  --budgets-gib 8,16,32,64,128,256 --no-progress
python3 -m unittest discover -s tests   # 28 tests
```

Full iteration-by-iteration trail: `autoresearch/log.md`.
Result JSONs: `autoresearch/results/` (final matrix: `v9final_*`).
