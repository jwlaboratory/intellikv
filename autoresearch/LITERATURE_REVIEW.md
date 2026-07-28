# Literature review: prompt-content-aware KV prefix-cache eviction (prior art for IntelliKV)

*Reviewed 2026-07-21 via multi-agent deep-research sweep (98 agents, 16 primary sources fetched, 25 claims adversarially verified with 3-vote panels; 22 confirmed, 3 refuted).*

## Question

Has anyone used the actual LLM prompt content (semantics/category/embedding of the request text) as a signal to guide KV prefix-cache eviction or LRU-style cache management — as distinct from (a) attention-score token-level KV compression (H2O, SnapKV) and (b) plain LRU on prefix trees?

## Verdict

**Yes — directly.** The core idea behind IntelliKV ("a learned, prompt-feature-driven eviction policy for prefix KV cache that beats LRU") is substantially anticipated by at least three works, one of them peer-reviewed at NeurIPS 2025. Novelty would need to rest on specific feature design, workload generality, or methodology — not on the idea itself.

## What IntelliKV does (for comparison)

IntelliKV (`simulate_custom` in `src/kvcache_sim/policies.py`; full details in
`autoresearch/FINDINGS.md`) is a prefix KV-cache eviction policy: an **LRU
backbone reshaped by an online censored survival model** over request-log
metadata and structural features. Per-block features: tenant
(system-prompt-hash) popularity, observed think-time since the conversation's
previous turn, turn count, trie depth, use count, logical gap. A survival
table per (feature-bucket × idle-age-bucket) accumulates log-binned
"requests-until-next-touch" outcomes with multi-stage censoring (blocks that
never return are counted at every idle age). Eviction ranks K=24 sampled
candidates by `(idle+1) × ((1−p_fb)/(1−p_glob))^γ`, where γ adapts to cache
residency — decisive at small caches, near-LRU at huge ones — and a ±24× clip
bounds the worst case near LRU.

Key operational properties that matter for positioning:
- **No auxiliary model**: ~90k integer counters (~400 KB), O(1) updates,
  integer table lookups at eviction time. It never embeds or reads the prompt
  *text* — it uses zero-cost request-log metadata (tenant identity,
  think-time, turn count). Notably, anonymized text content itself measured
  useless in feature analysis (AUC 0.500).
- **Results**: +2 to +9 hit-rate points over LRU across 41/42 cells
  (7 days × 6 budgets, 8–256 GiB, ART-Chat-2.5M traces at 100k requests/day);
  tuned only on day 1, all 36 holdout cells positive; captures 10–31% of the
  LRU→Optimal gap; roughly 2× LRU at small budgets on cold days.

## Closest prior art (category c: prompt content → eviction)

### 1. LPC — Learned Prefix Caching (NeurIPS 2025) ★ closest match
- Yang, Li, Li, Lloyd (Princeton). Peer-reviewed.
- Explicitly claims to be **"the first learned method to perform LLM prefix cache eviction."**
- Uses a 118M-parameter e5-small **text embedding model over conversation content** (last 4 user prompts + turn count → frozen multilingual-e5-small → 3-layer MLP) to predict which conversations will continue; score is time-decayed and drives a min-heap eviction order (max-pooled over shared prefixes). Eviction only — everything is admitted.
- Beats LRU: 18–47% cache-size reduction at equal hit ratio.
- **Predictor accuracy in isolation** (from the official NeurIPS poster — OpenReview PDF is
  challenge-gated to automated access; poster tables 1–2): MCC **0.39** on LMSys / **0.28** on
  ShareGPT; F1 **0.68–0.69** LMSys / **0.63** ShareGPT. A turn-count-only baseline gets MCC 0.35
  on LMSys but **0.00** on ShareGPT — text content carries most of the signal on ShareGPT and
  little marginal signal on LMSys. No AUC or calibration reported.
- Sources: https://neurips.cc/virtual/2025/poster/117662 , https://openreview.net/pdf?id=Vj48eXaQDM
- Differentiation opening: LPC assigns one conversation-level score to all blocks in a session — feature granularity and non-conversational workloads are not covered.

### 2. SAECache (arXiv 2605.18825, May 2026, preprint)
- Semantic-adaptive prefix-cache eviction: routes KV blocks to queues by **semantic token type** (system prompt / user query / tool output / response / chain-of-thought; reuse rates vary up to 756×), learns per-type reuse value **online via eviction feedback**.
- Beats vLLM-style LRU: 1.4–2.7× TTFT, +4.8–5.9pp hit ratio.
- Uses structural token-type categories, not prompt embeddings. Calls LPC "the closest predecessor to our work."
- **Predictor accuracy in isolation** (continuation classifier: serving model's final-layer hidden
  state of last input token → ~1M-param MLP): **77.1% accuracy, F1 0.803** (paper Fig. 17
  confusion matrix, 175 continuation / 115 stopping test cases; errors biased toward predicting
  continuation). Eval-dataset identity and split sizes unstated.

### 3. CacheWise (arXiv 2606.16824, June 2026, preprint)
- **TF-IDF embeddings of tool-call arguments** (KMeans-clustered) feed a learned reuse-time predictor that **replaces vLLM's LRU**; evicts blocks in decreasing order of predicted time-to-next-reuse.
- 2–2.6× fewer evictions, up to 3.5× faster session completion.
- Content signal is tool-call arguments in agentic coding sessions (lexical TF-IDF, not deep semantics).

## Admission-time saving ("dynamic KV block caching") — supplemental check 2026-07-21

Focused follow-up sweep on the *admission* question (decide at request completion
whether/which prefix blocks to persist, vs. eviction-time ranking):

- **Marconi** (MLSys 2025, arXiv 2411.19379): the main admission prior art, but purely
  **structural heuristics** — speculative radix-tree insertion admits shared-prefix entries only
  when insertion creates a branch point (effectively cache-on-second-occurrence); for
  conversation history it checkpoints only the last decoded token's SSM state. No learned model,
  no isolated accuracy numbers.
- **PEEK** (arXiv 2607.02525): admission *ordering* from the pending request queue (admit
  "cluster pioneers" first) — known future demand, not prediction.
- **SGLang HiCache `write_through_selective`**: production admission control via a hit-count
  threshold (TinyLFU-flavored) for tier promotion. Not predictive, not content-based.
- Name-collision warning: "KV Admission / Write-Gated KV" (arXiv 2512.17452) is token-level
  write gating within one request (H2O/SnapKV lineage), NOT prefix-block admission.
- **Gap confirmed**: no paper found makes a *learned, prompt-content-driven admission decision*
  at request completion, and none reports admission-decision accuracy in isolation. LPC/SAECache
  predictions moved to admission time appears unstaked. (But see
  `autoresearch/ADMISSION_PREDICTION.md`: on ART-Chat, hard save/drop admission among real
  conversations is near-unpredictable from metadata — the gap may be open because the win is
  small; a paper would need real text and probably soft ranking rather than binary admission.)

## "Learned > LRU" is established even without semantics

- **LARU/LCR** (arXiv 2509.20979, Alibaba): replaces SGLang RadixTree's LRU with an online LightGBM model predicting per-node next-request times. P99 TTFT −13.5–28.3% vs LRU on production traces. Features are access statistics + shallow metadata (request-interval deltas, decayed counters, session turns, prompt length) — zero semantic features.
- **Alibaba/SJTU production study** (arXiv 2506.02634, USENIX ATC 2025): per-request-category reuse-probability distributions (gateway request type + turn number) drive eviction; explicitly anti-"workload-agnostic LRU". No prompt text by construction (SipHash-anonymized traces). Its exact benchmark numbers failed verification (1–2 vote) — re-check before citing.
- **PEEK** (arXiv 2607.02525): overrides LRU/LFU in SGLang and vLLM using pending-queue demand over a radix trie (depth-weighted pending counts). Queue-structural, not semantic/learned. Code: github.com/xiexbing/peek
- **Marconi** (MLSys 2025): admission by forecast reuse likelihood + eviction by recency + FLOP-efficiency. All signals structural; no semantics/embeddings anywhere in the paper.
- **KVFlow** (arXiv 2507.07400): workflow-aware eviction for agent graphs (steps-to-execution), application-structure-aware, not prompt-semantics.

## Confirmed non-overlapping (category a — don't confuse)

- **KVP** (Apple, ICML 2026): RL-learned eviction but token-level within a sequence (H2O/SnapKV family), per-attention-head agents on key/value vectors. No cross-request reuse prediction, no prompt content.
- vLLM baseline confirmed non-semantic: token-hash block identity, LRU free-queue eviction (with reverse-depth-order re-queueing as the only structural tweak). No learned/predictive/content-based policy in the design doc.

## Where IntelliKV could still claim novelty

Given what IntelliKV actually is (metadata-driven, no auxiliary model — see above), the openings are sharper than "prompt semantics":

1. **Zero-cost metadata features, especially tenant identity as a reuse predictor.** LPC needs a 118M embedding-model forward pass; SAECache needs serving-model hidden states; CacheWise needs TF-IDF + KMeans over tool arguments. IntelliKV costs integer table lookups over request-log metadata. No surveyed paper uses tenant/system-prompt-hash popularity as a reuse signal (tenant-awareness elsewhere is only security/fairness). This also reframes the positioning: IntelliKV is honestly *not* a prompt-semantics policy — its own feature analysis found anonymized text useless (AUC 0.500) — so LPC/SAECache/CacheWise anticipate the *category* but not the *mechanism*. The nearest mechanism-level neighbor is LARU/LCR (shallow metadata + LightGBM), which must be cited and ideally benchmarked against.
2. **Explicitly censored online survival estimation** — no systems eviction paper formalizes never-returning blocks as censored observations (LHD/LRB handle it implicitly or by label capping).
3. **LRU × bounded learned multiplier with residency-adaptive amplification (γ)** as the robustness mechanism — prior anchored-learning work uses fallback (Cold-RL), candidate restriction (HALP, LARU), or marker phases, not bounded multiplicative modulation of the LRU score.
4. **Head-to-head evidence** — no source benchmarks LPC vs SAECache vs CacheWise vs LARU on a common trace; whether semantic features beat shallow metadata at equal cost is open. IntelliKV's think-time/tenant results are indirect evidence for the metadata side; a controlled comparison would itself be a contribution.

## Caveats

- Fast-moving space: SAECache, CacheWise, PEEK, KVP are all preprints from the last ~6 months; more overlapping work is likely in submission. Only LPC (NeurIPS 2025), Marconi (MLSys 2025), and KVP (ICML 2026) are confirmed peer-reviewed.
- LMCache, Mooncake, and proprietary production caching tiers were not deeply verified — their shipped eviction policies may include content signals beyond what papers describe.
- Pre-2025 general learned-caching literature (LRB, Parrot, CDN eviction) was not swept for request-content features.

---

# Part 2: Multi-tier KV placement (HBM / RAM / disk / RDMA / NONE)

*Second deep-research sweep, 2026-07-21 (103 agents, 25 claims verified: 24 confirmed, 1 refuted). Question: has anyone built policies that decide WHICH storage tier a KV block goes to, including "don't store / recompute" as an explicit choice?*

## Verdict

**Multi-tier KV placement itself is established prior art, but learned, request-log-metadata-driven tier selection is largely unstaked.** The pieces exist separately; nobody peer-reviewed combines them. One simulation-only preprint (arXiv 2604.26968) is the closest neighbor and partially stakes the recompute-economics angle.

## What exists (all verified against primary sources)

### Tiered KV systems — placement by heuristic or scheduler hints
- **CachedAttention / AttentionStore (USENIX ATC 2024)** — hierarchical KV cache across GPU HBM / host memory / disk for multi-turn conversations. Placement is *scheduler-aware* (job-queue hints, look-ahead prefetch/eviction), explicitly contrasted with LRU/FIFO. But: whole-conversation granularity (not per-block), heuristic (not learned), no NONE/recompute tier. ([arXiv 2403.19708](https://arxiv.org/abs/2403.19708))
- **IMPRESS (FAST 2025)** — GPU/CPU/disk; scores each chunk by access frequency × proportion of important tokens (attention-weight probes) to decide tier residency. Model-internal content signal, not request metadata; no recompute tier.
- **KVDrive (arXiv 2605.18071)** — HBM/DRAM/SSD; final-prompt-token attention profiles drive importance-guided HBM warm-up and lowest-attention eviction. Demotes and re-fetches; no recompute-vs-fetch economics.

### The closest neighbor to a learned tier policy ⚠
- **"Predictive Multi-Tier Memory Management for KV Cache" (arXiv 2604.26968, Apr 2026)** — Bayesian per-block tier placement over six tiers (HBM3 / DRAM / CXL 3.0 / NVMe+GDS / RDMA / parallel FS). Maintains Beta-distribution reuse predictors over 16 (block-type × transition-type) pairs — block types {system_prompt, tool_context, user_context, intermediate_reasoning}, transitions {same_tool_repeat, tool_switch, reasoning_step, agent_handoff} — and its value score **does** weigh recomputation cost against per-tier storage cost (a claim that it doesn't was refuted 0–3 in verification). Weaknesses: single-author, non-peer-reviewed, **simulation-only with "projected" cluster numbers**, and signals are semantic block categories, not request-log metadata. This is the paper IntelliKV's tier extension must differentiate against.

### Recompute-vs-storage economics — exists in fragments, never as a learned tier choice
- **Marconi (MLSys 2025)** — store-vs-recompute admission + FLOPs-per-byte eviction utility, but entirely within a single provisioned cache pool; no tier selection.
- **Cake (ICML 2025, arXiv 2410.03065)** — per-chunk recompute-vs-load decision, but only at *retrieval* time via positional cost asymmetry (compute early chunks while loading late chunks, two pointers converging); no learned model, no placement decision.

### Confirmed single-tier (full-text keyword audits)
- LPC and SAECache — the learned-eviction works closest to IntelliKV — are both explicitly single-tier. Learned multi-tier placement is unstaked by them.

## What's unstaked for an IntelliKV tier extension

No verified prior work combines all three of:
1. **Request-log metadata signals** (tenant popularity, think-time, turn count) driving the tier decision;
2. **Per-block placement across a full HBM/RAM/disk/RDMA hierarchy**;
3. **NONE/recompute as a first-class tier** priced against per-tier storage + fetch cost.

Existing tiered systems use scheduler hints or attention content; existing learned/metadata policies (LPC, SAECache, LARU, Marconi, IntelliKV itself) are single-tier; the one learned multi-tier + recompute-economics system (2604.26968) is simulation-only with semantic block types. Differentiation levers: signal type (request-log metadata), evaluation realism (real traces vs. simulation projections), and unified learned policy treating NONE as a tier.

## Open questions worth answering (from verification)

- Do Mooncake / LMCache / SGLang HiCache / NVIDIA Dynamo / vLLM KV-connectors ship anything beyond waterfall-LRU demotion, and do any expose hooks for a pluggable per-block tier policy? **(Not adversarially verified in this sweep — the "unstaked" conclusion assumes they don't; check before claiming.)**
- What is the measured recompute-vs-fetch break-even across tiers on current hardware (SSD/RDMA bandwidth vs. prefix length)? This determines whether a NONE tier materially changes serving cost.
- Are request-log metadata signals and attention-importance signals (IMPRESS/KVDrive) complementary or does one subsume the other for tier prediction?

## Caveats (part 2)

- Coverage gap: the named production systems (Mooncake, LMCache, vLLM connector, HiCache, Dynamo, InfiniGen, FlexGen, Pensieve, CacheGen) were not deeply verified in this sweep.
- Acute time-sensitivity: KVDrive, SAECache, and 2604.26968 are all preprints from the last ~3 months; 2604.26968 is under review and could strengthen.
- 2604.26968's *mechanism* description is verified against its text, but its performance numbers are projections — do not trust or cite them as results.
