# Admission-time reuse prediction — can we read the request and predict future prefix reuse?

*2026-07-21. Script: `autoresearch/admission_analysis.py` (run with
`PYTHONPATH=src python3 autoresearch/admission_analysis.py traces/day5_15k.jsonl.gz --real-only`).*

## Question

At the moment a request finishes, predict from what we can "read" of it
(tenant/system-prompt hash, turn count, user-message length, `?`-flag, output
length, block count, think-time) whether its prefix will be used again — and
decide whether / up to what point it should be saved. What accuracy is
achievable? (Prompt *text* is anonymized in ART-Chat, so only metadata is
testable here; LPC (NeurIPS 2025) is the text-embedding comparison point.)

Label: is the request's **deepest block** touched by any later request
(= the whole prefix was replayed = the conversation continued). Same task
framing as LPC. First half of trace trains, second half tests.

## Results (days 3, 5, 7 — 15k requests each, consistent across days)

**Whole population** (day5): combined logistic AUC **0.90**, accuracy **0.91**
(majority baseline 0.64). Misleading: 4.2k/15k requests are single-block,
empty-`sph` one-offs whose lone block is globally shared — "reuse" 99.5% —
while real conversations continue ~33%. Every feature separates these two
classes, so the headline number mostly measures junk detection.

**Real conversations only** (non-empty `sph`, >1 block; the honest task):

| metric | day3 | day5 | day7 |
|---|---|---|---|
| base rate, ever-continues | 0.33 | 0.32 | 0.33 |
| best single feature (think-time), AUC @64-req horizon | 0.60 | 0.59 | 0.63 |
| combined logistic AUC @64-req horizon | 0.60 | 0.62 | 0.62 |
| combined logistic AUC, ever-horizon | 0.56 | 0.57 | 0.57 |
| accuracy vs majority baseline | 0.684 / 0.674 | 0.690 / 0.684 | 0.675 / 0.671 |

Accuracy is within ~1 point of always-predicting-majority: per-request
continuation is close to unpredictable from metadata. Ranking retains some
value (save top-30% by score → precision ~0.38–0.40 vs base 0.33; at the
64-request horizon precision ~0.15–0.19 vs base 0.09–0.13, recall ~0.43–0.47),
which matches the overnight finding that AUC ≈ 0.6 signals only pay off
through γ-amplified *ranking*, never through binary keep/drop classification.

## At what point should the prefix be saved?

Two complementary answers:

**By position in the prompt** — reuse rate of newly created blocks when the
conversation does NOT continue (day5; ~identical days 3/7):

| pos ≤.25 | .25–.5 | .5–.75 | .75–.95 | tail |
|---|---|---|---|---|
| 0.56 | 0.46 | 0.43 | 0.42 | 0.30 |

When it does continue, every position is reused (1.00 — the whole prefix is
replayed). So the head ~25% of new blocks is worth saving *unconditionally*
(shared-prefix reuse across conversations of the same tenant gives it >50%
reuse regardless), and only the tail is the continuation gamble.

**By idle time (hazard curve, day5, censored at 30 min)** — the strongest
signal is not in the prompt at all, it's elapsed idle time:

| idle > | 0s | 15s | 30s | 60s | 120s | 300s | 600s |
|---|---|---|---|---|---|---|---|
| P(continues ≤30 min) | 0.36 | 0.26 | 0.22 | 0.14 | 0.065 | 0.020 | 0.007 |

A conversation idle 2 minutes is ~93% dead; idle 5 minutes ~98% dead. This is
why LRU is a strong baseline and why think-time is the best learned feature.

## Conclusion

A "read the prompt, predict future reuse" model on this workload:
- **as junk filter / population-level**: ~0.90 AUC, 0.91 accuracy — easy, and
  worth exploiting (never persist single-block empty-tenant requests beyond
  their shared head).
- **as per-conversation continuation predictor**: AUC 0.56–0.62, accuracy no
  better than majority class (best F1 ≈ 0.44 at the 50%-keep operating point,
  day 5, ever-horizon). Metadata caps out here. Literature comparison for the
  same task with real (non-anonymized) text: **LPC** (NeurIPS 2025, 118M
  e5-small over last 4 user prompts) reports MCC 0.28–0.39 and F1 0.63–0.69;
  **SAECache** (serving-model hidden state → 1M MLP) reports 77.1% accuracy,
  F1 0.803. So reading the actual text roughly moves F1 from ~0.45 to
  ~0.63–0.80 — better, but still far from clean separation; nobody reports
  AUC or calibration.
- **admission policy implied**: always save the head ≤25% of new blocks;
  rank tail blocks by continuation score and idle-time hazard rather than
  making a hard save/drop call. This is effectively what the IntelliKV
  eviction policy already converged to.

## Raw prompt-text smoke test

*2026-07-21. Script: `autoresearch/prompt_text_analysis.py`; raw ART-Chat
records streamed from day 5, no raw text written to disk. Features are capped,
hashed word unigrams/bigrams from the last user turn, last 4 user turns, and
system prompt, plus a role-sequence shape feature. The raw prompt text is
anonymized common-word word-soup, so this is testing lexical/template residue,
not human-readable semantics.*

On day5 first 5k requests, real conversations only (non-empty `sph`, >1
block), first half train / second half test, deepest-block ever-reuse label:

| model | AUC | F1 | acc |
|---|---:|---:|---:|
| metadata | 0.536 | 0.442 | 0.440 |
| last user text | 0.497 | 0.339 | 0.516 |
| last 4 user turns | 0.526 | 0.413 | 0.492 |
| system text | 0.542 | 0.392 | 0.557 |
| role sequence shape | 0.569 | 0.457 | 0.415 |
| system + last 4 text | 0.511 | 0.337 | 0.553 |
| metadata + system + last 4 | 0.521 | 0.315 | 0.602 |
| metadata + text + roles | 0.520 | 0.350 | 0.572 |

On the smaller 3k day5 sample at the 64-request horizon, metadata had AUC
0.594; last-user text reached 0.565; last-4/system text were ~0.50-0.54; and
text+metadata overfit below metadata alone.

Takeaway: in this anonymized dataset, actual lexical prompt text did not add a
stable improvement over cheap metadata. The one mildly useful "prompt" signal
was conversation shape / role pattern, which is structural rather than
semantic. A real-text embedding model may still help on non-anonymized traces
(as LPC/SAECache suggest), but ART-Chat's available text does not make a strong
case for putting prompt semantics at the center of IntelliKV.

## Real prompt-text follow-up: WildChat

*2026-07-21. Script: `autoresearch/wildchat_prompt_text_analysis.py`;
dataset: `allenai/WildChat`, streamed via Hugging Face row API. Raw text is
kept only in memory; features are hashed word unigrams/bigrams. Task:
first-turn-only continuation prediction -- from the initial user prompt,
predict whether the conversation later has another user turn. This avoids the
obvious all-turn leakage where the final transcript length reveals the label.*

On the first 5k WildChat rows, English non-redacted conversations only, first
half train / second half test:

| model | AUC | F1 | acc |
|---|---:|---:|---:|
| metadata / length / model | 0.621 | 0.646 | 0.595 |
| last user text | 0.639 | 0.567 | 0.598 |
| user-history text | 0.644 | 0.584 | 0.608 |
| prompt-history text | 0.618 | 0.570 | 0.583 |
| metadata + last user text | 0.670 | 0.615 | 0.641 |
| metadata + user-history text | 0.667 | 0.608 | 0.623 |
| metadata + prompt-history text | 0.653 | 0.595 | 0.616 |

Takeaway: unlike ART-Chat's anonymized text, real prompt text does carry signal.
On this clean first-turn continuation task, hashed lexical prompt features beat
the cheap causal metadata by ~0.02 AUC, and metadata + last-user text improves
by ~0.05 AUC over metadata alone. This is not yet a KV-cache hit-rate result:
WildChat lacks ART-Chat-style prefix block hashes / cache timing, so the test
is a public-data proxy for prompt-content reuse prediction rather than a drop-in
simulator trace.
