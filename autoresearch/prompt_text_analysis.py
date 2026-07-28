"""Prompt-text reuse prediction on raw ART-Chat records.

This is the admission-time task from admission_analysis.py, but the input is
the raw ART-Chat JSONL stream so we can test whether anonymized prompt text
adds signal beyond the compact metadata currently stored in traces.

The script does not write raw text. It keeps capped, hashed lexical features
from the system prompt and recent user turns in memory, then trains a tiny
pure-Python logistic model on the first half of a day and evaluates on the
second half.

Run:
    PYTHONPATH=src python3 autoresearch/prompt_text_analysis.py --day 20260405 --max-requests 15000 --real-only
"""
from __future__ import annotations

import argparse
import bisect
import collections
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
import re
import sys
from typing import Iterable
import urllib.request


URL = "https://huggingface.co/datasets/alessiotoniolo/ART-Chat-2.5M/resolve/main/jsonl/artchat_week_{day}.jsonl"
DAYS = ("20260401", "20260402", "20260403", "20260404", "20260405", "20260406", "20260407")
NEVER = 1 << 60
WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass
class Example:
    ts: int
    sph: str
    out: int
    nmsg: int
    lulen: int
    lq: int
    hash_ids: list[str]
    last_user: str
    last4_user: str
    system_text: str
    all_roles: tuple[str, ...]


def bucket(value: float, edges: tuple[float, ...]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def stable_hash(text: str, bins: int) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little") % bins


def words(text: str, limit: int) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for match in WORD_RE.finditer(text.lower()):
        out.append(match.group(0))
        if len(out) >= limit:
            break
    return out


def text_features(prefix: str, text: str, *, offset: int, bins: int, max_words: int) -> dict[int, float]:
    toks = words(text, max_words)
    counts: collections.Counter[int] = collections.Counter()
    for i, tok in enumerate(toks):
        counts[offset + stable_hash(f"{prefix}:1:{tok}", bins)] += 1
        if i:
            counts[offset + stable_hash(f"{prefix}:2:{toks[i - 1]}_{tok}", bins)] += 1
    if toks:
        counts[offset + stable_hash(f"{prefix}:len:{bucket(len(toks), (16, 64, 256, 1024))}", bins)] += 1
        counts[offset + stable_hash(f"{prefix}:uniq:{bucket(len(set(toks)) / len(toks), (0.08, 0.12, 0.20, 0.35))}", bins)] += 1
    return {k: math.log1p(v) for k, v in counts.items()}


def merge_features(*parts: dict[int, float]) -> dict[int, float]:
    merged: dict[int, float] = {}
    for part in parts:
        for key, value in part.items():
            merged[key] = merged.get(key, 0.0) + value
    return merged


def auc(pairs: list[tuple[float, int]]) -> float:
    pos = sum(label for _score, label in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    pairs = sorted(pairs, key=lambda p: p[0])
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (2 * rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum += avg
        rank += j - i
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def f1_at_threshold(scores: list[float], labels: list[int], threshold: float) -> float:
    tp = fp = fn = 0
    for score, label in zip(scores, labels):
        pred = 1 if score >= threshold else 0
        if pred and label:
            tp += 1
        elif pred:
            fp += 1
        elif label:
            fn += 1
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def best_threshold(scores: list[float], labels: list[int]) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for threshold in sorted(set(scores)):
        f1 = f1_at_threshold(scores, labels, threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_t = threshold
    return best_t


def load_raw(day: str, max_requests: int) -> list[Example]:
    request = urllib.request.Request(URL.format(day=day))
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    examples: list[Example] = []
    with urllib.request.urlopen(request) as response:
        text = io.TextIOWrapper(response, encoding="utf-8", errors="replace")
        for line in text:
            if max_requests and len(examples) >= max_requests:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            hash_ids = record.get("hash_ids")
            input_length = record.get("input_length")
            if not isinstance(hash_ids, list) or not hash_ids or not input_length:
                continue
            messages = (record.get("request") or {}).get("messages") or []
            if not isinstance(messages, list):
                messages = []

            user_texts: list[str] = []
            system_text = ""
            roles: list[str] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "")
                content = message.get("content")
                roles.append(role)
                if isinstance(content, str):
                    if role == "user":
                        user_texts.append(content)
                    elif role == "system" and not system_text:
                        system_text = content

            last_user = user_texts[-1] if user_texts else ""
            last4_user = "\n".join(user_texts[-4:])
            timestamp = int(record.get("timestamp") or record.get("timestamp_ms") or len(examples))
            sph = str(record.get("system_prompt_hash") or "")
            examples.append(
                Example(
                    ts=timestamp,
                    sph=sph,
                    out=int(record.get("output_length") or 0),
                    nmsg=len(messages),
                    lulen=len(last_user),
                    lq=1 if "?" in last_user else 0,
                    hash_ids=[str(h) for h in hash_ids],
                    last_user=last_user,
                    last4_user=last4_user,
                    system_text=system_text,
                    all_roles=tuple(roles),
                )
            )
            if len(examples) % 1000 == 0:
                print(f"  read {len(examples):,} records", file=sys.stderr, flush=True)
    examples.sort(key=lambda ex: ex.ts)
    return examples


def make_metadata_features(examples: list[Example]) -> list[dict[int, float]]:
    feats: list[dict[int, float]] = []
    last_seen_ts: dict[str, int] = {}
    tenant_window: collections.deque[str] = collections.deque()
    tenant_count: collections.Counter[str] = collections.Counter()
    for ex in examples:
        tenant_b = bucket(tenant_count.get(ex.sph, 0), (1, 5, 20, 100))
        tenant_window.append(ex.sph)
        tenant_count[ex.sph] += 1
        if len(tenant_window) > 1000:
            old = tenant_window.popleft()
            tenant_count[old] -= 1
            if not tenant_count[old]:
                del tenant_count[old]

        think_ms = -1
        for hid in ex.hash_ids:
            if hid in last_seen_ts:
                think_ms = ex.ts - last_seen_ts[hid]
        for hid in ex.hash_ids:
            last_seen_ts[hid] = ex.ts

        think_b = 0 if think_ms < 0 else 1 + bucket(think_ms, (5_000, 30_000, 120_000, 600_000, 3_600_000))
        fields = (
            ("tenant", tenant_b),
            ("nmsg", bucket(ex.nmsg, (2, 4, 8, 16, 32))),
            ("lulen", bucket(ex.lulen, (50, 200, 1000))),
            ("lq", ex.lq),
            ("out", bucket(ex.out, (16, 64, 256, 1024))),
            ("nblk", bucket(len(ex.hash_ids), (4, 8, 16, 32, 64))),
            ("think", think_b),
        )
        feats.append({stable_hash(f"meta:{name}:{value}", 4096): 1.0 for name, value in fields})
    return feats


def build_labels(examples: list[Example], horizon: int) -> list[int]:
    touches: dict[str, list[int]] = collections.defaultdict(list)
    for idx, ex in enumerate(examples):
        for hid in ex.hash_ids:
            touches[hid].append(idx)
    labels: list[int] = []
    for idx, ex in enumerate(examples):
        deepest = ex.hash_ids[-1]
        seq = touches[deepest]
        k = bisect.bisect_right(seq, idx)
        labels.append(1 if k < len(seq) and seq[k] - idx <= horizon else 0)
    return labels


def train_logistic(
    features: list[dict[int, float]],
    labels: list[int],
    train_indices: list[int],
    *,
    epochs: int = 18,
    lr: float = 0.08,
    l2: float = 1e-5,
) -> tuple[dict[int, float], float]:
    weights: dict[int, float] = {}
    if not train_indices:
        return weights, 0.0
    base = sum(labels[i] for i in train_indices) / len(train_indices)
    base = min(0.99, max(0.01, base))
    bias = math.log(base / (1.0 - base))
    for _epoch in range(epochs):
        for i in train_indices:
            feat = features[i]
            z = bias + sum(weights.get(k, 0.0) * v for k, v in feat.items())
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            g = labels[i] - p
            bias += lr * g
            for k, v in feat.items():
                old = weights.get(k, 0.0)
                weights[k] = old + lr * (g * v - l2 * old)
        lr *= 0.9
    return weights, bias


def score(features: list[dict[int, float]], weights: dict[int, float], bias: float, indices: Iterable[int]) -> list[float]:
    out: list[float] = []
    for i in indices:
        z = bias + sum(weights.get(k, 0.0) * v for k, v in features[i].items())
        out.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))
    return out


def evaluate_model(
    name: str,
    features: list[dict[int, float]],
    labels: list[int],
    train_indices: list[int],
    test_indices: list[int],
) -> tuple[str, float, float, float]:
    weights, bias = train_logistic(features, labels, train_indices)
    train_scores = score(features, weights, bias, train_indices)
    test_scores = score(features, weights, bias, test_indices)
    train_labels = [labels[i] for i in train_indices]
    test_labels = [labels[i] for i in test_indices]
    threshold = best_threshold(train_scores, train_labels)
    test_auc = auc(list(zip(test_scores, test_labels)))
    test_f1 = f1_at_threshold(test_scores, test_labels, threshold)
    pred = [1 if s >= threshold else 0 for s in test_scores]
    acc = sum(int(p == y) for p, y in zip(pred, test_labels)) / len(test_labels)
    return name, test_auc, test_f1, acc


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt-text admission prediction on raw ART-Chat.")
    parser.add_argument("--day", default="20260405", choices=DAYS)
    parser.add_argument("--max-requests", type=int, default=15000)
    parser.add_argument("--horizon", default="ever", help="'ever' or request-count horizon such as 64")
    parser.add_argument("--real-only", action="store_true", help="Exclude one-block empty-tenant requests.")
    args = parser.parse_args()

    horizon = NEVER if args.horizon == "ever" else int(args.horizon)
    examples = load_raw(args.day, args.max_requests)
    labels = build_labels(examples, horizon)
    half = len(examples) // 2
    eligible = list(range(len(examples)))
    if args.real_only:
        eligible = [i for i, ex in enumerate(examples) if ex.sph and len(ex.hash_ids) > 1]
    train_indices = [i for i in eligible if i < half]
    test_indices = [i for i in eligible if i >= half]
    if not train_indices or not test_indices:
        raise SystemExit("not enough eligible train/test examples")

    meta = make_metadata_features(examples)
    last_user = [text_features("lu", ex.last_user, offset=10_000, bins=8192, max_words=1024) for ex in examples]
    last4 = [text_features("l4u", ex.last4_user, offset=20_000, bins=8192, max_words=2048) for ex in examples]
    system = [text_features("sys", ex.system_text, offset=30_000, bins=4096, max_words=512) for ex in examples]
    roles = [
        {40_000 + stable_hash("roles:" + ">".join(ex.all_roles[-12:]), 4096): 1.0}
        for ex in examples
    ]

    models = [
        ("metadata", meta),
        ("last_user_text", last_user),
        ("last4_user_text", last4),
        ("system_text", system),
        ("roles_shape", roles),
        ("system+last4_text", [merge_features(system[i], last4[i]) for i in range(len(examples))]),
        ("metadata+system+last4", [merge_features(meta[i], system[i], last4[i]) for i in range(len(examples))]),
        ("metadata+text+roles", [merge_features(meta[i], system[i], last4[i], roles[i]) for i in range(len(examples))]),
    ]

    base = sum(labels[i] for i in test_indices) / len(test_indices)
    majority = max(base, 1.0 - base)
    horizon_name = "ever" if horizon == NEVER else f"<={horizon}req"
    print(
        f"day={args.day} requests={len(examples)} eligible={len(eligible)} "
        f"train={len(train_indices)} test={len(test_indices)} "
        f"horizon={horizon_name} real_only={args.real_only}"
    )
    print(f"test_base={base:.3f} majority_acc={majority:.3f}")
    print(f"{'model':>24s}  {'auc':>6s}  {'f1':>6s}  {'acc':>6s}")
    for name, feats in models:
        row = evaluate_model(name, feats, labels, train_indices, test_indices)
        print(f"{row[0]:>24s}  {row[1]:6.3f}  {row[2]:6.3f}  {row[3]:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
