"""Real-prompt continuation prediction on WildChat.

WildChat contains human-readable ChatGPT conversations, so it is a useful
counterpoint to ART-Chat's anonymized word-soup prompts. This script streams
rows from the Hugging Face dataset-server API, keeps raw text only in memory,
and compares cheap metadata/length features with hashed lexical prompt
features.

Task:
  For each user turn, predict whether the same conversation contains a later
  user turn. That is the closest public-data analogue of "will this request's
  prompt prefix be reused by a continuation request?"  With --first-turn-only,
  each conversation contributes only its initial user prompt, making the label
  "does this conversation become multi-turn?"

Run:
    PYTHONPATH=src python3 autoresearch/wildchat_prompt_text_analysis.py --rows 5000
    PYTHONPATH=src python3 autoresearch/wildchat_prompt_text_analysis.py --rows 5000 --first-turn-only
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch.prompt_text_analysis import (
    auc,
    bucket,
    evaluate_model,
    merge_features,
    stable_hash,
    text_features,
)


DATASET = "allenai/WildChat"
CONFIG = "default"
SPLIT = "train"
PAGE_SIZE = 100
ROWS_URL = "https://datasets-server.huggingface.co/rows"


def fetch_page(offset: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": PAGE_SIZE,
        }
    )
    url = f"{ROWS_URL}?{query}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"failed to fetch offset {offset}")
    return [item["row"] for item in payload.get("rows", [])]


def load_rows(limit: int, workers: int) -> list[dict]:
    offsets = list(range(0, limit, PAGE_SIZE))
    rows_by_offset: dict[int, list[dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_page, offset): offset for offset in offsets}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            offset = futures[future]
            rows_by_offset[offset] = future.result()
            completed += len(rows_by_offset[offset])
            if completed % 1000 == 0 or completed >= limit:
                print(f"  fetched {min(completed, limit):,} rows", file=sys.stderr, flush=True)

    rows: list[dict] = []
    for offset in offsets:
        rows.extend(rows_by_offset.get(offset, []))
        if len(rows) >= limit:
            return rows[:limit]
    return rows


def role_name(role: str) -> str:
    role = (role or "").lower()
    if role in {"human", "prompter"}:
        return "user"
    return role


def metadata_features(row: dict, turn_index: int, user_so_far: int, last_user: str, prompt_so_far: str) -> dict[int, float]:
    fields = (
        ("model", str(row.get("model") or "")),
        ("language", str(row.get("language") or "")),
        ("turn_index", bucket(turn_index, (0, 1, 2, 4, 8))),
        ("user_so_far", bucket(user_so_far, (1, 2, 4, 8))),
        ("prompt_msg_count", bucket(turn_index + 1, (1, 2, 4, 8, 16))),
        ("last_user_len", bucket(len(last_user), (80, 250, 800, 2000))),
        ("prompt_len", bucket(len(prompt_so_far), (500, 2000, 8000, 20000))),
        ("last_user_question", int("?" in last_user)),
        ("last_user_codeish", int("```" in last_user or "def " in last_user or "function " in last_user)),
    )
    return {stable_hash(f"meta:{name}:{value}", 8192): 1.0 for name, value in fields}


def shape_features(roles: list[str], turn_index: int) -> dict[int, float]:
    tail = ">".join(roles[max(0, turn_index - 8): turn_index + 1])
    return {
        12_000 + stable_hash(f"shape:{tail}", 4096): 1.0,
        12_000 + stable_hash(f"shape:role:{roles[turn_index] if turn_index < len(roles) else ''}", 4096): 1.0,
    }


def build_examples(rows: list[dict], *, first_turn_only: bool, english_only: bool, non_redacted_only: bool) -> tuple[list[dict[int, float]], dict[str, list[dict[int, float]]], list[int], list[int], list[int]]:
    meta_feats: list[dict[int, float]] = []
    text_sets: dict[str, list[dict[int, float]]] = {
        "last_user_text": [],
        "user_history_text": [],
        "prompt_history_text": [],
        "shape": [],
    }
    labels: list[int] = []
    train_indices: list[int] = []
    test_indices: list[int] = []
    conversation_ids: list[int] = []
    split_row = len(rows) // 2

    for row_index, row in enumerate(rows):
        if english_only and str(row.get("language") or "").lower() not in {"english", "en"}:
            continue
        if non_redacted_only and row.get("redacted"):
            continue
        conversation = row.get("conversation") or []
        if not isinstance(conversation, list):
            continue

        roles: list[str] = []
        texts: list[str] = []
        user_positions: list[int] = []
        for i, message in enumerate(conversation):
            if not isinstance(message, dict):
                continue
            role = role_name(str(message.get("role") or ""))
            content = message.get("content")
            if role not in {"user", "assistant", "system", "tool"} or not isinstance(content, str):
                continue
            if non_redacted_only and message.get("redacted"):
                continue
            roles.append(role)
            texts.append(content)
            if role == "user":
                user_positions.append(len(roles) - 1)

        if not user_positions:
            continue
        selected = user_positions[:1] if first_turn_only else user_positions
        for user_number, pos in enumerate(user_positions):
            if pos not in selected:
                continue
            label = 1 if user_number + 1 < len(user_positions) else 0
            prompt_parts = texts[: pos + 1]
            prompt_so_far = "\n".join(prompt_parts)
            user_history = "\n".join(texts[p] for p in user_positions[: user_number + 1])
            last_user = texts[pos]
            ex_index = len(labels)
            meta_feats.append(metadata_features(row, pos, user_number + 1, last_user, prompt_so_far))
            text_sets["last_user_text"].append(text_features("wc_last_user", last_user, offset=20_000, bins=16_384, max_words=512))
            text_sets["user_history_text"].append(text_features("wc_user_hist", user_history, offset=40_000, bins=16_384, max_words=2048))
            text_sets["prompt_history_text"].append(text_features("wc_prompt", prompt_so_far, offset=60_000, bins=16_384, max_words=4096))
            text_sets["shape"].append(shape_features(roles, pos))
            labels.append(label)
            conversation_ids.append(row_index)
            if row_index < split_row:
                train_indices.append(ex_index)
            else:
                test_indices.append(ex_index)
    return meta_feats, text_sets, labels, train_indices, test_indices


def evaluate_constant(labels: list[int], test_indices: list[int]) -> tuple[float, float]:
    base = sum(labels[i] for i in test_indices) / len(test_indices)
    return base, max(base, 1.0 - base)


def main() -> int:
    parser = argparse.ArgumentParser(description="WildChat real-prompt continuation prediction.")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--first-turn-only", action="store_true")
    parser.add_argument("--all-languages", action="store_true")
    parser.add_argument("--include-redacted", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.rows, args.workers)
    meta, text_sets, labels, train, test = build_examples(
        rows,
        first_turn_only=args.first_turn_only,
        english_only=not args.all_languages,
        non_redacted_only=not args.include_redacted,
    )
    if not train or not test:
        raise SystemExit("not enough examples after filtering")

    feature_sets: list[tuple[str, list[dict[int, float]]]] = [
        ("metadata", meta),
        ("last_user_text", text_sets["last_user_text"]),
        ("user_history_text", text_sets["user_history_text"]),
        ("prompt_history_text", text_sets["prompt_history_text"]),
        ("shape", text_sets["shape"]),
        ("metadata+last_user", [merge_features(meta[i], text_sets["last_user_text"][i]) for i in range(len(labels))]),
        ("metadata+user_history", [merge_features(meta[i], text_sets["user_history_text"][i]) for i in range(len(labels))]),
        ("metadata+prompt_history", [merge_features(meta[i], text_sets["prompt_history_text"][i]) for i in range(len(labels))]),
        ("metadata+text+shape", [merge_features(meta[i], text_sets["prompt_history_text"][i], text_sets["shape"][i]) for i in range(len(labels))]),
    ]

    base, majority = evaluate_constant(labels, test)
    print(
        f"dataset={DATASET} rows={len(rows)} examples={len(labels)} "
        f"train={len(train)} test={len(test)} first_turn_only={args.first_turn_only} "
        f"english_only={not args.all_languages} non_redacted_only={not args.include_redacted}"
    )
    print(f"test_base={base:.3f} majority_acc={majority:.3f}")
    print(f"{'model':>25s}  {'auc':>6s}  {'f1':>6s}  {'acc':>6s}")
    for name, feats in feature_sets:
        row = evaluate_model(name, feats, labels, train, test)
        print(f"{row[0]:>25s}  {row[1]:6.3f}  {row[2]:6.3f}  {row[3]:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
