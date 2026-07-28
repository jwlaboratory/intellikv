"""Build simulator traces from ART-Chat-2.5M keeping per-request metadata.

Like art_chat_trace.py but each output record also carries a compact `meta`
object with causal, content-derived features (no raw text is stored):

    out   response length in tokens (output_length)
    sph   system_prompt_hash (tenant/application id)
    ts    timestamp within the day (ms)
    nmsg  number of messages in the request
    lulen character length of the last user message
    lq    1 if the last user message contains '?'

Usage:
    python3 scripts/art_chat_trace_meta.py --days 20260401 --max-requests 100000 -o traces/day1_100k.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import urllib.request

DAYS = ["20260401", "20260402", "20260403", "20260404", "20260405", "20260406", "20260407"]
URL = "https://huggingface.co/datasets/alessiotoniolo/ART-Chat-2.5M/resolve/main/jsonl/artchat_week_{day}.jsonl"
BLOCK_SIZE = 256


def extract_meta(record: dict) -> dict:
    meta = {
        "out": int(record.get("output_length") or 0),
        "sph": str(record.get("system_prompt_hash") or ""),
        "ts": int(record.get("timestamp") or 0),
        "nmsg": 0,
        "lulen": 0,
        "lq": 0,
    }
    request = record.get("request")
    if isinstance(request, dict):
        messages = request.get("messages")
        if isinstance(messages, list):
            meta["nmsg"] = len(messages)
            last_user = None
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    last_user = message
            if last_user is not None:
                content = last_user.get("content")
                if isinstance(content, str):
                    meta["lulen"] = len(content)
                    meta["lq"] = 1 if "?" in content else 0
    return meta


def stream_day(day: str, max_requests: int, seen: int, out) -> int:
    request = urllib.request.Request(URL.format(day=day))
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    written = 0
    with urllib.request.urlopen(request) as response:
        text = io.TextIOWrapper(response, encoding="utf-8", errors="replace")
        for line in text:
            if max_requests and seen + written >= max_requests:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            hash_ids = record.get("hash_ids")
            input_length = record.get("input_length")
            if not hash_ids or not input_length:
                continue
            compact = json.dumps(
                {
                    "block_size": BLOCK_SIZE,
                    "hash_ids": hash_ids,
                    "input_length": int(input_length),
                    "meta": extract_meta(record),
                },
                separators=(",", ":"),
            )
            out.write(compact + "\n")
            written += 1
            if written % 10_000 == 0:
                print(f"  {day}: {written:,} requests", file=sys.stderr, flush=True)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="ART-Chat-2.5M -> trace with metadata.")
    parser.add_argument("--days", default="20260401")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    days = DAYS if args.days.strip().lower() == "all" else [part.strip() for part in args.days.split(",") if part.strip()]
    for day in days:
        if day not in DAYS:
            parser.error(f"unknown day {day}")

    opener = gzip.open if args.output.endswith(".gz") else open
    total = 0
    with opener(args.output, "wt", encoding="utf-8") as out:
        for day in days:
            if args.max_requests and total >= args.max_requests:
                break
            print(f"streaming {day}...", file=sys.stderr, flush=True)
            total += stream_day(day, args.max_requests, total, out)
    print(f"wrote {args.output}: {total:,} requests", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
