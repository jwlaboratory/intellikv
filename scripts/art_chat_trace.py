"""Build simulator traces from the ART-Chat-2.5M dataset.

Streams the daily JSONL files from
https://huggingface.co/datasets/alessiotoniolo/ART-Chat-2.5M and strips each
record down to what the simulator needs (hash_ids, input_length, block_size),
so nothing close to the full 148 GB is ever downloaded or stored. Records are
sorted by timestamp_ms within each day; days are processed in date order.

Examples:

    # Quick sample: first 20k requests of day 1
    python3 scripts/art_chat_trace.py --days 20260401 --max-requests 20000 -o art_chat_20k.jsonl.gz

    # One full day
    python3 scripts/art_chat_trace.py --days 20260401 -o art_chat_day1.jsonl.gz

    # The whole week
    python3 scripts/art_chat_trace.py --days all -o art_chat_week.jsonl.gz

Then: kvcache-simulator run --trace art_chat_20k.jsonl.gz --model ... --backend python --policies lru,custom,optimal
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
BLOCK_SIZE = 256  # hash_ids are 256-token blocks per the dataset card


def stream_day(day: str, max_requests: int, seen: int) -> list[tuple[int, str]]:
    request = urllib.request.Request(URL.format(day=day))
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    records: list[tuple[int, str]] = []
    with urllib.request.urlopen(request) as response:
        text = io.TextIOWrapper(response, encoding="utf-8", errors="replace")
        for line in text:
            if max_requests and seen + len(records) >= max_requests:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            hash_ids = record.get("hash_ids")
            input_length = record.get("input_length")
            if not hash_ids or not input_length:
                continue
            timestamp = int(record.get("timestamp_ms") or 0)
            compact = json.dumps(
                {"block_size": BLOCK_SIZE, "hash_ids": hash_ids, "input_length": int(input_length)},
                separators=(",", ":"),
            )
            records.append((timestamp, compact))
            if len(records) % 50_000 == 0:
                print(f"  {day}: {len(records):,} requests read", file=sys.stderr)
    records.sort(key=lambda item: item[0])
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ART-Chat-2.5M into a kvcache-simulator trace.")
    parser.add_argument("--days", default="20260401", help=f"Comma-separated days from {DAYS[0]}..{DAYS[-1]}, or 'all'")
    parser.add_argument("--max-requests", type=int, default=0, help="Stop after this many requests (0 = no limit)")
    parser.add_argument("--output", "-o", default="art_chat.jsonl.gz", help="Output trace path (.jsonl or .jsonl.gz)")
    args = parser.parse_args()

    days = DAYS if args.days.strip().lower() == "all" else [part.strip() for part in args.days.split(",") if part.strip()]
    for day in days:
        if day not in DAYS:
            parser.error(f"unknown day {day}; valid: {', '.join(DAYS)} or 'all'")

    opener = gzip.open if args.output.endswith(".gz") else open
    total = 0
    with opener(args.output, "wt", encoding="utf-8") as out:
        for day in days:
            if args.max_requests and total >= args.max_requests:
                break
            print(f"streaming {day}...", file=sys.stderr)
            records = stream_day(day, args.max_requests, total)
            for _timestamp, compact in records:
                out.write(compact + "\n")
            total += len(records)
            print(f"  {day}: wrote {len(records):,} requests (total {total:,})", file=sys.stderr)
    print(f"wrote {args.output}: {total:,} requests, block size {BLOCK_SIZE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
