"""JSONL trace loading.

Each line is one request:

    {"block_size": 64, "hash_ids": [2001, 2002], "input_length": 128}

- hash_ids: cache block identities in request-prefix order (ints or strings).
- input_length: prefill input token count for the request.
- block_size: tokens per block; may be omitted if a fallback is given.

Adapted from kvcache-ai/kvcache-blog packages/kvcache-simulator (Apache-2.0).
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass
class TraceData:
    ids: list[int]              # interned block ids, all requests concatenated
    tokens: list[int]           # token weight of each block event
    request_starts: list[int]   # request i spans ids[request_starts[i]:request_starts[i+1]]
    block_size: int
    request_count: int
    unique_blocks: int


def _block_tokens(input_length: int, block_size: int, index: int, count: int) -> int:
    remaining = input_length - index * block_size
    if remaining <= 0:
        return 1
    return max(1, min(block_size, remaining))


def parse_trace_lines(lines: Iterable[str], *, block_size: int | None = None) -> TraceData:
    interner: dict[str, int] = {}
    ids: list[int] = []
    tokens: list[int] = []
    request_starts: list[int] = []
    trace_block_size = 0
    request_count = 0

    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            record: Any = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {line_number}: invalid JSON: {error}") from None
        hash_ids = record.get("hash_ids")
        if not isinstance(hash_ids, list) or not hash_ids:
            raise ValueError(f"line {line_number}: hash_ids must be a non-empty list")
        input_length = int(record.get("input_length", 0))
        if input_length <= 0:
            raise ValueError(f"line {line_number}: input_length must be a positive integer")
        record_block_size = int(record.get("block_size") or 0)
        if record_block_size > 0:
            if trace_block_size and record_block_size != trace_block_size:
                raise ValueError(f"line {line_number}: inconsistent block_size {record_block_size} != {trace_block_size}")
            trace_block_size = record_block_size
        selected = trace_block_size or int(block_size or 0)
        if selected <= 0:
            raise ValueError(f"line {line_number}: no block_size in record and no fallback provided")

        request_starts.append(len(ids))
        count = len(hash_ids)
        for index, value in enumerate(hash_ids):
            key = str(value)
            interned = interner.setdefault(key, len(interner))
            ids.append(interned)
            tokens.append(_block_tokens(input_length, selected, index, count))
        request_count += 1

    if not request_count:
        raise ValueError("no valid trace records found")
    request_starts.append(len(ids))
    return TraceData(
        ids=ids,
        tokens=tokens,
        request_starts=request_starts,
        block_size=trace_block_size or int(block_size or 0),
        request_count=request_count,
        unique_blocks=len(interner),
    )


def parse_trace_file(path: str | Path, *, block_size: int | None = None) -> TraceData:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return parse_trace_lines(handle, block_size=block_size)
