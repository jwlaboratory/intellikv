"""Eviction policies.

The cache is a prefix trie: a block only produces hits while its entire
prefix is cached, so eviction must always remove leaves (blocks with no
cached children). Hit accounting matches the KVCache.AI simulator: only the
longest continuous cached prefix of each request counts, and hit rate is
measured over the post-warmup window. A capacity that never fills before the
measurement window is reported as underfilled.

To add your own policy: copy simulate_custom, change the SCORING hooks, and
register it in POLICIES at the bottom.

Adapted from kvcache-ai/kvcache-blog packages/kvcache-simulator (Apache-2.0).
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq

from .plan import ExecutionPlan


@dataclass
class PolicyResult:
    policy: str
    capacity: int
    hit_tokens: int
    total_tokens: int
    hit_rate: float
    underfilled: bool = False


def _finish(policy: str, capacity: int, hit_tokens: int, total_tokens: int, underfilled: bool = False) -> PolicyResult:
    rate = (hit_tokens / total_tokens) if total_tokens else 0.0
    return PolicyResult(policy, capacity, hit_tokens, total_tokens, rate, underfilled)


def simulate_ceiling(plan: ExecutionPlan) -> PolicyResult:
    """Hit rate with infinite capacity: the best any policy can do."""
    seen = bytearray(len(plan.parent))
    hit_tokens = 0
    total_tokens = 0
    for request_index in range(plan.request_count):
        start = plan.request_starts[request_index]
        end = plan.request_starts[request_index + 1]
        measured = request_index >= plan.warmup_requests
        prefix_alive = True
        for index in range(start, end):
            node = plan.node_for_event[index]
            hit = seen[node] == 1
            if measured:
                total_tokens += plan.tokens[index]
                if prefix_alive and hit:
                    hit_tokens += plan.tokens[index]
            if not hit:
                prefix_alive = False
        for index in range(start, end):
            seen[plan.node_for_event[index]] = 1
    return _finish("ceiling", max(plan.unique_blocks, 1), hit_tokens, total_tokens)


def simulate_fifo(plan: ExecutionPlan, capacity: int) -> PolicyResult:
    if capacity <= 0 or not plan.ids:
        return _finish("fifo", capacity, 0, plan.total_measured_tokens)
    in_cache = bytearray(len(plan.parent))
    queue: list[int] = []
    head = 0
    cache_size = 0
    filled_in_warmup = False
    hit_tokens = 0
    total_tokens = 0

    for request_index in range(plan.request_count):
        if request_index >= plan.warmup_requests and not filled_in_warmup:
            return _finish("fifo", capacity, 0, plan.total_measured_tokens, underfilled=True)
        start = plan.request_starts[request_index]
        end = plan.request_starts[request_index + 1]
        measured = request_index >= plan.warmup_requests
        prefix_alive = True
        for index in range(start, end):
            node = plan.node_for_event[index]
            hit = in_cache[node] == 1
            if measured:
                total_tokens += plan.tokens[index]
                if prefix_alive and hit:
                    hit_tokens += plan.tokens[index]
            if not hit:
                prefix_alive = False
        for index in range(start, end):
            node = plan.node_for_event[index]
            if in_cache[node]:
                continue
            if cache_size >= capacity:
                while head < len(queue):
                    victim = queue[head]
                    head += 1
                    if in_cache[victim]:
                        in_cache[victim] = 0
                        cache_size -= 1
                        break
            if cache_size < capacity:
                in_cache[node] = 1
                cache_size += 1
                queue.append(node)
                if cache_size >= capacity and request_index < plan.warmup_requests:
                    filled_in_warmup = True

    if not filled_in_warmup or total_tokens <= 0:
        return _finish("fifo", capacity, 0, plan.total_measured_tokens, underfilled=True)
    return _finish("fifo", capacity, hit_tokens, total_tokens)


class _LeafHeap:
    """Heap of eviction candidates with lazy invalidation via versions."""

    def __init__(self, max_heap: bool) -> None:
        self.max_heap = max_heap
        self.items: list[tuple[int, int, int, int]] = []

    def push(self, node: int, key: int, version: int) -> None:
        if self.max_heap:
            item = (-key, -node, version, node)
        else:
            item = (key, -node, version, node)
        heapq.heappush(self.items, item)

    def pop(self) -> tuple[int, int, int] | None:
        if not self.items:
            return None
        key_part, _neg_node, version, node = heapq.heappop(self.items)
        key = -key_part if self.max_heap else key_part
        return node, key, version


def simulate_trie_policy(plan: ExecutionPlan, capacity: int, *, optimal: bool) -> PolicyResult:
    """LRU (evict least-recently-touched leaf) or Belady-style optimal
    (evict the leaf whose next reuse is farthest in the future)."""
    policy = "optimal" if optimal else "lru"
    if capacity <= 0 or not plan.ids:
        return _finish(policy, capacity, 0, plan.total_measured_tokens)

    node_count = len(plan.parent)
    present = bytearray(node_count)
    child_count = [0] * node_count
    state_version = [0] * node_count
    state_key = [0] * node_count
    protected_mark = [0] * node_count
    heap = _LeafHeap(max_heap=optimal)
    present[0] = 1
    cache_size = 0
    clock = 0
    mark_value = 1
    filled_in_warmup = False
    hit_tokens = 0
    total_tokens = 0

    def push_leaf(node: int) -> None:
        if node == 0 or not present[node] or child_count[node] != 0:
            return
        state_version[node] += 1
        heap.push(node, state_key[node], state_version[node])

    def touch(node: int, event_index: int) -> None:
        nonlocal clock
        if node == 0 or not present[node]:
            return
        if optimal:
            state_key[node] = plan.next_request_for_event[event_index]
        else:
            clock += 1
            state_key[node] = clock
        push_leaf(node)

    def evict_leaf(candidate_key: int) -> bool:
        nonlocal cache_size
        skipped: list[tuple[int, int, int]] = []
        while True:
            top = heap.pop()
            if top is None:
                for item in skipped:
                    heap.push(*item)
                return False
            node, key, version = top
            if not present[node] or child_count[node] != 0 or state_version[node] != version or state_key[node] != key:
                continue
            if protected_mark[node] == mark_value:
                skipped.append(top)
                continue
            if optimal and key <= candidate_key:
                heap.push(node, key, version)
                for item in skipped:
                    heap.push(*item)
                return False
            present[node] = 0
            cache_size -= 1
            parent = plan.parent[node]
            child_count[parent] -= 1
            if parent > 0 and present[parent] and child_count[parent] == 0:
                push_leaf(parent)
            for item in skipped:
                heap.push(*item)
            return True

    def mark_protected_path(start: int, end: int) -> None:
        nonlocal mark_value
        mark_value += 1
        if mark_value == 0x7FFFFFFF:
            protected_mark[:] = [0] * len(protected_mark)
            mark_value = 1
        protected_mark[0] = mark_value
        for index in range(start, end):
            node = plan.node_for_event[index]
            if present[node]:
                protected_mark[node] = mark_value

    for request_index in range(plan.request_count):
        if request_index >= plan.warmup_requests and not filled_in_warmup:
            return _finish(policy, capacity, 0, plan.total_measured_tokens, underfilled=True)
        start = plan.request_starts[request_index]
        end = plan.request_starts[request_index + 1]
        measured = request_index >= plan.warmup_requests
        prefix_alive = True
        for index in range(start, end):
            node = plan.node_for_event[index]
            hit = present[node] == 1
            if measured:
                total_tokens += plan.tokens[index]
                if prefix_alive and hit:
                    hit_tokens += plan.tokens[index]
            if prefix_alive and hit:
                touch(node, index)
            elif not hit:
                prefix_alive = False

        mark_protected_path(start, end)
        for index in range(start, end):
            node = plan.node_for_event[index]
            if present[node]:
                continue
            if cache_size >= capacity:
                candidate_key = plan.next_request_for_event[index] if optimal else 0
                if not evict_leaf(candidate_key):
                    break
            if cache_size < capacity and present[plan.parent[node]]:
                parent = plan.parent[node]
                present[node] = 1
                cache_size += 1
                child_count[parent] += 1
                touch(node, index)
                if cache_size >= capacity and request_index < plan.warmup_requests:
                    filled_in_warmup = True
            else:
                break

    if not filled_in_warmup or total_tokens <= 0:
        return _finish(policy, capacity, 0, plan.total_measured_tokens, underfilled=True)
    return _finish(policy, capacity, hit_tokens, total_tokens)


def simulate_custom(plan: ExecutionPlan, capacity: int) -> PolicyResult:
    """Your eviction policy goes here.

    All the trie bookkeeping (leaf-only eviction, prefix hit accounting,
    warmup/underfilled semantics) is handled; change only the SCORING hooks.
    The cached leaf with the SMALLEST score is evicted first.

    As shipped, the scoring implements LFU: evict the least-frequently-used
    leaf, breaking ties toward the least recently used one.

    Signals you can use inside the hooks:
      - plan.tokens[event_index]: token weight of the block
      - plan.next_request_for_event[event_index]: next request reusing it
      - plan.parent[node]: walk toward the root for depth-based scores
    """
    policy = "custom"
    if capacity <= 0 or not plan.ids:
        return _finish(policy, capacity, 0, plan.total_measured_tokens)

    node_count = len(plan.parent)
    present = bytearray(node_count)
    child_count = [0] * node_count
    state_version = [0] * node_count
    state_key = [0] * node_count
    protected_mark = [0] * node_count
    heap = _LeafHeap(max_heap=False)
    present[0] = 1
    cache_size = 0
    clock = 0
    mark_value = 1
    filled_in_warmup = False
    hit_tokens = 0
    total_tokens = 0

    # ---- SCORING (edit this section) -------------------------------------
    frequency = [0] * node_count

    def _pack(freq: int, tick: int) -> int:
        # Primary key: frequency. Tie-break: LRU order via the shared clock.
        return freq * (1 << 40) + tick

    def score_on_insert(node: int, event_index: int) -> int:
        frequency[node] = 1
        return _pack(frequency[node], clock)

    def score_on_hit(node: int, event_index: int) -> int:
        frequency[node] += 1
        return _pack(frequency[node], clock)

    def score_on_becomes_leaf(node: int) -> int:
        # Called when a node's last cached child was evicted, making it an
        # eviction candidate again.
        return _pack(frequency[node], clock)
    # ----------------------------------------------------------------------

    def push_leaf(node: int, key: int) -> None:
        if node == 0 or not present[node] or child_count[node] != 0:
            return
        state_key[node] = key
        state_version[node] += 1
        heap.push(node, key, state_version[node])

    def evict_leaf() -> bool:
        nonlocal cache_size
        skipped: list[tuple[int, int, int]] = []
        while True:
            top = heap.pop()
            if top is None:
                for item in skipped:
                    heap.push(*item)
                return False
            node, key, version = top
            if not present[node] or child_count[node] != 0 or state_version[node] != version or state_key[node] != key:
                continue
            if protected_mark[node] == mark_value:
                skipped.append(top)
                continue
            present[node] = 0
            cache_size -= 1
            parent = plan.parent[node]
            child_count[parent] -= 1
            if parent > 0 and present[parent] and child_count[parent] == 0:
                push_leaf(parent, score_on_becomes_leaf(parent))
            for item in skipped:
                heap.push(*item)
            return True

    def mark_protected_path(start: int, end: int) -> None:
        nonlocal mark_value
        mark_value += 1
        if mark_value == 0x7FFFFFFF:
            protected_mark[:] = [0] * len(protected_mark)
            mark_value = 1
        protected_mark[0] = mark_value
        for index in range(start, end):
            node = plan.node_for_event[index]
            if present[node]:
                protected_mark[node] = mark_value

    for request_index in range(plan.request_count):
        if request_index >= plan.warmup_requests and not filled_in_warmup:
            return _finish(policy, capacity, 0, plan.total_measured_tokens, underfilled=True)
        start = plan.request_starts[request_index]
        end = plan.request_starts[request_index + 1]
        measured = request_index >= plan.warmup_requests
        prefix_alive = True
        for index in range(start, end):
            node = plan.node_for_event[index]
            hit = present[node] == 1
            if measured:
                total_tokens += plan.tokens[index]
                if prefix_alive and hit:
                    hit_tokens += plan.tokens[index]
            if prefix_alive and hit:
                clock += 1
                push_leaf(node, score_on_hit(node, index))
            elif not hit:
                prefix_alive = False

        mark_protected_path(start, end)
        for index in range(start, end):
            node = plan.node_for_event[index]
            if present[node]:
                continue
            if cache_size >= capacity and not evict_leaf():
                break
            if cache_size < capacity and present[plan.parent[node]]:
                parent = plan.parent[node]
                present[node] = 1
                cache_size += 1
                child_count[parent] += 1
                clock += 1
                push_leaf(node, score_on_insert(node, index))
                if cache_size >= capacity and request_index < plan.warmup_requests:
                    filled_in_warmup = True
            else:
                break

    if not filled_in_warmup or total_tokens <= 0:
        return _finish(policy, capacity, 0, plan.total_measured_tokens, underfilled=True)
    return _finish(policy, capacity, hit_tokens, total_tokens)


POLICIES = {
    "fifo": simulate_fifo,
    "lru": lambda plan, capacity: simulate_trie_policy(plan, capacity, optimal=False),
    "optimal": lambda plan, capacity: simulate_trie_policy(plan, capacity, optimal=True),
    "custom": simulate_custom,
}


def simulate_policy(plan: ExecutionPlan, policy: str, capacity: int) -> PolicyResult:
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy} (known: {', '.join(POLICIES)})")
    return POLICIES[policy](plan, capacity)
