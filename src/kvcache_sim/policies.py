from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq

from .plan import ExecutionPlan


@dataclass
class PolicyResult:
    policy: str
    cacheBlocks: int
    warmupRequests: int
    measurementStartRequest: int | None
    measurementMode: str
    hitTokens: int
    totalTokens: int
    hitRate: float

    def to_json(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def _finish(policy: str, capacity: int, plan: ExecutionPlan, hit_tokens: int, total_tokens: int, mode: str = "fixed_window") -> PolicyResult:
    return PolicyResult(
        policy=policy,
        cacheBlocks=capacity,
        warmupRequests=plan.warmup_requests,
        measurementStartRequest=plan.warmup_requests if mode == "fixed_window" else None,
        measurementMode=mode,
        hitTokens=hit_tokens,
        totalTokens=total_tokens,
        hitRate=(hit_tokens / total_tokens) if total_tokens else 0.0,
    )


def _no_cache(policy: str, capacity: int, plan: ExecutionPlan) -> PolicyResult:
    return _finish(policy, capacity, plan, 0, plan.total_measured_tokens, "fixed_window")


def _underfilled(policy: str, capacity: int, plan: ExecutionPlan) -> PolicyResult:
    return _finish(policy, capacity, plan, 0, plan.total_measured_tokens, "underfilled_at_window")


def simulate_ceiling(plan: ExecutionPlan) -> PolicyResult:
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
    return _finish("ceiling", max(plan.unique_blocks, 1), plan, hit_tokens, total_tokens)


def simulate_fifo(plan: ExecutionPlan, capacity: int) -> PolicyResult:
    if capacity <= 0 or not plan.ids:
        return _no_cache("fifo", capacity, plan)
    in_cache = bytearray(len(plan.parent))
    queue: list[int] = []
    head = 0
    cache_size = 0
    full_before_measurement = False
    hit_tokens = 0
    total_tokens = 0

    for request_index in range(plan.request_count):
        if request_index >= plan.warmup_requests and not full_before_measurement:
            return _underfilled("fifo", capacity, plan)
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
                    full_before_measurement = True
            if head > 1_000_000 and head * 2 > len(queue):
                del queue[:head]
                head = 0

    if not full_before_measurement or total_tokens <= 0:
        return _underfilled("fifo", capacity, plan)
    return _finish("fifo", capacity, plan, hit_tokens, total_tokens)


class _LeafHeap:
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
    policy = "optimal" if optimal else "lru"
    if capacity <= 0 or not plan.ids:
        return _no_cache(policy, capacity, plan)

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
    full_before_measurement = False
    hit_tokens = 0
    total_tokens = 0

    def push_leaf(node: int) -> None:
        if node == 0 or not present[node] or child_count[node] != 0:
            return
        state_version[node] += 1
        heap.push(node, state_key[node], state_version[node])

    def touch_lru(node: int) -> None:
        nonlocal clock
        if node == 0 or not present[node]:
            return
        clock += 1
        state_key[node] = clock
        push_leaf(node)

    def update_optimal(node: int, next_use: int) -> None:
        if node == 0 or not present[node]:
            return
        state_key[node] = next_use
        push_leaf(node)

    def add_node(node: int, event_index: int) -> None:
        nonlocal cache_size
        parent = plan.parent[node]
        present[node] = 1
        cache_size += 1
        child_count[parent] += 1
        if optimal:
            update_optimal(node, plan.next_request_for_event[event_index])
        else:
            touch_lru(node)

    def restore(skipped: list[tuple[int, int, int]]) -> None:
        for node, key, version in skipped:
            heap.push(node, key, version)

    def evict_leaf(candidate_key: int) -> bool:
        nonlocal cache_size
        skipped: list[tuple[int, int, int]] = []
        while True:
            top = heap.pop()
            if top is None:
                restore(skipped)
                return False
            node, key, version = top
            if not present[node] or child_count[node] != 0 or state_version[node] != version or state_key[node] != key:
                continue
            if protected_mark[node] == mark_value:
                skipped.append(top)
                continue
            if optimal and key <= candidate_key:
                heap.push(node, key, version)
                restore(skipped)
                return False
            present[node] = 0
            cache_size -= 1
            parent = plan.parent[node]
            child_count[parent] -= 1
            if parent > 0 and present[parent] and child_count[parent] == 0:
                push_leaf(parent)
            restore(skipped)
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
        if request_index >= plan.warmup_requests and not full_before_measurement:
            return _underfilled(policy, capacity, plan)
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
                if optimal:
                    update_optimal(node, plan.next_request_for_event[index])
                else:
                    touch_lru(node)
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
                add_node(node, index)
                if cache_size >= capacity and request_index < plan.warmup_requests:
                    full_before_measurement = True
            else:
                break

    if not full_before_measurement or total_tokens <= 0:
        return _underfilled(policy, capacity, plan)
    return _finish(policy, capacity, plan, hit_tokens, total_tokens)


_IDLE_EDGES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _idle_bucket(idle: int) -> int:
    # Edges are 0,1,2,4,...,4096: the bucket index equals bit_length capped.
    if idle <= 0:
        return 0
    bl = idle.bit_length()
    return bl if bl < 13 else 13


def _depth_bucket(depth: int) -> int:
    if depth <= 2:
        return 0
    if depth <= 8:
        return 1
    if depth <= 32:
        return 2
    if depth <= 128:
        return 3
    return 4


def _freq_bucket(uses: int) -> int:
    if uses <= 1:
        return 0
    if uses == 2:
        return 1
    if uses <= 4:
        return 2
    if uses <= 8:
        return 3
    if uses <= 16:
        return 4
    if uses <= 64:
        return 5
    if uses <= 256:
        return 6
    return 7


def _gap_bucket(gap: int) -> int:
    if gap <= 0:
        return 0
    if gap <= 5:
        return 1
    if gap <= 20:
        return 2
    if gap <= 80:
        return 3
    return 4


def simulate_custom(plan: ExecutionPlan, capacity: int) -> PolicyResult:
    """IntelliKV: LRU reshaped by an online censored survival model.

    The policy observes every request's block path causally and independently
    of cache state, learning per-(feature-bucket, idle-age) histograms of
    "requests until next touch" with multi-stage censoring (never-returning
    blocks are counted at every idle age, so there is no survivor bias).
    Features combine trie structure (depth, frequency) with request content
    metadata when the trace provides it (think-time between turns, tenant
    popularity, turn count).

    Eviction samples K cached leaves and evicts the worst by
        score = (idle + 1) * ((1 - p_fb + eps) / (1 - p_glob + eps)) ** gamma
    where p is the (hierarchically shrunk) probability of returning within
    the current cache-residency window. gamma adapts to estimated residency
    (capacity / first-touch insert rate): decisive at small caches, near-LRU
    at huge ones. The clip on the multiplier bounds worst case near LRU.

    No future information is used: `plan.next_request_for_event` is only read
    by the env-gated oracle_* diagnostic modes, never by the default policy.
    See autoresearch/log.md for the full research trail.
    """
    policy = "custom"
    if capacity <= 0 or not plan.ids:
        return _no_cache(policy, capacity, plan)

    import collections as _collections
    import math as _math
    import random as _random

    import os as _os_seed

    rng = _random.Random(int(_os_seed.environ.get("INTELLIKV_SEED", "12345")))
    node_count = len(plan.parent)
    present = bytearray(node_count)
    child_count = [0] * node_count
    protected_mark = [0] * node_count
    present[0] = 1
    cache_size = 0
    mark_value = 1
    full_before_measurement = False
    hit_tokens = 0
    total_tokens = 0

    # --- causal per-node observation state (cache-independent) ---
    NEVER = -1
    depth = [0] * node_count
    last_seen = [NEVER] * node_count
    last_seen_ts = [0] * node_count
    use_count = [0] * node_count
    prev_gap = [0] * node_count
    feat_state = [0] * node_count  # feature bucket frozen at last touch
    coarse_state = [0] * node_count
    import os as _os

    metas = None if _os.environ.get("INTELLIKV_NO_META") else plan.request_meta

    # --- survival tables: (feature bucket x idle bucket) -> histogram of
    # remaining requests-to-next-touch, log-binned, capped at REM_CAP.
    # P(return <= delta | features, idle) is a cumulative sum over bins.
    # Censoring is multi-stage: for each idle edge e_j, the outcome
    # "returned within REM_CAP of reaching idle e_j, or not" becomes known at
    # age e_j + REM_CAP; a stage queue records the negative outcomes then, so
    # never-returning blocks are counted at every idle without survivor bias.
    REM_CAP = 128
    REM_EDGES = (8, 16, 32, 64, 128)
    N_BINS = len(REM_EDGES) + 1  # last bin: beyond cap / never
    n_idle = len(_IDLE_EDGES)
    # feature planes: shallow(depth<=8): tenant(5) x think(7) x freq(8) = 280;
    # deep (offset 280): think(7) x tenant(5) x freq(8) = 280 (default), nmsg
    # variant think(7) x nmsg(6) x freq(8) = 336, tenant_out x out(4) = 1120;
    # structural fallback (no meta): depth(5) x freq(8) x prev_gap(5) = 200.
    n_feat = 1400
    surv = [0] * (n_feat * n_idle * N_BINS)
    # coarse plane for hierarchical shrinkage: depth-class(2) x freq(8)
    n_coarse = 16
    surv_c = [0] * (n_coarse * n_idle * N_BINS)
    glob = [0] * (n_idle * N_BINS)

    # Optional trained prior: warm-start the survival tables from a previous
    # run's saved tables (INTELLIKV_SAVE_TABLES writes, INTELLIKV_PRIOR reads;
    # prior counts are divided by INTELLIKV_PRIOR_SHRINK to act as
    # pseudo-counts that online learning can overrule).
    import json as _json
    import os as _os_tab

    _prior_path = _os_tab.environ.get("INTELLIKV_PRIOR")
    if _prior_path:
        with open(_prior_path) as _fh:
            _prior = _json.load(_fh)
        _shrink = float(_os_tab.environ.get("INTELLIKV_PRIOR_SHRINK", "4"))
        if len(_prior["surv"]) == len(surv):
            surv = [int(v / _shrink) for v in _prior["surv"]]
            surv_c = [int(v / _shrink) for v in _prior["surv_c"]]
            glob = [int(v / _shrink) for v in _prior["glob"]]

    # per-request meta feature state (set once per request in the main loop)
    cur_ts = 0
    cur_nmsg_b = 0
    cur_tenant_b = 0
    cur_out_b = 0
    last_think = [-1] * node_count  # real-ms gap observed at last re-touch
    tenant_window: _collections.deque[str] = _collections.deque()
    tenant_count: dict[str, int] = {}

    def _think_bucket(gap_ms: int) -> int:
        if gap_ms < 0:
            return 0
        if gap_ms <= 5_000:
            return 1
        if gap_ms <= 30_000:
            return 2
        if gap_ms <= 120_000:
            return 3
        if gap_ms <= 600_000:
            return 4
        if gap_ms <= 3_600_000:
            return 5
        return 6

    # lookup table: remaining (1..REM_CAP) -> bin index
    _REM_BIN_TAB = [0] * (REM_CAP + 1)
    for _r in range(REM_CAP + 1):
        _b = N_BINS - 1
        for _k, _edge in enumerate(REM_EDGES):
            if _r <= _edge:
                _b = _k
                break
        _REM_BIN_TAB[_r] = _b

    def _rem_bin(remaining: int) -> int:
        return _REM_BIN_TAB[remaining] if remaining <= REM_CAP else N_BINS - 1

    def feature_bucket(node: int) -> int:
        if metas is not None:
            if last_seen[node] != NEVER:
                think_ms = cur_ts - last_seen_ts[node]
            else:
                # first touch: inherit the conversation's latest think time
                think_ms = last_think[plan.parent[node]]
            tb = _think_bucket(think_ms)
            fq = _freq_bucket(use_count[node])
            if depth[node] <= SHALLOW_DEPTH:
                return (cur_tenant_b * 7 + tb) * 8 + fq
            if DEEP_FEAT == "tenant":
                return 280 + (tb * 5 + cur_tenant_b) * 8 + fq
            if DEEP_FEAT == "tenant_out":
                return 280 + ((tb * 5 + cur_tenant_b) * 8 + fq) * 4 + cur_out_b
            return 280 + (tb * 6 + cur_nmsg_b) * 8 + fq
        return (_depth_bucket(depth[node]) * 8 + _freq_bucket(use_count[node])) * 5 + _gap_bucket(prev_gap[node])

    # stage s covers idle edges (STAGE_LO[s] .. STAGE_HI[s]); its outcome is
    # known REM_CAP requests after the largest covered edge.
    # For idle edges beyond REM_CAP the outcome window scales with the edge
    # ("returns before idle doubles"), so slow-but-alive blocks separate from
    # dead ones at every timescale; deadlines are 2*edge for those stages.
    STAGE_HI_EDGE = (32, 128, 512, 1024, 2048, 4096)
    STAGE_DEADLINE = tuple(
        (edge + REM_CAP if edge <= REM_CAP else 2 * edge) + 1 for edge in STAGE_HI_EDGE
    )
    STAGE_JS: list[tuple[int, ...]] = []
    _lo = 0
    for _hi_edge in STAGE_HI_EDGE:
        _js = tuple(j for j in range(n_idle) if _lo <= _IDLE_EDGES[j] <= _hi_edge)
        STAGE_JS.append(_js)
        _lo = _hi_edge + 1
    # one deque per stage keeps deadlines monotone within each queue
    stage_queues: list[_collections.deque[tuple[int, int, int]]] = [
        _collections.deque() for _ in STAGE_HI_EDGE
    ]

    def record_return(fb: int, cb: int, gap: int) -> None:
        # A re-touch after `gap` requests: for every idle edge the interval
        # survived past, record the outcome if it falls inside that edge's
        # window (REM_CAP for small edges, the edge itself beyond that; later
        # fates belong to the stage queue). Successes past REM_CAP land in the
        # last non-overflow bin — only the success/overflow split matters at
        # those idles.
        fb_base = fb * n_idle
        cb_base = cb * n_idle
        for j in range(n_idle):
            edge = _IDLE_EDGES[j]
            if edge >= gap:
                break
            remaining = gap - edge
            if remaining <= REM_CAP:
                b = _REM_BIN_TAB[remaining]
            elif remaining <= edge:  # scale-free window for edges > REM_CAP
                b = N_BINS - 2
            else:
                continue
            surv[(fb_base + j) * N_BINS + b] += 1
            surv_c[(cb_base + j) * N_BINS + b] += 1
            glob[j * N_BINS + b] += 1

    def drain_censor_queue(request_index: int) -> None:
        for stage, queue in enumerate(stage_queues):
            while queue and queue[0][0] < request_index:
                _deadline, start, node = queue.popleft()
                if last_seen[node] != start:
                    continue  # returned; positive outcomes recorded at re-touch
                fb = feat_state[node]
                cb = coarse_state[node]
                for j in STAGE_JS[stage]:
                    surv[(fb * n_idle + j) * N_BINS + N_BINS - 1] += 1
                    surv_c[(cb * n_idle + j) * N_BINS + N_BINS - 1] += 1
                    glob[j * N_BINS + N_BINS - 1] += 1
                if stage + 1 < len(STAGE_DEADLINE):
                    stage_queues[stage + 1].append((start + STAGE_DEADLINE[stage + 1], start, node))

    def observe_touch(node: int, request_index: int) -> None:
        seen = last_seen[node]
        if seen != NEVER:
            gap = request_index - seen
            if gap > 0:
                record_return(feat_state[node], coarse_state[node], gap)
                prev_gap[node] = gap
                if metas is not None:
                    last_think[node] = cur_ts - last_seen_ts[node]
        use_count[node] += 1
        # feature snapshot must see pre-update timestamps (think time = gap
        # between the previous touch and this one).
        feat_state[node] = feature_bucket(node)
        coarse_state[node] = (0 if depth[node] <= SHALLOW_DEPTH else 8) + _freq_bucket(use_count[node])
        last_seen[node] = request_index
        last_seen_ts[node] = cur_ts
        stage_queues[0].append((request_index + STAGE_DEADLINE[0], request_index, node))

    scoring_mode = _os.environ.get("INTELLIKV_CUSTOM_MODE", "hazard")
    _gamma_env = _os.environ.get("INTELLIKV_GAMMA", "auto")
    GAMMA_FIXED = None if _gamma_env == "auto" else int(_gamma_env)
    CLIP = float(_os.environ.get("INTELLIKV_CLIP", "24"))
    DEEP_FEAT = _os.environ.get("INTELLIKV_DEEP_FEAT", "tenant")
    SHALLOW_DEPTH = int(_os.environ.get("INTELLIKV_SHALLOW_DEPTH", "8"))
    # Admission control is off by default: T=6 helps cold workloads slightly
    # (+0.1..+0.4) but regresses warm ones (day2: up to -1.1 at 64GiB).
    ADMIT = float(_os.environ.get("INTELLIKV_ADMIT", "0"))
    GAMMA_STRUCT_HI = float(_os.environ.get("INTELLIKV_GAMMA_STRUCT_HI", "5"))
    DEADZONE = float(_os.environ.get("INTELLIKV_DEADZONE", "0"))
    SAMPLE = int(_os.environ.get("INTELLIKV_SAMPLE", "24"))
    # oracle_* modes are DIAGNOSTIC ONLY: they read the clairvoyant next-use
    # signal to measure how much of the LRU->Optimal gap specific kinds of
    # information could close. Never part of a deployable policy.
    oracle = scoring_mode.startswith("oracle")
    oracle_next = [1 << 60] * node_count if oracle else None
    never_request = plan.request_count + 1

    mult_memo: dict[int, float] = {}

    # residency estimate: capacity / insert rate (blocks per request), a
    # policy-independent proxy for how long an untouched block survives.
    # Insert rate is an EMA updated once per request in the main loop.
    _events_per_request = len(plan.ids) / max(1, plan.request_count)
    insert_rate_ema = [max(2.0, _events_per_request * 0.1)]
    residency_ema = [max(16.0, capacity / insert_rate_ema[0])]
    _k0 = 0
    while _k0 < len(REM_EDGES) - 1 and REM_EDGES[_k0] < residency_ema[0]:
        _k0 += 1
    residency_bin = [_k0]

    def update_residency(inserts_this_request: int) -> None:
        rate = insert_rate_ema[0] * 0.99 + inserts_this_request * 0.01
        insert_rate_ema[0] = rate
        est = capacity / rate if rate > 0.5 else capacity * 2.0
        residency_ema[0] = est
        k = 0
        while k < len(REM_EDGES) - 1 and REM_EDGES[k] < est:
            k += 1
        residency_bin[0] = k

    def expected_remaining(node: int, request_index: int) -> float:
        idle = request_index - last_seen[node]
        if idle < 0:
            idle = 0
        if oracle:
            next_use = oracle_next[node]
            dead = next_use >= never_request
            if scoring_mode == "oracle_belady":
                return float((next_use - request_index) if not dead else 1 << 40)
            if scoring_mode == "oracle_dead":
                return 1e15 if dead else float(idle)
            if scoring_mode == "oracle_near":
                if not dead and next_use - request_index <= 64:
                    return -1e9 + (next_use - request_index)
                return float(idle)
        if scoring_mode == "lru":
            return float(idle)
        # LRU backbone x bounded probability multiplier. delta tracks observed
        # cache residency (EMA of victim idle) so "returns soon" means
        # "returns before this cache would evict it".
        ib = _idle_bucket(idle)
        # The multiplier depends only on (feature, coarse, idle) buckets and
        # the tables, which are static during a request's eviction burst —
        # memoized per request (mult_memo is cleared in the main loop).
        key = (feat_state[node] * n_coarse + coarse_state[node]) * n_idle + ib
        mult = mult_memo.get(key)
        if mult is None:
            mult = _compute_mult(feat_state[node], coarse_state[node], ib)
            mult_memo[key] = mult
        return (idle + 1.0) * mult

    def _compute_mult(fb: int, cb: int, ib: int) -> float:
        split = residency_bin[0] + 1
        base_fb = (fb * n_idle + ib) * N_BINS
        base_c = (cb * n_idle + ib) * N_BINS
        base_g = ib * N_BINS
        fb_cum = sum(surv[base_fb:base_fb + split])
        fb_tot = fb_cum + sum(surv[base_fb + split:base_fb + N_BINS])
        c_cum = sum(surv_c[base_c:base_c + split])
        c_tot = c_cum + sum(surv_c[base_c + split:base_c + N_BINS])
        g_cum = sum(glob[base_g:base_g + split])
        g_tot = g_cum + sum(glob[base_g + split:base_g + N_BINS])
        if g_tot < 10:
            return 1.0
        # hierarchical shrinkage: feature bucket -> coarse depth x freq -> global
        p_g = g_cum / g_tot
        p_c = (c_cum + 20.0 * p_g) / (c_tot + 20.0)
        p_fb = (fb_cum + 10.0 * p_c) / (fb_tot + 10.0)
        mult = ((1.0 - p_fb) + 0.02) / ((1.0 - p_g) + 0.02)
        if GAMMA_FIXED is not None:
            gamma = GAMMA_FIXED
        else:
            # faster cache turnover -> nearer-term predictions -> amplify more
            # (smooth in log-residency, clamped to [2, 5])
            ema = residency_ema[0]
            gamma = 9.2 - 0.6 * _math.log2(ema if ema > 4 else 4.0)
            # structural-only features are weaker: amplify less mid-curve
            gamma_hi = 5.0 if metas is not None else GAMMA_STRUCT_HI
            if gamma > gamma_hi:
                gamma = gamma_hi
            elif gamma < 2.0:
                gamma = 2.0
        if gamma > 1:
            mult = mult ** gamma
        if DEADZONE > 0 and (1.0 / DEADZONE) < mult < DEADZONE:
            return 1.0  # not confident enough to depart from LRU
        if mult < 1.0 / CLIP:
            mult = 1.0 / CLIP
        elif mult > CLIP:
            mult = CLIP
        return mult

    # --- cached-leaf set with O(1) add/remove/sample ---
    leaf_list: list[int] = []
    leaf_pos: dict[int, int] = {}

    def leaf_add(node: int) -> None:
        if node != 0 and node not in leaf_pos:
            leaf_pos[node] = len(leaf_list)
            leaf_list.append(node)

    def leaf_remove(node: int) -> None:
        pos = leaf_pos.pop(node, None)
        if pos is None:
            return
        tail = leaf_list.pop()
        if tail != node:
            leaf_list[pos] = tail
            leaf_pos[tail] = pos

    def pick_victim(request_index: int) -> int:
        count = len(leaf_list)
        if count == 0:
            return -1
        best = -1
        best_score = -1.0
        best_idle = -1
        tries = SAMPLE if count > SAMPLE else count
        if count <= SAMPLE * 2:
            candidates = leaf_list
        else:
            candidates = [leaf_list[rng.randrange(count)] for _ in range(tries)]
        for node in candidates:
            if protected_mark[node] == mark_value:
                continue
            score = expected_remaining(node, request_index)
            idle = request_index - last_seen[node]
            if score > best_score or (score == best_score and idle > best_idle):
                best = node
                best_score = score
                best_idle = idle
        if best < 0 and count > SAMPLE * 2:
            for node in leaf_list:
                if protected_mark[node] == mark_value:
                    continue
                score = expected_remaining(node, request_index)
                idle = request_index - last_seen[node]
                if score > best_score or (score == best_score and idle > best_idle):
                    best = node
                    best_score = score
                    best_idle = idle
        return best

    def evict_leaf(request_index: int) -> bool:
        nonlocal cache_size
        victim = pick_victim(request_index)
        if victim < 0:
            return False
        present[victim] = 0
        cache_size -= 1
        leaf_remove(victim)
        parent = plan.parent[victim]
        child_count[parent] -= 1
        if parent > 0 and present[parent] and child_count[parent] == 0:
            leaf_add(parent)
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
        if request_index >= plan.warmup_requests and not full_before_measurement:
            return _underfilled(policy, capacity, plan)
        drain_censor_queue(request_index)
        if metas is not None:
            meta = metas[request_index] or {}
            cur_ts = int(meta.get("ts") or 0)
            nmsg = int(meta.get("nmsg") or 0)
            cur_nmsg_b = (
                0 if nmsg <= 2 else 1 if nmsg <= 4 else 2 if nmsg <= 8 else 3 if nmsg <= 16 else 4 if nmsg <= 32 else 5
            )
            out_tokens = int(meta.get("out") or 0)
            cur_out_b = 0 if out_tokens <= 32 else 1 if out_tokens <= 128 else 2 if out_tokens <= 512 else 3
            sph = str(meta.get("sph") or "")
            tcount = tenant_count.get(sph, 0)
            cur_tenant_b = 0 if tcount <= 1 else 1 if tcount <= 5 else 2 if tcount <= 20 else 3 if tcount <= 100 else 4
            tenant_window.append(sph)
            tenant_count[sph] = tcount + 1
            if len(tenant_window) > 1000:
                old = tenant_window.popleft()
                remaining_count = tenant_count.get(old, 1) - 1
                if remaining_count:
                    tenant_count[old] = remaining_count
                else:
                    tenant_count.pop(old, None)
        start = plan.request_starts[request_index]
        end = plan.request_starts[request_index + 1]
        measured = request_index >= plan.warmup_requests
        prefix_alive = True
        for index in range(start, end):
            node = plan.node_for_event[index]
            if depth[node] == 0 and node != 0:
                depth[node] = depth[plan.parent[node]] + 1
            hit = present[node] == 1
            if measured:
                total_tokens += plan.tokens[index]
                if prefix_alive and hit:
                    hit_tokens += plan.tokens[index]
            if not hit:
                prefix_alive = False
            observe_touch(node, request_index)
            if oracle:
                oracle_next[node] = plan.next_request_for_event[index]

        mark_protected_path(start, end)
        mult_memo.clear()
        inserts_this_request = 0
        for index in range(start, end):
            node = plan.node_for_event[index]
            if present[node]:
                continue
            if ADMIT > 0 and cache_size >= capacity:
                # admission control: refuse blocks already predicted near-dead
                # (their descendants cannot insert either -> stop the request)
                key = (feat_state[node] * n_coarse + coarse_state[node]) * n_idle
                mult = mult_memo.get(key)
                if mult is None:
                    mult = _compute_mult(feat_state[node], coarse_state[node], 0)
                    mult_memo[key] = mult
                if mult > ADMIT:
                    break
            if cache_size >= capacity and not evict_leaf(request_index):
                break
            if cache_size < capacity and present[plan.parent[node]]:
                parent = plan.parent[node]
                present[node] = 1
                cache_size += 1
                if use_count[node] <= 1:
                    inserts_this_request += 1
                child_count[parent] += 1
                leaf_remove(parent)
                leaf_add(node)
                if cache_size >= capacity and request_index < plan.warmup_requests:
                    full_before_measurement = True
            else:
                break
        update_residency(inserts_this_request)

    _save_path = _os_tab.environ.get("INTELLIKV_SAVE_TABLES")
    if _save_path:
        with open(_save_path, "w") as _fh:
            _json.dump({"surv": surv, "surv_c": surv_c, "glob": glob}, _fh)

    if not full_before_measurement or total_tokens <= 0:
        return _underfilled(policy, capacity, plan)
    return _finish(policy, capacity, plan, hit_tokens, total_tokens)


def simulate_policy(plan: ExecutionPlan, policy: str, capacity: int) -> PolicyResult:
    if policy == "fifo":
        return simulate_fifo(plan, capacity)
    if policy == "lru":
        return simulate_trie_policy(plan, capacity, optimal=False)
    if policy == "optimal":
        return simulate_trie_policy(plan, capacity, optimal=True)
    if policy == "custom":
        return simulate_custom(plan, capacity)
    raise ValueError(f"Unsupported policy: {policy}")
