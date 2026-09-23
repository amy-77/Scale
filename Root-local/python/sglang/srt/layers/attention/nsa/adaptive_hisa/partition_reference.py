"""CPU reference solvers for the P-key partition (numba / numpy).

This module is the *reference* backend (``split_backend=cpu_reference``). It
holds the exact algorithms the GPU backend (``partition_gpu``) must reproduce:

* ``_dp_solve``: one λ for the whole forest (bisection to ``count <= M``),
  then heap repair (gain desc, start asc, level asc, idx asc) to exactly ``M``
  leaves. Parity-tested against the sibling ``lambda_dp_repair``.
* ``heap_reference`` merge (``_merge_loop``): the old sequential min-heap
  greedy that keeps merging the cheapest adjacent pair until no pair is below
  the threshold. This is **not** the two-round algorithm.
* ``sync_nonoverlap`` merge (``sync_nonoverlap_merge``): a port of the
  sibling ``batched_adjacent_merge_rounds`` (E10). Every round freezes the
  leaves, computes all adjacent Ward costs, visits eligible pairs by
  ``(cost, start)`` and selects those whose two leaves are still free, merges
  all selected pairs at once, then stops after ``rounds`` rounds.

Nothing here touches the GPU; energies and scores arrive as host arrays.
"""

from __future__ import annotations

import heapq

import numpy as np
import torch

try:
    from numba import njit as _njit
except Exception:  # pragma: no cover - numba is an sglang dependency

    def _njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn


# --------------------------------------------------------------------------- #
# λ-DP with exact-budget repair
# --------------------------------------------------------------------------- #


@_njit(cache=True, nogil=True)
def _dp_pass(flat, offs, lam, cost, do_split, leaves):
    """Bottom-up penalised DP for one λ. Ties keep the parent. Returns leaf count."""
    levels = offs.shape[0] - 1
    for i in range(offs[0], offs[1]):
        cost[i] = flat[i] + lam
        do_split[i] = False
        leaves[i] = 1
    for lv in range(1, levels):
        base = offs[lv]
        prev = offs[lv - 1]
        for i in range(offs[lv + 1] - base):
            split = cost[prev + 2 * i] + cost[prev + 2 * i + 1]
            keep = flat[base + i] + lam
            if split < keep:
                cost[base + i] = split
                do_split[base + i] = True
                leaves[base + i] = leaves[prev + 2 * i] + leaves[prev + 2 * i + 1]
            else:
                cost[base + i] = keep
                do_split[base + i] = False
                leaves[base + i] = 1
    total = 0
    for i in range(offs[levels - 1], offs[levels]):
        total += leaves[i]
    return total


@_njit(cache=True, nogil=True)
def _dp_count(flat, offs, lam, cost, leaves):
    """Leaf count of the penalised DP for one λ; level 0 is folded into level 1."""
    levels = offs.shape[0] - 1
    for lv in range(1, levels):
        base = offs[lv]
        prev = offs[lv - 1]
        for i in range(offs[lv + 1] - base):
            if lv == 1:
                split = flat[prev + 2 * i] + flat[prev + 2 * i + 1] + 2.0 * lam
                kids = 2
            else:
                split = cost[prev + 2 * i] + cost[prev + 2 * i + 1]
                kids = leaves[prev + 2 * i] + leaves[prev + 2 * i + 1]
            keep = flat[base + i] + lam
            if split < keep:
                cost[base + i] = split
                leaves[base + i] = kids
            else:
                cost[base + i] = keep
                leaves[base + i] = 1
    total = 0
    for i in range(offs[levels - 1], offs[levels]):
        total += leaves[i]
    return total


@_njit(cache=True, nogil=True)
def _emit_leaves(offs, is_leaf, atom, out_start, out_len):
    """Pre-order walk of every root; leaves come out sorted by logical start."""
    levels = offs.shape[0] - 1
    n_roots = offs[levels] - offs[levels - 1]
    stack_lv = np.empty(2 * levels + 2, np.int64)
    stack_ix = np.empty(2 * levels + 2, np.int64)
    n = 0
    for r in range(n_roots):
        sp = 0
        stack_lv[0] = levels - 1
        stack_ix[0] = r
        sp = 1
        while sp > 0:
            sp -= 1
            lv = stack_lv[sp]
            ix = stack_ix[sp]
            if is_leaf[offs[lv] + ix]:
                ln = atom << lv
                out_start[n] = ix * ln
                out_len[n] = ln
                n += 1
            else:
                # push right first so the left child is visited first
                stack_lv[sp] = lv - 1
                stack_ix[sp] = 2 * ix + 1
                sp += 1
                stack_lv[sp] = lv - 1
                stack_ix[sp] = 2 * ix
                sp += 1
    return n


@_njit(cache=True, nogil=True)
def _dp_solve(flat, offs, m_leaves, atom, max_iter):
    """One λ for the whole forest, then split the largest gain until exactly ``M``.

    λ* is the smallest λ (to 1e-12 relative) whose DP has at most ``M`` leaves.
    Repair replays the offline heap order (gain desc, start asc, level asc,
    idx asc). Returns ``(start, length, lambda, dp_leaves, repairs)``.
    """
    levels = offs.shape[0] - 1
    n_nodes = offs[levels]
    n_roots = offs[levels] - offs[levels - 1]
    cost = np.empty(n_nodes, np.float64)
    do_split = np.zeros(n_nodes, np.bool_)
    leaves = np.empty(n_nodes, np.int64)

    hi = 0.0
    for i in range(offs[levels - 1], offs[levels]):
        if flat[i] > hi:
            hi = flat[i]
    hi = hi * 2.0 + 1.0
    lo = 0.0
    if _dp_count(flat, offs, lo, cost, leaves) <= m_leaves:
        best = lo
    else:
        best = hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if _dp_count(flat, offs, mid, cost, leaves) <= m_leaves:
                best = mid
                hi = mid
            else:
                lo = mid
            if hi - lo <= 1e-12 * max(1.0, hi):
                break
    dp_leaves = _dp_pass(flat, offs, best, cost, do_split, leaves)

    # Top-down: a node is a leaf when every ancestor split and it did not.
    is_leaf = np.zeros(n_nodes, np.bool_)
    active = np.zeros(n_nodes, np.bool_)
    for r in range(n_roots):
        active[offs[levels - 1] + r] = True
    for lv in range(levels - 1, 0, -1):
        base = offs[lv]
        down = offs[lv - 1]
        for i in range(offs[lv + 1] - base):
            node = base + i
            if active[node]:
                if do_split[node]:
                    active[down + 2 * i] = True
                    active[down + 2 * i + 1] = True
                else:
                    is_leaf[node] = True
    for i in range(offs[0], offs[1]):
        if active[i]:
            is_leaf[i] = True

    repairs = 0
    need = m_leaves - dp_leaves
    if need > 0:
        heap = [(0.0, np.int64(0), np.int64(0), np.int64(0))]
        heap.pop()
        for lv in range(1, levels):
            base = offs[lv]
            down = offs[lv - 1]
            for i in range(offs[lv + 1] - base):
                if is_leaf[base + i]:
                    gain = flat[base + i] - flat[down + 2 * i] - flat[down + 2 * i + 1]
                    heapq.heappush(heap, (-gain, np.int64(i * (atom << lv)), np.int64(lv), np.int64(i)))
        while repairs < need and len(heap) > 0:
            _, _, lv, i = heapq.heappop(heap)
            node = offs[lv] + i
            if not is_leaf[node]:
                continue
            is_leaf[node] = False
            down = offs[lv - 1]
            for kid in (2 * i, 2 * i + 1):
                is_leaf[down + kid] = True
                if lv - 1 > 0:
                    gain = flat[down + kid] - flat[offs[lv - 2] + 2 * kid] - flat[offs[lv - 2] + 2 * kid + 1]
                    heapq.heappush(heap, (-gain, np.int64(kid * (atom << (lv - 1))), np.int64(lv - 1), np.int64(kid)))
            repairs += 1

    out_start = np.empty(m_leaves, np.int64)
    out_len = np.empty(m_leaves, np.int64)
    n = _emit_leaves(offs, is_leaf, atom, out_start, out_len)
    return out_start[:n], out_len[:n], best, dp_leaves, repairs


def flatten_energies(energies: list[torch.Tensor]) -> tuple[torch.Tensor, np.ndarray]:
    """Level-major flat node array and ``offs[level]`` (``offs[-1]`` = node count)."""
    offs = np.zeros(len(energies) + 1, dtype=np.int64)
    offs[1:] = np.cumsum([int(e.numel()) for e in energies])
    return torch.cat([e.to(torch.float64) for e in energies]), offs


def lambda_dp_repair(
    energies: list[torch.Tensor],
    m_leaves: int,
    *,
    atom: int = 1,
    max_iter: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Host wrapper around :func:`_dp_solve`. Returns CPU ``(start, length, meta)``."""
    if not energies:
        if m_leaves:
            raise ValueError("empty tree cannot satisfy a positive leaf budget")
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, {"M": 0, "lambda": 0.0, "dp_leaves": 0, "repairs": 0}
    n_roots, n_atoms = int(energies[-1].numel()), int(energies[0].numel())
    if not n_roots <= m_leaves <= n_atoms:
        raise ValueError(f"M={m_leaves} infeasible: roots={n_roots}, atoms={n_atoms}")
    flat, offs = flatten_energies(energies)
    start, length, lam, dp_leaves, repairs = _dp_solve(
        flat.cpu().numpy(), offs, int(m_leaves), int(atom), int(max_iter)
    )
    if start.shape[0] != m_leaves:
        raise RuntimeError(f"repair produced {start.shape[0]} leaves for M={m_leaves}")
    meta = {"M": int(m_leaves), "lambda": float(lam), "dp_leaves": int(dp_leaves), "repairs": int(repairs)}
    return torch.from_numpy(start), torch.from_numpy(length), meta


# --------------------------------------------------------------------------- #
# Leaf score sums and Ward costs
# --------------------------------------------------------------------------- #


@_njit(cache=True, nogil=True)
def _leaf_totals(scores, starts, lengths):
    """Per-leaf, per-query float64 score sums from ``scores[C, N]``."""
    m = starts.shape[0]
    c_n = scores.shape[0]
    tot = np.empty((m, c_n), np.float64)
    for i in range(m):
        s0 = starts[i]
        s1 = s0 + lengths[i]
        for c in range(c_n):
            acc = 0.0
            for t in range(s0, s1):
                acc += scores[c, t]
            tot[i, c] = acc
    return tot


@_njit(cache=True, nogil=True)
def _pair_cost(tot, ln, left, right):
    nl, nr = ln[left], ln[right]
    acc = 0.0
    for c in range(tot.shape[1]):
        d = tot[left, c] / nl - tot[right, c] / nr
        acc += d * d
    return nl * nr / (nl + nr) * acc


# --------------------------------------------------------------------------- #
# heap_reference: sequential greedy until stable (old algorithm)
# --------------------------------------------------------------------------- #


@_njit(cache=True, nogil=True)
def _merge_loop(starts, lengths, totals, lam, max_len):
    """Exact sequential greedy merge; identical order to the offline heap version.

    ``max_len <= 0`` means no length cap.
    """
    m = starts.shape[0]
    cap = 2 * m
    st = np.empty(cap, np.int64)
    ln = np.empty(cap, np.int64)
    tot = np.empty((cap, totals.shape[1]), np.float64)
    prev = np.empty(cap, np.int64)
    nxt = np.empty(cap, np.int64)
    alive = np.zeros(cap, np.bool_)
    for i in range(m):
        st[i] = starts[i]
        ln[i] = lengths[i]
        for c in range(totals.shape[1]):
            tot[i, c] = totals[i, c]
        prev[i] = i - 1
        nxt[i] = i + 1 if i + 1 < m else -1
        alive[i] = True
    no_cap = max_len <= 0

    heap = [(0.0, np.int64(0), np.int64(0), np.int64(0))]
    heap.pop()
    for i in range(m - 1):
        if no_cap or ln[i] + ln[i + 1] <= max_len:
            cost = _pair_cost(tot, ln, i, i + 1)
            if cost < lam:
                heapq.heappush(heap, (max(0.0, cost), st[i], np.int64(i), np.int64(i + 1)))

    head = 0
    n_nodes = m
    merges = 0
    while len(heap):
        _, _, left, right = heapq.heappop(heap)
        if not alive[left] or not alive[right] or nxt[left] != right:
            continue
        before, after = prev[left], nxt[right]
        k = n_nodes
        n_nodes += 1
        st[k] = st[left]
        ln[k] = ln[left] + ln[right]
        for c in range(tot.shape[1]):
            tot[k, c] = tot[left, c] + tot[right, c]
        prev[k] = before
        nxt[k] = after
        alive[k] = True
        alive[left] = False
        alive[right] = False
        if before < 0:
            head = k
        else:
            nxt[before] = k
        if after >= 0:
            prev[after] = k
        merges += 1
        if before >= 0 and (no_cap or ln[before] + ln[k] <= max_len):
            cost = _pair_cost(tot, ln, before, k)
            if cost < lam:
                heapq.heappush(heap, (max(0.0, cost), st[before], before, np.int64(k)))
        if after >= 0 and (no_cap or ln[k] + ln[after] <= max_len):
            cost = _pair_cost(tot, ln, k, after)
            if cost < lam:
                heapq.heappush(heap, (max(0.0, cost), st[k], np.int64(k), after))

    out_n = m - merges
    out_start = np.empty(out_n, np.int64)
    out_len = np.empty(out_n, np.int64)
    node = head
    pos = 0
    while node >= 0:
        out_start[pos] = st[node]
        out_len[pos] = ln[node]
        node = nxt[node]
        pos += 1
    return out_start, out_len, merges


def heap_reference_merge(
    starts: np.ndarray,
    lengths: np.ndarray,
    totals: np.ndarray,
    threshold: float,
    *,
    max_merge_len: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    out_start, out_len, merges = _merge_loop(
        starts.astype(np.int64), lengths.astype(np.int64), totals, float(threshold), int(max_merge_len)
    )
    return out_start, out_len, {
        "merge_policy": "heap_reference",
        "merge_count": int(merges),
        "merge_rounds_run": 1,
        "round_merge_counts": [int(merges)],
        "max_merge_len": int(max_merge_len),
    }


# --------------------------------------------------------------------------- #
# sync_nonoverlap: frozen rounds, (cost,start) greedy matching, bulk merge
# --------------------------------------------------------------------------- #


# Keep in sync with ``partition_gpu.TARGET_ADMIT_FACTOR``.
TARGET_ADMIT_FACTOR = 2


def _adjacent_costs(lengths: np.ndarray, totals: np.ndarray) -> np.ndarray:
    n_l = lengths[:-1].astype(np.float64)
    n_r = lengths[1:].astype(np.float64)
    delta = totals[:-1] / n_l[:, None] - totals[1:] / n_r[:, None]
    return n_l * n_r / (n_l + n_r) * np.einsum("ic,ic->i", delta, delta)


def sync_nonoverlap_matching(
    costs: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
    *,
    max_merge_len: int = 0,
) -> np.ndarray:
    """Return the selected left indices of one synchronous round.

    Eligible edges (``cost < threshold`` and, if capped, combined length
    ``<= max_merge_len``) are visited by ascending ``(cost, start)``; an edge is
    selected when neither of its leaves has been taken in this round. This is
    exactly the sibling reference's ``sorted(eligible)`` + ``occupied`` scan.
    """
    m = starts.shape[0]
    if m < 2:
        return np.zeros(0, dtype=np.int64)
    eligible = costs < threshold
    if max_merge_len > 0:
        eligible &= lengths[:-1] + lengths[1:] <= max_merge_len
    idx = np.nonzero(eligible)[0]
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = idx[np.lexsort((starts[idx], np.maximum(costs[idx], 0.0)))]
    occupied = np.zeros(m, dtype=np.bool_)
    selected = []
    for left in order.tolist():
        if not occupied[left] and not occupied[left + 1]:
            selected.append(left)
            occupied[left] = True
            occupied[left + 1] = True
    return np.array(sorted(selected), dtype=np.int64)


def sync_nonoverlap_merge(
    starts: np.ndarray,
    lengths: np.ndarray,
    totals: np.ndarray,
    threshold: float,
    *,
    rounds: int = 2,
    max_merge_len: int = 0,
    target: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Port of the sibling ``batched_adjacent_merge_rounds`` with ``max_rounds``.

    ``totals`` is ``[M, C]`` float64 per-leaf score sums. Stops after
    ``rounds`` rounds or when a round selects nothing.

    ``target > 0`` switches to the target-count merge (``threshold`` is
    ignored): each round admits the ``TARGET_ADMIT_FACTOR * (count - target)``
    cheapest eligible edges, matches them, and merges the ``count - target``
    cheapest matched pairs (boundary ties included), mirroring
    ``partition_gpu.merge_sync_nonoverlap``.
    """
    starts = starts.astype(np.int64)
    lengths = lengths.astype(np.int64)
    totals = totals.astype(np.float64)
    per_round: list[int] = []
    for _ in range(int(rounds)):
        if starts.shape[0] < 2 or (threshold <= 0 and not target):
            break
        costs = _adjacent_costs(lengths, totals)
        thr = float(threshold)
        if target:
            deficit = min(starts.shape[0] - int(target), costs.shape[0])
            if deficit <= 0:
                break
            eligible = np.isfinite(costs)
            if max_merge_len > 0:
                eligible &= lengths[:-1] + lengths[1:] <= max_merge_len
            ranked = np.sort(np.where(eligible, costs, np.inf))
            admit = min(deficit * TARGET_ADMIT_FACTOR, costs.shape[0])
            thr = float(np.nextafter(ranked[admit - 1], np.inf))
        selected = sync_nonoverlap_matching(
            costs, starts, lengths, thr, max_merge_len=max_merge_len
        )
        if target and selected.size > deficit:
            picked = np.sort(costs[selected])
            selected = selected[costs[selected] < np.nextafter(picked[deficit - 1], np.inf)]
        if selected.size == 0:
            break
        sel = np.zeros(starts.shape[0], dtype=np.bool_)
        sel[selected] = True
        right = np.zeros_like(sel)
        right[1:] = sel[:-1]
        keep = ~right
        new_len = lengths.copy()
        new_tot = totals.copy()
        new_len[selected] += lengths[selected + 1]
        new_tot[selected] += totals[selected + 1]
        starts, lengths, totals = starts[keep], new_len[keep], new_tot[keep]
        per_round.append(int(selected.size))
    return starts, lengths, {
        "merge_policy": "sync_nonoverlap",
        "merge_count": int(sum(per_round)),
        "merge_rounds_run": len(per_round),
        "round_merge_counts": per_round,
        "max_merge_len": int(max_merge_len),
        "merge_target": int(target),
    }


def merge_reference(
    policy: str,
    starts: np.ndarray,
    lengths: np.ndarray,
    totals: np.ndarray,
    threshold: float,
    *,
    rounds: int = 2,
    max_merge_len: int = 0,
    target: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if policy == "heap_reference":
        if target:
            raise ValueError("target-count merge requires policy=sync_nonoverlap")
        return heap_reference_merge(starts, lengths, totals, threshold, max_merge_len=max_merge_len)
    if policy == "sync_nonoverlap":
        return sync_nonoverlap_merge(
            starts, lengths, totals, threshold, rounds=rounds, max_merge_len=max_merge_len, target=target
        )
    raise ValueError(f"unknown merge policy {policy!r}")


def warmup_host_kernels(root: int = 256) -> None:
    """Compile (or load the cached) numba kernels before the first request."""
    from sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_partition import score_energies

    energies = score_energies(torch.zeros(1, root), atom=1, root=root)
    lambda_dp_repair(energies, 2)
    _merge_loop(
        np.array([0, 1], dtype=np.int64),
        np.array([1, 1], dtype=np.int64),
        np.zeros((2, 1), dtype=np.float64),
        1.0,
        root,
    )
    _leaf_totals(np.zeros((1, 2), np.float32), np.array([0, 1], np.int64), np.array([1, 1], np.int64))
