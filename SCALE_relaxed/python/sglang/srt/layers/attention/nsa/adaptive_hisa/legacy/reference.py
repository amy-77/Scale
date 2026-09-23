"""Causal Adaptive-HISA reference.

One 256-token root is partitioned once, then frozen. Later roots are appended;
they never rewrite earlier leaves. The incomplete tail is not summarized: it is
forced into the candidate set and exact-reranked.

Coordinates are logical token ids. ``logical_to_physical`` is the only place a
page table is applied, so a permutation of physical pages cannot change the
selected logical tokens.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import (
    ATOM,
    CANDIDATE_TOKENS,
    FP8_MAX,
    HEAD_DIM,
    INDEX_TOPK,
    SUMMARY_COMPRESSION,
    PAGE_SIZE,
    ROOT,
    SINK_TOKENS,
    TAIL_TOKENS,
)


def n_complete(n_tokens: int, root: int = ROOT) -> int:
    return (n_tokens // root) * root


def requant_mean(mean: torch.Tensor, fp8_max: float = FP8_MAX) -> torch.Tensor:
    scale = torch.clamp_min(mean.abs().amax(dim=-1, keepdim=True) / fp8_max, 1e-10)
    return (mean / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale


@dataclass
class Partition:
    start: torch.Tensor  # [M] int64
    length: torch.Tensor  # [M] int64
    n_complete: int
    atom: int = ATOM
    method: str = ""

    def __post_init__(self) -> None:
        if self.start.numel() == 0:
            return
        if int(self.start[0]) != 0 or int((self.start + self.length).sum()) != self.n_complete:
            # coverage check via sort, not via the sum of lengths alone
            order = torch.argsort(self.start)
            start = self.start[order]
            length = self.length[order]
            if int(start[0]) != 0:
                raise ValueError("partition does not start at 0")
            ends = start + length
            if int(ends[-1]) != self.n_complete or not bool(torch.equal(ends[:-1], start[1:])):
                raise ValueError("partition must cover the sealed prefix without gaps or overlap")
            self.start = start
            self.length = length
        if not bool((self.length % self.atom == 0).all() and (self.start % self.atom == 0).all()):
            raise ValueError("leaves must be atom-aligned")


def _node_radius_energy(block: torch.Tensor) -> float:
    mu = block.mean(dim=0)
    radius2 = ((block - mu) ** 2).sum(dim=-1).max()
    return float(block.shape[0] * radius2)


def _node_key_energy(block: torch.Tensor) -> float:
    mu = block.mean(dim=0)
    return float(((block - mu) ** 2).sum())


def _energies(keys: torch.Tensor, atom: int, kind: str) -> list[torch.Tensor]:
    n = keys.shape[0]
    if n % atom:
        raise ValueError(f"sealed length {n} is not a multiple of atom {atom}")
    metric = _node_radius_energy if kind == "radius" else _node_key_energy
    level = 0
    length = atom
    out: list[torch.Tensor] = []
    while length <= n:
        vals = [metric(keys[s : s + length]) for s in range(0, n, length)]
        out.append(torch.tensor(vals, dtype=torch.float64))
        level += 1
        length = atom * (1 << level)
    return out


def _partition_from_energy(keys: torch.Tensor, *, atom: int, leaves: int, kind: str) -> Partition:
    n = keys.shape[0]
    n_atoms = n // atom
    if not (1 <= leaves <= n_atoms):
        raise ValueError(f"leaves={leaves} infeasible for {n_atoms} atoms")
    if kind == "fixed":
        leaf = n // leaves
        if leaf % atom or n % leaf:
            raise ValueError(f"fixed leaf {leaf} does not tile {n}")
        start = torch.arange(leaves, dtype=torch.long) * leaf
        length = torch.full((leaves,), leaf, dtype=torch.long)
        return Partition(start, length, n, atom, method="fixed")
    energy = _energies(keys, atom, kind)
    levels = len(energy)

    def collect(lam: float) -> list[tuple[int, int]]:
        do_split = [torch.zeros(energy[0].numel(), dtype=torch.bool)]
        cost = energy[0] + lam
        for level in range(1, levels):
            split = cost[0::2] + cost[1::2]
            keep = energy[level] + lam
            ds = split < keep
            cost = torch.where(ds, split, keep)
            do_split.append(ds)
        active = torch.ones(energy[-1].numel(), dtype=torch.bool)
        leaves_out: list[tuple[int, int]] = []
        for level in range(levels - 1, -1, -1):
            if level == 0:
                leaves_out.extend((0, int(i)) for i in torch.nonzero(active).flatten().tolist())
                break
            is_leaf = active & ~do_split[level]
            leaves_out.extend((level, int(i)) for i in torch.nonzero(is_leaf).flatten().tolist())
            active = torch.repeat_interleave(active & do_split[level], 2)
        return leaves_out

    hi = float(energy[-1].max()) * 2 + 1
    best = collect(hi)
    lo = 0.0
    if len(collect(0.0)) <= leaves:
        best = collect(0.0)
    else:
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            got = collect(mid)
            if len(got) <= leaves:
                best = got
                hi = mid
            else:
                lo = mid
    leaf_set = set(best)
    heap: list[tuple[float, int, int, int]] = []

    def gain(level: int, idx: int) -> float:
        return float(energy[level][idx] - energy[level - 1][2 * idx] - energy[level - 1][2 * idx + 1])

    for level, idx in leaf_set:
        if level:
            heapq.heappush(heap, (-gain(level, idx), idx * atom * (1 << level), level, idx))
    while len(leaf_set) < leaves:
        _, _, level, idx = heapq.heappop(heap)
        if (level, idx) not in leaf_set:
            continue
        leaf_set.remove((level, idx))
        for child in (2 * idx, 2 * idx + 1):
            leaf_set.add((level - 1, child))
            if level - 1:
                heapq.heappush(
                    heap,
                    (-gain(level - 1, child), child * atom * (1 << (level - 1)), level - 1, child),
                )
    starts, lens = [], []
    for level, idx in leaf_set:
        ln = atom * (1 << level)
        starts.append(idx * ln)
        lens.append(ln)
    order = torch.argsort(torch.tensor(starts))
    start_t = torch.tensor(starts, dtype=torch.long)[order]
    len_t = torch.tensor(lens, dtype=torch.long)[order]
    return Partition(start_t, len_t, n, atom, method=kind)


def partition_root(keys: torch.Tensor, *, policy: str = "radius", atom: int = ATOM, root: int = ROOT) -> Partition:
    if keys.shape[0] != root:
        raise ValueError(f"a root must contain {root} tokens, got {keys.shape[0]}")
    kind = {"radius": "radius", "key": "key", "fixed": "fixed"}[policy]
    return _partition_from_energy(keys.to(torch.float32), atom=atom, leaves=root // 8, kind=kind)


def leaf_means(keys: torch.Tensor, part: Partition, *, requant: bool = True) -> torch.Tensor:
    out = torch.empty((part.start.numel(), keys.shape[1]), dtype=torch.float32)
    for i, (start, length) in enumerate(zip(part.start.tolist(), part.length.tolist())):
        mean = keys[start : start + length].to(torch.float32).mean(dim=0)
        out[i] = requant_mean(mean) if requant else mean
    return out


@dataclass
class RequestLeaves:
    """Frozen leaves of one request/layer. ``epoch`` changes when the slot is freed."""

    epoch: int = 0
    sealed_end: int = 0
    start: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    length: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    mean: torch.Tensor = field(default_factory=lambda: torch.empty(0, HEAD_DIM))
    method: str = ""

    def append_root(self, keys_root: torch.Tensor, *, policy: str, offset: int, requant: bool = True) -> None:
        part = partition_root(keys_root, policy=policy)
        self.start = torch.cat([self.start, part.start + offset])
        self.length = torch.cat([self.length, part.length])
        self.mean = torch.cat([self.mean, leaf_means(keys_root, part, requant=requant)])
        self.sealed_end = offset + keys_root.shape[0]
        self.method = part.method


def seal_available(
    state: RequestLeaves,
    keys: torch.Tensor,
    *,
    policy: str = "radius",
    requant: bool = True,
) -> RequestLeaves:
    """Seal every newly completed root. Keys at or after ``sealed_end`` are ignored by old leaves."""
    done = n_complete(keys.shape[0])
    if done < state.sealed_end:
        raise ValueError("prefix shrank; free the request slot instead of rewinding leaves")
    cursor = state.sealed_end
    while cursor + ROOT <= done:
        state.append_root(keys[cursor : cursor + ROOT], policy=policy, offset=cursor, requant=requant)
        cursor += ROOT
    return state


def _guard_atoms(ke: int, *, atom: int, sink_tokens: int, tail_tokens: int) -> list[int]:
    if ke <= 0:
        return []
    last = (ke - 1) // atom
    atoms = set(range(min(sink_tokens // atom, last + 1)))
    tail_start = max(0, ke - tail_tokens)
    atoms |= set(range(tail_start // atom, last + 1))
    return sorted(a for a in atoms if a * atom < ke)


def select_candidates(
    state: RequestLeaves,
    ke: int,
    *,
    budget: int = CANDIDATE_TOKENS,
    atom: int = ATOM,
    sink_tokens: int = SINK_TOKENS,
    tail_tokens: int = TAIL_TOKENS,
    query: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[budget]`` logical token ids. ``-1`` pads unused slots.

    ``query`` is ``[H, D]`` dequantized and ``weights`` is ``[H]``. When omitted,
    leaves are ranked by their stored order (tests of the budget rule only).
    """
    if budget % atom:
        raise ValueError("candidate budget must be a multiple of atom")
    slots = budget // atom
    chosen: list[int] = []
    seen: set[int] = set()
    for atom_id in _guard_atoms(ke, atom=atom, sink_tokens=sink_tokens, tail_tokens=tail_tokens):
        if atom_id not in seen:
            seen.add(atom_id)
            chosen.append(atom_id)
    if len(chosen) > slots:
        chosen = chosen[:slots]
    if state.start.numel() and len(chosen) < slots:
        visible = state.start + state.length <= ke
        if scores is None:
            scores = leaf_scores(state.mean, query, weights) if query is not None else torch.arange(state.mean.shape[0], dtype=torch.float32)
        scores = torch.where(visible, scores, torch.full_like(scores, -torch.inf))
        order_start = torch.argsort(state.start, stable=True)
        order = order_start[torch.argsort(-scores[order_start], stable=True)]
        a0 = state.start // atom
        a1 = (state.start + state.length) // atom
        for leaf in order.tolist():
            if len(chosen) >= slots or not bool(visible[leaf]):
                continue
            atoms = [a for a in range(int(a0[leaf]), int(a1[leaf])) if a not in seen and a * atom < ke]
            if not atoms:
                continue
            take = atoms[: slots - len(chosen)]
            chosen.extend(take)
            seen.update(take)
    # Exact tail / any not-yet-atomized token is already inside the tail guard.
    out = torch.full((budget,), -1, dtype=torch.long)
    for slot, atom_id in enumerate(chosen):
        base = atom_id * atom
        for off in range(atom):
            token = base + off
            dst = slot * atom + off
            if dst < budget and token < ke:
                out[dst] = token
    return out


def leaf_scores(mean: torch.Tensor, query: torch.Tensor | None, weights: torch.Tensor | None) -> torch.Tensor:
    if query is None or weights is None:
        return torch.zeros(mean.shape[0], dtype=torch.float32)
    dots = torch.relu(query.to(torch.float32) @ mean.to(torch.float32).T)  # [H, M]
    return (weights.to(torch.float32)[:, None] * dots).sum(dim=0)


def canonical_scores(
    query_fp8: torch.Tensor,
    weights: torch.Tensor,
    keys_fp8: torch.Tensor,
    key_scale: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Exact indexer score of one query against ``token_ids`` (``-1`` -> -inf)."""
    valid = token_ids >= 0
    scores = torch.full((token_ids.numel(),), -torch.inf)
    if not bool(valid.any()):
        return scores
    ids = token_ids[valid]
    keys = keys_fp8[ids].to(torch.float32) * key_scale[ids, None]
    q = query_fp8.to(torch.float32)
    dots = torch.relu(torch.einsum("hd,nd->hn", q, keys))
    scores[valid] = (weights.to(torch.float32)[:, None] * dots).sum(dim=0)
    return scores


def rerank(scores: torch.Tensor, token_ids: torch.Tensor, topk: int = INDEX_TOPK) -> torch.Tensor:
    """Stable top-k. Ties break toward the smaller logical token, then slot."""
    finite = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -1e30))
    # lexsort: last key is primary. Higher score first, then smaller token, then smaller slot.
    slots = torch.arange(token_ids.numel())
    token_key = torch.where(token_ids >= 0, token_ids, torch.full_like(token_ids, 2**31 - 1))
    order = torch.argsort(slots, stable=True)
    order = order[torch.argsort(token_key[order], stable=True)]
    order = order[torch.argsort(-finite[order], stable=True)]
    picked = token_ids[order[:topk]]
    out = torch.full((topk,), -1, dtype=torch.long)
    out[: picked.numel()] = picked
    # Drop sentinel if the candidate list itself was shorter than topk and padded.
    out[out < 0] = -1
    return out


def logical_to_physical(token_ids: torch.Tensor, page_table: torch.Tensor, page_size: int = PAGE_SIZE) -> torch.Tensor:
    out = torch.full_like(token_ids, -1)
    valid = token_ids >= 0
    if not bool(valid.any()):
        return out
    ids = token_ids[valid]
    page = ids // page_size
    if int(page.max()) >= page_table.numel():
        raise IndexError("logical token is outside the page table")
    physical_page = page_table[page].to(torch.long)
    out[valid] = physical_page * page_size + ids % page_size
    return out


def pack_index_buffer(keys_fp8: torch.Tensor, key_scale: torch.Tensor, page_size: int = PAGE_SIZE) -> torch.Tensor:
    """Page-major ``[num_pages, page*(D+4)]`` uint8 buffer, identity page order."""
    n, dim = keys_fp8.shape
    pages = (n + page_size - 1) // page_size
    buf = torch.zeros((pages, page_size * (dim + 4)), dtype=torch.uint8)
    flat_k = torch.zeros((pages * page_size, dim), dtype=torch.uint8)
    flat_k[:n] = keys_fp8.view(torch.uint8)
    flat_s = torch.zeros((pages * page_size,), dtype=torch.float32)
    flat_s[:n] = key_scale
    for p in range(pages):
        k0 = p * page_size * dim
        buf[p, : page_size * dim] = flat_k[p * page_size : (p + 1) * page_size].reshape(-1)
        scale_bytes = flat_s[p * page_size : (p + 1) * page_size].view(torch.uint8)
        buf[p, page_size * dim :] = scale_bytes
    return buf


def gather_compact(
    buf: torch.Tensor,
    page_table: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    page_size: int = PAGE_SIZE,
    dim: int = HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only the referenced logical tokens. Invalid ids yield zeros."""
    n = token_ids.numel()
    keys = torch.zeros((n, dim), dtype=torch.uint8)
    scale = torch.zeros((n,), dtype=torch.float32)
    valid = token_ids >= 0
    if not bool(valid.any()):
        return keys, scale
    ids = token_ids[valid]
    physical = logical_to_physical(ids, page_table, page_size)
    page = physical // page_size
    off = physical % page_size
    for dst, (p, o) in enumerate(zip(page.tolist(), off.tolist())):
        k_off = o * dim
        keys[torch.nonzero(valid).flatten()[dst]] = buf[p, k_off : k_off + dim]
        scale_off = page_size * dim + o * 4
        scale[torch.nonzero(valid).flatten()[dst]] = buf[p, scale_off : scale_off + 4].view(torch.float32)[0]
    return keys, scale


def adaptive_topk(
    keys: torch.Tensor,
    query_fp8: torch.Tensor,
    weights: torch.Tensor,
    ke: int,
    *,
    policy: str = "radius",
    budget: int = CANDIDATE_TOKENS,
    topk: int = INDEX_TOPK,
    state: RequestLeaves | None = None,
) -> tuple[torch.Tensor, RequestLeaves]:
    """Full causal reference: seal, select ``budget`` tokens, exact-rerank to ``topk``."""
    state = state or RequestLeaves()
    q = query_fp8.to(torch.float32)
    # Scale is 1 for an already-dequantized reference key tensor. Callers that
    # have fp8 keys pass them through ``keys_fp8`` via the runtime path.
    seal_available(state, keys[:ke], policy=policy)
    candidates = select_candidates(state, ke, budget=budget, query=q, weights=weights)
    scale = torch.ones(keys.shape[0], dtype=torch.float32)
    scores = canonical_scores(query_fp8, weights, keys, scale, candidates.clamp(min=0))
    scores = torch.where(candidates >= 0, scores, torch.full_like(scores, -torch.inf))
    # ``keys`` here is the dequantized reference, so pretend it is fp8 with unit scale.
    # canonical_scores multiplies by scale; pass keys as float stored in a float tensor
    # through a dedicated branch:
    valid = candidates >= 0
    scores = torch.full((candidates.numel(),), -torch.inf)
    if bool(valid.any()):
        ids = candidates[valid]
        dots = torch.relu(torch.einsum("hd,nd->hn", q, keys[ids].to(torch.float32)))
        scores[valid] = (weights.to(torch.float32)[:, None] * dots).sum(dim=0)
    return rerank(scores, candidates, topk), state
