"""Page-aligned atom statistics and per-request leaf epochs.

Atom sums live with physical pages, so a radix-cache hit can rebuild a logical
view through the page table. Leaf partitions do not survive ``free``: the slot
epoch changes and the next request seals its own roots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import ATOM, HEAD_DIM, PAGE_SIZE
from sglang.srt.layers.attention.nsa.adaptive_hisa.legacy.reference import RequestLeaves


@dataclass
class PageAtoms:
    """``sum`` is ``[num_pages, atoms_per_page, D]`` and ``count`` matches it."""

    sum: torch.Tensor
    count: torch.Tensor
    epoch_by_req: dict[int, int] = field(default_factory=dict)
    leaves_by_req: dict[int, RequestLeaves] = field(default_factory=dict)

    @classmethod
    def empty(cls, num_pages: int, device, atoms_per_page: int = PAGE_SIZE // ATOM) -> "PageAtoms":
        return cls(
            sum=torch.zeros((num_pages, atoms_per_page, HEAD_DIM), dtype=torch.float32, device=device),
            count=torch.zeros((num_pages, atoms_per_page), dtype=torch.int32, device=device),
        )


def update_atom_stats(stats: PageAtoms, key: torch.Tensor, loc: torch.Tensor, page_size: int = PAGE_SIZE, atom: int = ATOM) -> None:
    """Fold newly stored BF16 keys into their physical atoms.

    The first token of an atom (``loc % atom == 0``) resets that atom. Pages are
    request-exclusive and prefix hits are page-aligned, so this does not leak a
    previous tenant into a full atom. A partial tail atom is rebuilt the next
    time its first token is written.
    """
    if key.numel() == 0:
        return
    loc = loc.to(torch.int64)
    page = loc // page_size
    atom_in_page = (loc % page_size) // atom
    fresh = atom_in_page[(loc % atom) == 0]
    fresh_page = page[(loc % atom) == 0]
    if fresh.numel():
        stats.sum[fresh_page, fresh] = 0
        stats.count[fresh_page, fresh] = 0
    ones = torch.ones(loc.shape[0], dtype=torch.int32, device=key.device)
    stats.count.index_put_((page, atom_in_page), ones, accumulate=True)
    stats.sum.index_put_((page, atom_in_page), key.to(torch.float32), accumulate=True)


class SlotEpoch:
    """Process-local epoch table. Tests use this directly; serving hooks it."""

    def __init__(self) -> None:
        self.epoch: dict[int, int] = {}
        self.leaves: dict[tuple[int, int], RequestLeaves] = {}

    def current(self, req_idx: int) -> int:
        return self.epoch.get(req_idx, 0)

    def free(self, req_idx: int) -> None:
        self.epoch[req_idx] = self.current(req_idx) + 1
        dead = [key for key in self.leaves if key[1] == req_idx]
        for key in dead:
            del self.leaves[key]

    def state(self, layer_id: int, req_idx: int) -> RequestLeaves:
        key = (layer_id, req_idx)
        epoch = self.current(req_idx)
        state = self.leaves.get(key)
        if state is None or state.epoch != epoch:
            state = RequestLeaves(epoch=epoch)
            self.leaves[key] = state
        return state


SLOTS = SlotEpoch()


def on_req_free(req_idx: int) -> None:
    SLOTS.free(int(req_idx))
