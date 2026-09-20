"""Per-layer candidate-wise Assignment Router shared by training and runtime."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def standardize(x: torch.Tensor, dim: int) -> torch.Tensor:
    return (x - x.mean(dim=dim, keepdim=True)) / x.std(
        dim=dim, keepdim=True, correction=0
    ).clamp_min(1e-5)


class LayerwiseLinear(nn.Module):
    """Independent bias-free linear projection for every transformer layer."""

    def __init__(self, n_layers: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(n_layers, output_dim, input_dim)
        )
        for weight in self.weight:
            nn.init.xavier_uniform_(weight)

    def forward(
        self, x: torch.Tensor, layer_ids: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("b...i,boi->b...o", x, self.weight[layer_ids])


class MISAAssignmentRouter(nn.Module):
    """Return assignment scores shaped [batch, MLA groups, MISA candidates].

    Group16 training concatenates the 16 queries already local to a TP rank.
    Every layer has independent projections and head embeddings. The first
    version intentionally has no layer embedding, pair embedding, or static
    assignment bias, so improvements must come from the current query/context.
    """

    def __init__(
        self,
        q_dim: int,
        indexer_dim: int,
        context_dim: int,
        context_stats_dim: int,
        n_layers: int,
        rank: int = 64,
        n_indexer_heads: int = 64,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.q_proj = LayerwiseLinear(n_layers, q_dim, rank)
        self.indexer_q_proj = LayerwiseLinear(n_layers, indexer_dim, rank)
        self.context_proj = LayerwiseLinear(n_layers, context_dim, rank)
        self.indexer_embedding = nn.Parameter(
            torch.randn(n_layers, n_indexer_heads, rank) * 0.02
        )
        self.feature_proj = LayerwiseLinear(
            n_layers, 2 + context_stats_dim, 1
        )

    def forward(
        self,
        q_group: torch.Tensor,
        candidate_head_ids: torch.Tensor,
        candidate_q_indexer: torch.Tensor,
        candidate_context_summary: torch.Tensor,
        candidate_gate: torch.Tensor,
        candidate_importance: torch.Tensor,
        candidate_context_stats: torch.Tensor,
        layer_ids: torch.Tensor,
    ) -> torch.Tensor:
        group = self.q_proj(
            F.layer_norm(q_group, (q_group.shape[-1],)), layer_ids
        )
        candidate = self.indexer_embedding[
            layer_ids.unsqueeze(1), candidate_head_ids
        ]
        candidate = candidate + self.indexer_q_proj(
            F.layer_norm(
                candidate_q_indexer, (candidate_q_indexer.shape[-1],)
            ),
            layer_ids,
        )
        candidate = candidate + self.context_proj(
            F.layer_norm(
                candidate_context_summary,
                (candidate_context_summary.shape[-1],),
            ),
            layer_ids,
        )
        score = torch.einsum("bgr,bmr->bgm", group, candidate) / math.sqrt(
            self.rank
        )
        features = torch.cat(
            [
                standardize(candidate_importance, 1).unsqueeze(-1),
                standardize(candidate_gate, 1).unsqueeze(-1),
                standardize(candidate_context_stats, 1),
            ],
            dim=-1,
        )
        return score + self.feature_proj(features, layer_ids).squeeze(
            -1
        ).unsqueeze(1)
