"""Group-balanced real-graph edge-coherence loss.

The evaluator's GT tree edges are divided by their relationship in the stored
node array: consecutive edges have ``abs(i-j) == 1`` and non-consecutive edges
have ``abs(i-j) > 1``. These labels describe serialization, not anatomy.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def group_terms(
    x0_hat, x0_true, edge_index, edge_mask, abar_t, *, huber_beta=0.05,
):
    """Return per-sample-then-batch means for consecutive/non-consecutive edges."""
    if not math.isfinite(huber_beta) or huber_beta <= 0:
        raise ValueError("huber_beta must be positive and finite")
    x, gt = x0_hat.float(), x0_true.float()
    edges, mask = edge_index, edge_mask.bool()
    if edges.shape != (*mask.shape, 2) or edges.dtype != torch.long:
        raise ValueError(
            "Expected long edge_index (B,E,2) and edge_mask (B,E)")

    # Invalid padding indices must never reach gather.
    safe = torch.where(mask[..., None], edges, torch.zeros_like(edges))
    first = safe[..., 0, None].expand(-1, -1, 3)
    second = safe[..., 1, None].expand(-1, -1, 3)
    pred_delta = (
        x[..., :3].gather(1, second) - x[..., :3].gather(1, first))
    gt_delta = (
        gt[..., :3].gather(1, second) - gt[..., :3].gather(1, first))
    residual = F.smooth_l1_loss(
        pred_delta, gt_delta, reduction="none", beta=huber_beta).mean(-1)
    weighted = residual * abar_t.float().reshape(-1, 1)

    consecutive = mask & ((safe[..., 1] - safe[..., 0]).abs() == 1)
    groups = (consecutive, mask & ~consecutive)
    terms = []
    for group in groups:
        counts = group.sum(1)
        valid = counts > 0
        if valid.any():
            terms.append(
                ((weighted * group).sum(1)[valid] / counts[valid]).mean())
        else:
            terms.append(x.sum() * 0.0)
    return tuple(terms)


def edge_coherence_loss(
    x0_hat,
    x0_true,
    edge_index,
    edge_mask,
    abar_t,
    *,
    huber_beta=0.05,
    weight_by_abar=True,
    hinge_k=0.0,
    hinge_weight=0.0,
    group_balanced=False,
    consecutive_weight=0.0,
    nonconsecutive_weight=0.0,
):
    """Use the legacy real-edge loss or the frozen two-group experiment."""
    if not group_balanced:
        from .edge_coherence import edge_coherence_loss as legacy_loss

        return legacy_loss(
            x0_hat, x0_true, edge_index, edge_mask, abar_t,
            huber_beta=huber_beta,
            weight_by_abar=weight_by_abar,
            hinge_k=hinge_k,
            hinge_weight=hinge_weight,
        )

    if not weight_by_abar or hinge_weight != 0.0:
        raise ValueError(
            "The grouped experiment requires abar weighting and disables hinge")
    weights = (float(consecutive_weight), float(nonconsecutive_weight))
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Group weights must be finite and non-negative")
    if sum(weights) <= 0:
        raise ValueError("At least one group weight must be positive")

    consecutive, nonconsecutive = group_terms(
        x0_hat, x0_true, edge_index, edge_mask, abar_t,
        huber_beta=huber_beta)
    return weights[0] * consecutive + weights[1] * nonconsecutive
