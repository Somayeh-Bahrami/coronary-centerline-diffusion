import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coronarycl.edge_coherence_grouped import (
    edge_coherence_loss,
    group_terms,
)


def fixture():
    torch.manual_seed(7)
    ground_truth = torch.randn(2, 6, 4)
    prediction = (ground_truth + 0.1 * torch.randn_like(ground_truth)).requires_grad_()
    edges = torch.tensor([
        [[0, 1], [1, 2], [0, 4], [-99, -99]],
        [[0, 1], [1, 2], [0, 4], [-99, -99]],
    ])
    edge_mask = torch.tensor([
        [True, True, True, False],
        [True, True, True, False],
    ])
    return prediction, ground_truth, edges, edge_mask, torch.tensor([0.3, 0.7])


def test_group_reduction_matches_independent_manual_calculation():
    prediction, ground_truth, edges, mask, abar = fixture()
    consecutive, nonconsecutive = group_terms(
        prediction, ground_truth, edges, mask, abar, huber_beta=0.05)

    pred_delta = prediction[:, [1, 2, 4], :3] - prediction[:, [0, 1, 0], :3]
    gt_delta = ground_truth[:, [1, 2, 4], :3] - ground_truth[:, [0, 1, 0], :3]
    residual = torch.nn.functional.smooth_l1_loss(
        pred_delta, gt_delta, beta=0.05, reduction="none").mean(-1)
    residual = residual * abar[:, None]
    assert torch.allclose(consecutive, residual[:, :2].mean())
    assert torch.allclose(nonconsecutive, residual[:, 2].mean())


def test_padding_indices_are_never_gathered_and_gradients_are_finite():
    prediction, ground_truth, edges, mask, abar = fixture()
    loss = edge_coherence_loss(
        prediction, ground_truth, edges, mask, abar,
        group_balanced=True,
        consecutive_weight=0.09476100415945009,
        nonconsecutive_weight=0.029309171820258398,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_grouped_mode_rejects_changed_experimental_contract():
    prediction, ground_truth, edges, mask, abar = fixture()
    with pytest.raises(ValueError, match="abar weighting"):
        edge_coherence_loss(
            prediction, ground_truth, edges, mask, abar,
            group_balanced=True, weight_by_abar=False,
            consecutive_weight=1.0, nonconsecutive_weight=1.0)
    with pytest.raises(ValueError, match="Group weights"):
        edge_coherence_loss(
            prediction, ground_truth, edges, mask, abar,
            group_balanced=True,
            consecutive_weight=-1.0, nonconsecutive_weight=1.0)


def test_legacy_mode_delegates_exactly():
    from src.coronarycl.edge_coherence import edge_coherence_loss as legacy

    prediction, ground_truth, edges, mask, abar = fixture()
    edges = torch.where(mask[..., None], edges, torch.zeros_like(edges))
    expected = legacy(prediction, ground_truth, edges, mask, abar)
    actual = edge_coherence_loss(
        prediction, ground_truth, edges, mask, abar,
        group_balanced=False)
    assert torch.equal(actual, expected)
