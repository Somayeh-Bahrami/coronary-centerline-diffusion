"""Runs locally on M4 -- no GPU needed."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coronarycl.metrics import (
    chamfer_l2,
    crop_bounds_mm,
    evaluate_case,
    overlap_metric,
    topology_tree_edges,
)


def test_chamfer_l2_zero_for_identical_sets():
    pts = np.random.randn(50, 3)
    assert chamfer_l2(pts, pts) == 0.0


def test_overlap_metric_full_for_identical_sets():
    pts = np.random.randn(50, 3)
    assert overlap_metric(pts, pts, d=0.01) == 1.0


def test_overlap_metric_decreases_with_noise():
    pts = np.random.randn(50, 3)
    noisy = pts + np.random.randn(50, 3) * 5.0
    assert overlap_metric(noisy, pts, d=0.1) < overlap_metric(pts, pts, d=0.1)


def test_identity_curve_metrics_are_ideal():
    gt = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    edges = topology_tree_edges(gt, iso_mm=1.0)
    assert len(edges) == len(gt) - 1
    result = evaluate_case(
        gt, gt, edges=edges,
        xyz_lower_mm=[-3, -3, -3], xyz_upper_mm=[3, 3, 3])
    assert result["chamfer_l2"] == 0.0
    assert result["hd95_mm"] == 0.0
    assert result["overlap@2.0mm"] == 1.0
    assert result["edge_continuity_5x"] == 1.0
    assert result["largest_connected_component_fraction_5x"] == 1.0
    assert result["tree_length_ratio"] == 1.0
    assert result["out_of_crop_fraction"] == 0.0


def test_broken_edge_and_crop_escape_are_detected():
    gt = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    pred = gt.copy()
    pred[3] = [10.0, 10.0, 0.0]
    result = evaluate_case(
        pred, gt, edges=topology_tree_edges(gt, 1.0),
        xyz_lower_mm=[-3, -3, -3], xyz_upper_mm=[3, 3, 3])
    assert result["edge_continuity_5x"] < 1.0
    assert result["largest_connected_component_fraction_5x"] == 0.75
    assert result["tree_length_ratio"] > 1.0
    assert result["out_of_crop_any"]


def test_tree_metrics_do_not_depend_on_dfs_row_order():
    gt = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    permutation = np.array([3, 0, 2, 1])
    first = evaluate_case(gt, gt, edges=topology_tree_edges(gt, 1.0))
    shuffled = gt[permutation]
    second = evaluate_case(
        shuffled, shuffled, edges=topology_tree_edges(shuffled, 1.0))
    assert first["gt_tree_length_mm"] == second["gt_tree_length_mm"]
    assert first["n_topology_edges"] == second["n_topology_edges"]


def test_crop_bounds_match_voxel_centres():
    lower, upper = crop_bounds_mm([105.7, 105.7, 105.0], 0.35)
    np.testing.assert_allclose(lower, -upper)
    np.testing.assert_allclose(upper, [52.675, 52.675, 52.325])


if __name__ == "__main__":
    test_chamfer_l2_zero_for_identical_sets()
    test_overlap_metric_full_for_identical_sets()
    test_overlap_metric_decreases_with_noise()
    print("All tests passed.")
