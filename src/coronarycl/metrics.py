"""Physical point-set and given-topology centerline metrics.

``chamfer_l2`` deliberately preserves this project's historical convention:
the sum of the two directed mean Euclidean distances (unsquared). Curve
metrics use a deterministic tree recovered from GT skeleton neighbours. They
therefore measure predicted coordinates under *given/oracle GT topology*;
they do not claim that the model predicted connectivity.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def _xyz(points, name):
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 3 or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty (N,3+) array")
    value = value[:, :3]
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return value


def ground_truth_to_mm(centerline_voxel, volume_shape, voxel_spacing):
    """Convert voxel coordinates to the TIGRE volume-centred mm frame."""
    center_voxel = np.asarray(volume_shape) / 2
    xyz_voxel = np.asarray(centerline_voxel)[:, :3]
    return (xyz_voxel - center_voxel) * np.asarray(voxel_spacing)


def chamfer_l2(pred, gt):
    """Sum of directed mean Euclidean NN distances, in mm (not squared)."""
    pred, gt = _xyz(pred, "pred"), _xyz(gt, "gt")
    pred_to_gt, _ = cKDTree(gt).query(pred)
    gt_to_pred, _ = cKDTree(pred).query(gt)
    return float(pred_to_gt.mean() + gt_to_pred.mean())


def hausdorff95(pred, gt):
    """Maximum of the two directed 95th-percentile distances, in mm."""
    pred, gt = _xyz(pred, "pred"), _xyz(gt, "gt")
    pred_to_gt, _ = cKDTree(gt).query(pred)
    gt_to_pred, _ = cKDTree(pred).query(gt)
    return float(max(np.percentile(pred_to_gt, 95),
                     np.percentile(gt_to_pred, 95)))


def overlap_metric(pred, gt, d):
    """Symmetric Ot(d): mean of the two directed in-threshold fractions."""
    distance_mm = float(d)
    if not np.isfinite(distance_mm) or distance_mm <= 0:
        raise ValueError("overlap threshold must be finite and positive")
    pred, gt = _xyz(pred, "pred"), _xyz(gt, "gt")
    pred_to_gt, _ = cKDTree(gt).query(pred)
    gt_to_pred, _ = cKDTree(pred).query(gt)
    return float(
        ((pred_to_gt <= distance_mm).mean()
         + (gt_to_pred <= distance_mm).mean()) / 2.0)


def crop_bounds_mm(s_voxel, iso_mm):
    """Exact symmetric bounds of voxel centres in the cropped volume."""
    s_voxel = np.asarray(s_voxel, dtype=np.float64).reshape(3)
    iso_mm = float(iso_mm)
    if not np.isfinite(s_voxel).all() or np.any(s_voxel <= 0):
        raise ValueError("sVoxel must contain three finite positive extents")
    if not np.isfinite(iso_mm) or iso_mm <= 0:
        raise ValueError("iso_mm must be finite and positive")
    half_extent = (s_voxel - iso_mm) / 2.0
    if np.any(half_extent <= 0):
        raise ValueError("crop extent must exceed one isotropic voxel")
    return -half_extent, half_extent


def _candidate_skeleton_edges(gt, iso_mm, rtol):
    radius = math.sqrt(3.0) * iso_mm * (1.0 + rtol)
    pairs = cKDTree(gt).query_pairs(radius, output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    delta = np.abs(gt[pairs[:, 0]] - gt[pairs[:, 1]])
    keep = np.all(delta <= iso_mm * (1.0 + rtol), axis=1)
    return pairs[keep]


def topology_tree_edges(gt_mm, iso_mm, *, rtol=1e-3):
    """Recover a deterministic minimum spanning forest of GT 26-neighbours.

    Dataset rows are depth-first-search ordered. Consecutive rows can be a
    backtracking jump at a bifurcation, so they are not a valid polyline. This
    function first finds true 26-neighbour skeleton connections, then removes
    redundant local cycles with deterministic Kruskal ordering.
    """
    gt = _xyz(gt_mm, "gt_mm")
    iso_mm = float(iso_mm)
    if not np.isfinite(iso_mm) or iso_mm <= 0:
        raise ValueError("iso_mm must be finite and positive")
    candidates = _candidate_skeleton_edges(gt, iso_mm, float(rtol))
    if len(candidates) == 0:
        return np.empty((0, 2), dtype=np.int64)

    lengths = np.linalg.norm(
        gt[candidates[:, 0]] - gt[candidates[:, 1]], axis=1)
    order = np.lexsort((candidates[:, 1], candidates[:, 0], lengths))
    parent = np.arange(len(gt), dtype=np.int64)
    rank = np.zeros(len(gt), dtype=np.int8)

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            node, parent[node] = parent[node], root
        return root

    edges = []
    for index in order:
        left, right = map(int, candidates[index])
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        edges.append((left, right))
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def _component_summary(n_nodes, edges):
    parent = np.arange(n_nodes, dtype=np.int64)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        left_root, right_root = find(int(left)), find(int(right))
        if left_root != right_root:
            parent[right_root] = left_root
    roots = np.asarray([find(i) for i in range(n_nodes)])
    _, counts = np.unique(roots, return_counts=True)
    return int(len(counts)), float(counts.max() / n_nodes)


def curve_metrics(pred_mm, gt_mm, edges, *, discontinuity_factor=5.0):
    """Given-topology continuity and tree-length metrics."""
    pred, gt = _xyz(pred_mm, "pred_mm"), _xyz(gt_mm, "gt_mm")
    if len(pred) != len(gt):
        raise ValueError("given-topology metrics require equal node counts")
    edge_index = np.asarray(edges, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[1:] != (2,):
        raise ValueError("edges must have shape (E,2)")
    if not np.isfinite(discontinuity_factor) or discontinuity_factor <= 1:
        raise ValueError("discontinuity_factor must be finite and > 1")
    if len(edge_index) == 0:
        return {
            "n_topology_edges": 0,
            "n_topology_components": len(gt),
            "edge_continuity_5x": float("nan"),
            "broken_edge_fraction_5x": float("nan"),
            "largest_connected_component_fraction_5x": 1.0 / len(gt),
            "pred_edge_length_median_mm": float("nan"),
            "pred_edge_length_p95_mm": float("nan"),
            "pred_tree_length_mm": 0.0,
            "gt_tree_length_mm": 0.0,
            "tree_length_ratio": float("nan"),
        }
    if edge_index.min() < 0 or edge_index.max() >= len(gt):
        raise ValueError("edge index is outside the point array")

    pred_lengths = np.linalg.norm(
        pred[edge_index[:, 0]] - pred[edge_index[:, 1]], axis=1)
    gt_lengths = np.linalg.norm(
        gt[edge_index[:, 0]] - gt[edge_index[:, 1]], axis=1)
    if np.any(gt_lengths <= 0):
        raise ValueError("GT topology contains a zero-length edge")
    continuous = pred_lengths <= discontinuity_factor * gt_lengths
    topology_components, _ = _component_summary(len(gt), edge_index)
    _, largest_fraction = _component_summary(
        len(gt), edge_index[continuous])
    pred_total = float(pred_lengths.sum())
    gt_total = float(gt_lengths.sum())
    suffix = f"{discontinuity_factor:g}x"
    return {
        "n_topology_edges": int(len(edge_index)),
        "n_topology_components": topology_components,
        f"edge_continuity_{suffix}": float(continuous.mean()),
        f"broken_edge_fraction_{suffix}": float((~continuous).mean()),
        f"largest_connected_component_fraction_{suffix}": largest_fraction,
        "pred_edge_length_median_mm": float(np.median(pred_lengths)),
        "pred_edge_length_p95_mm": float(np.percentile(pred_lengths, 95)),
        "pred_tree_length_mm": pred_total,
        "gt_tree_length_mm": gt_total,
        "tree_length_ratio": float(pred_total / gt_total),
    }


def out_of_crop_metrics(
    pred_mm, xyz_lower_mm, xyz_upper_mm, *, tolerance_mm=1e-4,
):
    """Node-level crop violations and overflow beyond numeric tolerance."""
    pred = _xyz(pred_mm, "pred_mm")
    lower = np.asarray(xyz_lower_mm, dtype=np.float64).reshape(3)
    upper = np.asarray(xyz_upper_mm, dtype=np.float64).reshape(3)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("crop bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("crop lower bounds must be below upper bounds")
    if tolerance_mm < 0:
        raise ValueError("tolerance_mm must be non-negative")
    # A normalized clamp followed by float32 de-normalization can land a few
    # micrometres beyond the algebraically identical mm boundary. Do not label
    # that roundoff as a clinically meaningful crop escape.
    overflow = (
        np.maximum(lower - pred - tolerance_mm, 0.0)
        + np.maximum(pred - upper - tolerance_mm, 0.0))
    overflow_mm = np.linalg.norm(overflow, axis=1)
    outside = overflow_mm > 0
    return {
        "out_of_crop_fraction": float(outside.mean()),
        "out_of_crop_any": bool(outside.any()),
        "mean_crop_excess_mm": float(overflow_mm.mean()),
        "max_crop_excess_mm": float(overflow_mm.max()),
    }


def evaluate_case(
    pred, gt, thresholds=(1.0, 2.0, 5.0), *,
    edges=None, iso_mm=None, xyz_lower_mm=None, xyz_upper_mm=None,
    discontinuity_factor=5.0,
):
    """Return point-set plus optional given-topology/crop metrics."""
    pred, gt = _xyz(pred, "pred"), _xyz(gt, "gt")
    results = {
        "chamfer_l2": chamfer_l2(pred, gt),
        "hd95_mm": hausdorff95(pred, gt),
    }
    for threshold in thresholds:
        results[f"overlap@{float(threshold)}mm"] = overlap_metric(
            pred, gt, float(threshold))
    if edges is None and iso_mm is not None:
        edges = topology_tree_edges(gt, iso_mm)
    if edges is not None:
        results.update(curve_metrics(
            pred, gt, edges,
            discontinuity_factor=discontinuity_factor))
    if xyz_lower_mm is not None or xyz_upper_mm is not None:
        if xyz_lower_mm is None or xyz_upper_mm is None:
            raise ValueError("both crop bounds are required")
        results.update(out_of_crop_metrics(
            pred, xyz_lower_mm, xyz_upper_mm))
    return results
