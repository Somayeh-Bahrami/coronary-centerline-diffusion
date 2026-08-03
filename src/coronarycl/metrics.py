"""Step 3.1 — quantitative evaluation. Runs fully on M4 locally —
lightweight distance computations, no GPU needed.

Chamfer L2 distance + threshold-based overlap metric Ot(d), following
DeepCA's (Wang et al., WACV 2025) evaluation protocol.
"""

import numpy as np
from scipy.spatial import cKDTree


def ground_truth_to_mm(centerline_voxel: np.ndarray, volume_shape, voxel_spacing) -> np.ndarray:
    """Convert raw voxel-index centerline coordinates (Step 1.1's saved
    format, e.g. data/processed/centerlines/<id>_centerline.npy) into
    the same volume-centered mm coordinate frame TIGRE used during DRR
    generation (Step 1.2) -- required before comparing against
    triangulated/predicted 3D points, which already live in that frame.

    Args:
        centerline_voxel: (N, 4+) array, first 3 columns are raw
            (x, y, z) voxel indices.
        volume_shape: shape of the source CT volume (e.g. from
            nibabel's .shape), used to find the center voxel.
        voxel_spacing: (3,) physical spacing per voxel axis, in mm
            (e.g. from nibabel's header.get_zooms()[:3]).

    Returns:
        (N, 3) array of (x, y, z) in centered mm coordinates.
    """
    center_voxel = np.array(volume_shape) / 2
    xyz_voxel = centerline_voxel[:, :3]
    xyz_mm = (xyz_voxel - center_voxel) * np.array(voxel_spacing)
    return xyz_mm


def chamfer_l2(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric Chamfer L2 distance between two (N,3)/(M,3) point sets."""
    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)
    d_pred_to_gt, _ = tree_gt.query(pred)
    d_gt_to_pred, _ = tree_pred.query(gt)
    return float(d_pred_to_gt.mean() + d_gt_to_pred.mean())


def overlap_metric(pred: np.ndarray, gt: np.ndarray, d: float) -> float:
    """Ot(d): fraction of predicted points within threshold distance d
    of the ground truth (and vice versa), following DeepCA's protocol.
    Useful under motion/deformation where exact point correspondence
    isn't expected.
    """
    tree_gt = cKDTree(gt)
    dist_pred_to_gt, _ = tree_gt.query(pred)
    frac_pred = (dist_pred_to_gt <= d).mean()

    tree_pred = cKDTree(pred)
    dist_gt_to_pred, _ = tree_pred.query(gt)
    frac_gt = (dist_gt_to_pred <= d).mean()

    return float((frac_pred + frac_gt) / 2)


def evaluate_case(pred: np.ndarray, gt: np.ndarray, thresholds=(1.0, 2.0, 5.0)):
    """Returns Chamfer L2 plus Ot(d) at each threshold in `thresholds` (mm).

    Both pred and gt must already be in the same coordinate frame --
    if gt comes from Step 1.1's raw saved centerline, convert it first
    via ground_truth_to_mm().
    """
    results = {"chamfer_l2": chamfer_l2(pred, gt)}
    for d in thresholds:
        results[f"overlap@{d}mm"] = overlap_metric(pred, gt, d)
    return results
