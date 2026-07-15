"""Step 3.1 — quantitative evaluation. Runs fully on M4 locally —
lightweight distance computations, no GPU needed.

Chamfer L2 distance + threshold-based overlap metric Ot(d), following
DeepCA's (Wang et al., WACV 2025) evaluation protocol.
"""

import numpy as np
from scipy.spatial import cKDTree


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
    """Returns Chamfer L2 plus Ot(d) at each threshold in `thresholds` (mm)."""
    results = {"chamfer_l2": chamfer_l2(pred, gt)}
    for d in thresholds:
        results[f"overlap@{d}mm"] = overlap_metric(pred, gt, d)
    return results
