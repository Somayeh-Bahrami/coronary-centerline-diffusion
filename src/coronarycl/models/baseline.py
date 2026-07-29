"""Step 2.1 — deterministic epipolar-constraint baseline.
Prototype/small-scale on M4 locally (CPU); move full run to Kaggle if slow.

Used to get a floor Chamfer L2 number before the conditional diffusion
model (2.2).

Pipeline: extract 2D centerline from each view's vessel mask (already
segmented in Step 1.2, no learned model needed) -> match points across
views via epipolar constraint -> triangulate to 3D.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def extract_2d_centerline(mask_2d: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Skeletonize one 2D vessel-mask projection and attach a radius
    (in pixels) at each centerline point via the distance transform.
    Mirrors Step 1.1's 3D skeletonization, applied to a 2D image.

    Args:
        mask_2d: (H, W) vessel-mask projection (from Step 1.2). Values
            are continuous (projected density), not already binary --
            thresholded here first.
        threshold: fraction of the mask's max value used to binarize.

    Returns:
        (N, 3) array of (row, col, radius) in pixel coordinates.
    """
    binary_mask = mask_2d > (mask_2d.max() * threshold)
    skeleton = skeletonize(binary_mask)
    dist = distance_transform_edt(binary_mask)

    coords = np.argwhere(skeleton)          # (N, 2) -- (row, col)
    radii = dist[skeleton]                   # (N,)
    return np.concatenate([coords, radii[:, None]], axis=1)  # (N, 3)


def triangulate_point(pt_view0, pt_view1, pose0, pose1):
    """Placeholder for standard two-view triangulation (DLT) given
    matched 2D points and their projection matrices.
    """
    raise NotImplementedError("Implement DLT triangulation once poses are available.")


def epipolar_baseline(views, poses):
    """Match candidate centerline points across the 2 views via the
    epipolar constraint, then triangulate to a 3D point cloud.

    Args:
        views: list of 2 2D centerline-point arrays (from 2D vessel
               segmentation of each projection).
        poses: list of 2 projection matrices/poses (from Step 1.2).

    Returns:
        (N, 3) array of triangulated 3D points.
    """
    raise NotImplementedError("Baseline TODO — see docs/work_breakdown.md Step 2.1.")
