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
    """Two-view DLT triangulation (Hartley & Zisserman, "Multiple View
    Geometry in Computer Vision", 2004) -- given a point's 2D pixel
    location in each of 2 views, and each view's 3x4 projection matrix,
    solves for the 3D point that projects to both observations.

    Args:
        pt_view0: (x, y) pixel coordinates in view 0.
        pt_view1: (x, y) pixel coordinates in view 1.
        pose0: (3, 4) projection matrix for view 0.
        pose1: (3, 4) projection matrix for view 1.

    Returns:
        (3,) array -- the triangulated 3D point.
    """
    x0, y0 = pt_view0
    x1, y1 = pt_view1

    A = np.array([
        x0 * pose0[2, :] - pose0[0, :],
        y0 * pose0[2, :] - pose0[1, :],
        x1 * pose1[2, :] - pose1[0, :],
        y1 * pose1[2, :] - pose1[1, :],
    ])

    _, _, Vt = np.linalg.svd(A)
    point_h = Vt[-1]
    point_3d = point_h[:3] / point_h[3]
    return point_3d
