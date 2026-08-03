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


def compute_camera_center(P: np.ndarray) -> np.ndarray:
    """Camera center C is the null space of the 3x4 projection matrix
    P. Returns C in non-homogeneous (3,) form.
    """
    _, _, Vt = np.linalg.svd(P)
    C_h = Vt[-1]
    return C_h[:3] / C_h[3]


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric cross-product matrix [v]_x, such that
    [v]_x @ w == v cross w for any vector w."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def compute_fundamental_matrix(P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
    """Fundamental matrix F relating view 0 to view 1, computed
    directly from the two projection matrices (no point correspondences
    needed -- valid since we already have calibrated camera geometry).

    F = [e1]_x @ P1 @ pinv(P0), where e1 is the epipole (camera 0's
    center, projected into view 1).
    """
    C0 = compute_camera_center(P0)
    e1_h = P1 @ np.append(C0, 1.0)
    e1_h = e1_h / e1_h[2]

    return skew_symmetric(e1_h) @ P1 @ np.linalg.pinv(P0)


def match_via_epipolar(points0: np.ndarray, points1: np.ndarray,
                       P0: np.ndarray, P1: np.ndarray,
                       distance_threshold: float = 3.0):
    """For each point in view 0, find its best-matching point in view 1
    by distance to the epipolar line, within a pixel threshold.

    Args:
        points0: (N, 2+) array, first 2 columns are (x, y) pixel coords
            in view 0 (any extra columns, e.g. radius, are ignored here
            but preserved in the return).
        points1: (M, 2+) array, same format for view 1.
        P0, P1: (3, 4) projection matrices.
        distance_threshold: max allowed point-to-epipolar-line distance
            (pixels) for a candidate match to be accepted.

    Returns:
        matched0: (K, points0.shape[1]) matched points from view 0.
        matched1: (K, points1.shape[1]) corresponding matched points from view 1.
    """
    F = compute_fundamental_matrix(P0, P1)

    pts0_xy = points0[:, :2]
    pts1_xy = points1[:, :2]

    pts0_h = np.hstack([pts0_xy, np.ones((len(pts0_xy), 1))])
    pts1_h = np.hstack([pts1_xy, np.ones((len(pts1_xy), 1))])

    lines1 = pts0_h @ F.T  # (N, 3), each row is (a, b, c) for ax+by+c=0

    matched0, matched1 = [], []
    for i, line in enumerate(lines1):
        a, b, c = line
        norm = np.hypot(a, b)
        if norm < 1e-8:
            continue
        dists = np.abs(pts1_h @ line) / norm
        best_j = np.argmin(dists)
        if dists[best_j] < distance_threshold:
            matched0.append(points0[i])
            matched1.append(points1[best_j])

    return np.array(matched0), np.array(matched1)


def epipolar_baseline(views, poses, distance_threshold=3.0):
    """Match candidate centerline points across the 2 views via the
    epipolar constraint, then triangulate to a 3D point cloud.

    Args:
        views: list of 2 2D centerline-point arrays (from
               extract_2d_centerline), each (N, 3) = (row, col, radius).
        poses: list of 2 (3,4) projection matrices (from Step 1.2).

    Returns:
        (K, 3) array of triangulated 3D points.
    """
    points0, points1 = views[0], views[1]
    P0, P1 = poses[0], poses[1]

    # NOTE: extract_2d_centerline returns (row, col, radius) -- row=y, col=x.
    # Flip to (x, y) = (col, row) to match projection-matrix pixel convention.
    points0_xy = points0[:, [1, 0, 2]]
    points1_xy = points1[:, [1, 0, 2]]

    matched0, matched1 = match_via_epipolar(
        points0_xy, points1_xy, P0, P1, distance_threshold)

    points_3d = np.array([
        triangulate_point(m0[:2], m1[:2], P0, P1)
        for m0, m1 in zip(matched0, matched1)
    ])
    return points_3d
