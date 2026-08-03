"""Step 2.1 — deterministic epipolar-constraint baseline.
Prototype/small-scale on M4 locally (CPU); move full run to Kaggle if slow.

Used to get a floor Chamfer L2 number before the conditional diffusion
model (2.2).

Pipeline: extract 2D centerline from each view's vessel mask (already
segmented in Step 1.2, no learned model needed) -> match points across
views via epipolar constraint + radius-similarity disambiguation ->
triangulate to 3D.

Debugging note: initial matching (epipolar distance only) produced a
100% match rate and severe depth-axis (Z) scatter in the reconstructed
3D points, despite X/Y looking reasonable -- diagnosed as ambiguous
correspondence along the vessel's own curving path, not a
coordinate-frame or calibration bug (ruled out by tightening the
distance threshold, which produced identical results). Adding radius
similarity as a second disambiguating signal did not meaningfully
improve results (match count and Chamfer L2 both stayed essentially
unchanged), suggesting the limitation may be structural (independent
per-view projection-matrix calibration, or residual 2D extraction
noise near crossings) rather than fixable via matching strategy alone.
Documented as an open limitation; result accepted as the classical
baseline floor.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def extract_2d_centerline(mask_2d: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Skeletonize one 2D vessel-mask projection and attach a radius
    (in pixels) at each centerline point via the distance transform.
    Mirrors Step 1.1's 3D skeletonization, applied to a 2D image.

    Returns:
        (N, 3) array of (row, col, radius) in pixel coordinates.
    """
    binary_mask = mask_2d > (mask_2d.max() * threshold)
    skeleton = skeletonize(binary_mask)
    dist = distance_transform_edt(binary_mask)

    coords = np.argwhere(skeleton)
    radii = dist[skeleton]
    return np.concatenate([coords, radii[:, None]], axis=1)


def compute_camera_center(P: np.ndarray) -> np.ndarray:
    """Camera center C is the null space of the 3x4 projection matrix P."""
    _, _, Vt = np.linalg.svd(P)
    C_h = Vt[-1]
    return C_h[:3] / C_h[3]


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric cross-product matrix [v]_x."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def compute_fundamental_matrix(P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
    """Fundamental matrix F relating view 0 to view 1, computed directly
    from the two projection matrices (Hartley & Zisserman, "Multiple
    View Geometry in Computer Vision", 2004). F = [e1]_x @ P1 @ pinv(P0).
    """
    P0 = P0.astype(np.float64)
    P1 = P1.astype(np.float64)

    C0 = compute_camera_center(P0)
    e1_h = P1 @ np.append(C0, 1.0)
    e1_h = e1_h / e1_h[2]

    return skew_symmetric(e1_h) @ P1 @ np.linalg.pinv(P0)


def match_via_epipolar(points0: np.ndarray, points1: np.ndarray,
                       P0: np.ndarray, P1: np.ndarray,
                       distance_threshold: float = 3.0,
                       radius_weight: float = 2.0):
    """For each point in view 0, find its best-matching point in view 1
    among candidates near the epipolar line, disambiguated by radius
    similarity.

    Args:
        points0: (N, 3) array -- (x, y, radius) in view 0.
        points1: (M, 3) array -- (x, y, radius) in view 1.
        P0, P1: (3, 4) projection matrices.
        distance_threshold: max allowed point-to-epipolar-line distance
            (pixels) for a candidate match to be considered.
        radius_weight: weight trading off epipolar-line distance vs.
            radius similarity. Tunable hyperparameter, not derived from
            a paper.

    Returns:
        matched0, matched1: corresponding matched points from each view.
    """
    F = compute_fundamental_matrix(P0, P1)

    pts0_xy = points0[:, :2].astype(np.float64)
    pts1_xy = points1[:, :2].astype(np.float64)
    r0 = points0[:, 2] if points0.shape[1] > 2 else None
    r1 = points1[:, 2] if points1.shape[1] > 2 else None

    pts0_h = np.hstack([pts0_xy, np.ones((len(pts0_xy), 1))])
    pts1_h = np.hstack([pts1_xy, np.ones((len(pts1_xy), 1))])

    with np.errstate(all='ignore'):  # known Apple Silicon Accelerate false-positive warning
        lines1 = pts0_h @ F.T
    assert np.isfinite(lines1).all(
    ), "lines1 contains non-finite values -- real bug, investigate"

    matched0, matched1 = [], []
    for i, line in enumerate(lines1):
        a, b, c = line
        norm = np.hypot(a, b)
        if norm < 1e-8:
            continue
        with np.errstate(all='ignore'):
            dists = np.abs(pts1_h @ line) / norm
        assert np.isfinite(dists).all(
        ), "dists contains non-finite values -- real bug, investigate"

        candidates = np.where(dists < distance_threshold)[0]
        if len(candidates) == 0:
            continue

        if r0 is not None and r1 is not None:
            radius_diff = np.abs(r1[candidates] - r0[i])
            score = dists[candidates] + radius_weight * radius_diff
            best_j = candidates[np.argmin(score)]
        else:
            best_j = candidates[np.argmin(dists[candidates])]

        matched0.append(points0[i])
        matched1.append(points1[best_j])

    return np.array(matched0), np.array(matched1)


def triangulate_point(pt_view0, pt_view1, pose0, pose1):
    """Two-view DLT triangulation (Hartley & Zisserman, 2004) -- given
    a matched point pair and each view's projection matrix, solves for
    the 3D point that projects to both observations.
    """
    pose0 = pose0.astype(np.float64)
    pose1 = pose1.astype(np.float64)
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


def epipolar_baseline(views, poses, distance_threshold=3.0, radius_weight=2.0):
    """Full baseline pipeline: match centerline points across the 2
    views via the epipolar constraint + radius similarity, then
    triangulate to 3D.

    Args:
        views: list of 2 2D centerline-point arrays (from
               extract_2d_centerline), each (N, 3) = (row, col, radius).
        poses: list of 2 (3,4) projection matrices (from Step 1.2).

    Returns:
        (K, 3) array of triangulated 3D points.
    """
    points0, points1 = views[0], views[1]
    P0, P1 = poses[0], poses[1]

    # extract_2d_centerline returns (row, col, radius) -- row=y, col=x.
    # Flip to (x, y, radius) to match projection-matrix pixel convention.
    points0_xy = points0[:, [1, 0, 2]]
    points1_xy = points1[:, [1, 0, 2]]

    matched0, matched1 = match_via_epipolar(
        points0_xy, points1_xy, P0, P1, distance_threshold, radius_weight
    )

    points_3d = np.array([
        triangulate_point(m0[:2], m1[:2], P0, P1)
        for m0, m1 in zip(matched0, matched1)
    ])
    return points_3d
