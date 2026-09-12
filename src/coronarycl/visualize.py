"""Step 3.2 — qualitative visualization for clinical review. Runs fully
on M4 locally — simple deterministic geometric operation, no GPU / no
learned model.

Generates a tube surface from a predicted centerline + radius via a
geometric sweep (a circle of the local radius swept along the 3D
curve). This is how stenosis/shape/foreshortening get shown to
Prof. Bo Zhu -- NOT a learned mesh-generation model, so it doesn't
reintroduce the mesh-generation pipeline dropped from the first draft.

Grounded in how minimal lumen diameter / % diameter stenosis are
already computed clinically from a centerline + radius profile
(Quantitative Coronary Angiography).

TODO: swap the manual sweep below for VMTK's centerline-to-surface
utilities if higher-fidelity rendering is needed later.
"""

import numpy as np


def edge_segments(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Convert indexed graph edges to ``(E,2,3)`` line segments."""
    points = np.asarray(points, dtype=float)
    edges = np.asarray(edges, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N,3+)")
    if edges.ndim != 2 or edges.shape[1:] != (2,):
        raise ValueError("edges must have shape (E,2)")
    if len(edges) and (edges.min() < 0 or edges.max() >= len(points)):
        raise ValueError("edge index outside point array")
    return points[edges, :3]


def set_equal_crop_axes(axis, lower_mm, upper_mm):
    """Apply identical numeric xyz span and a cubic 3-D box aspect."""
    lower = np.asarray(lower_mm, dtype=float).reshape(3)
    upper = np.asarray(upper_mm, dtype=float).reshape(3)
    if np.any(lower >= upper):
        raise ValueError("invalid plot bounds")
    centre = (lower + upper) / 2.0
    half_span = float(np.max(upper - lower) / 2.0)
    axis.set_xlim(centre[0] - half_span, centre[0] + half_span)
    axis.set_ylim(centre[1] - half_span, centre[1] + half_span)
    axis.set_zlim(centre[2] - half_span, centre[2] + half_span)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")


def plot_given_topology_comparison(
    gt_mm, pred_mm, edges, lower_mm, upper_mm, *, sample="", seed=None,
):
    """Plot GT and prediction with the same given/oracle GT tree edges.

    This deliberately does not connect consecutive DFS rows. It returns a
    Matplotlib figure so callers decide where and how to save it.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    gt = np.asarray(gt_mm, dtype=float)[:, :3]
    pred = np.asarray(pred_mm, dtype=float)[:, :3]
    if gt.shape != pred.shape:
        raise ValueError("GT and prediction must have matching node arrays")
    gt_segments = edge_segments(gt, edges)
    pred_segments = edge_segments(pred, edges)
    lower = np.asarray(lower_mm, dtype=float).reshape(3)
    upper = np.asarray(upper_mm, dtype=float).reshape(3)
    outside = np.any(
        (pred < lower - 1e-4) | (pred > upper + 1e-4), axis=1)

    figure = plt.figure(figsize=(13, 6), constrained_layout=True)
    titles = ("Ground truth", "Prediction — given/oracle GT topology")
    for position, (points, segments, title) in enumerate(
        ((gt, gt_segments, titles[0]), (pred, pred_segments, titles[1])), 1
    ):
        axis = figure.add_subplot(1, 2, position, projection="3d")
        axis.add_collection3d(Line3DCollection(
            segments, colors="#1565c0" if position == 1 else "#ef6c00",
            linewidths=0.7, alpha=0.75))
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=2.2, alpha=0.55,
            color="#0d47a1" if position == 1 else "#e65100")
        if position == 2 and outside.any():
            axis.scatter(
                pred[outside, 0], pred[outside, 1], pred[outside, 2],
                s=12, color="crimson", label="outside crop")
            axis.legend(loc="upper right")
        set_equal_crop_axes(axis, lower, upper)
        axis.set_title(title)
    suffix = f" | seed {seed}" if seed is not None else ""
    figure.suptitle(f"{sample}{suffix}")
    return figure


def sweep_tube(centerline: np.ndarray, n_circle_pts: int = 16):
    """Sweep a circle of local radius along the centerline curve to
    produce a tube surface mesh (vertices + faces).

    ``centerline`` must be one true polyline. Do not pass a branched skeleton
    stored in DFS order; branch backtracking would create false tube segments.

    Args:
        centerline: (N, 4) array of (x, y, z, radius).
        n_circle_pts: points per circular cross-section.

    Returns:
        vertices: (N * n_circle_pts, 3)
        faces: (F, 3) triangle indices
    """
    points = centerline[:, :3]
    radii = centerline[:, 3]
    n = len(points)

    tangents = np.gradient(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8

    up = np.array([0.0, 0.0, 1.0])
    normals = np.cross(tangents, up)
    norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = np.array([1.0, 0.0, 0.0])
    normals = np.where(norm_lens < 1e-6, fallback, normals / (norm_lens + 1e-8))
    binormals = np.cross(tangents, normals)

    theta = np.linspace(0, 2 * np.pi, n_circle_pts, endpoint=False)
    vertices = np.zeros((n, n_circle_pts, 3))
    for i in range(n):
        circle = (np.outer(np.cos(theta), normals[i]) +
                  np.outer(np.sin(theta), binormals[i])) * radii[i]
        vertices[i] = points[i] + circle
    vertices = vertices.reshape(-1, 3)

    faces = []
    for i in range(n - 1):
        for j in range(n_circle_pts):
            j_next = (j + 1) % n_circle_pts
            a = i * n_circle_pts + j
            b = i * n_circle_pts + j_next
            c = (i + 1) * n_circle_pts + j
            d = (i + 1) * n_circle_pts + j_next
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces = np.array(faces)

    return vertices, faces


def save_obj(vertices, faces, out_path):
    with open(out_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
