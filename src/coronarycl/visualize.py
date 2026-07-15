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


def sweep_tube(centerline: np.ndarray, n_circle_pts: int = 16):
    """Sweep a circle of local radius along the centerline curve to
    produce a tube surface mesh (vertices + faces).

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
