"""Step 2.1 — deterministic epipolar-constraint baseline.
Prototype/small-scale on M4 locally (CPU); move full run to Colab if slow.

Used to get a floor Chamfer L2 number before the conditional diffusion
model (2.2).

TODO: implement epipolar point matching between the 2 views using the
saved projection matrices from Step 1.2, then triangulate matched
points to 3D.
"""

import numpy as np


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
