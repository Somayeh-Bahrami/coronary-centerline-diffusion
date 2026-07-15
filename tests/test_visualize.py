"""Runs locally on M4 -- no GPU needed."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coronarycl.visualize import sweep_tube


def test_sweep_tube_shapes():
    t = np.linspace(0, 4 * np.pi, 100)
    centerline = np.stack([
        np.cos(t) * 10, np.sin(t) * 10, t * 2,
        1.5 + 0.5 * np.sin(t * 3),
    ], axis=1)
    verts, faces = sweep_tube(centerline, n_circle_pts=16)
    assert verts.shape == (100 * 16, 3)
    assert faces.shape[1] == 3


if __name__ == "__main__":
    test_sweep_tube_shapes()
    print("All tests passed.")
