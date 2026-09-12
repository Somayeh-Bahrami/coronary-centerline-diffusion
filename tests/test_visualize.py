"""Runs locally on M4 -- no GPU needed."""

import numpy as np
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coronarycl.visualize import (
    edge_segments,
    plot_given_topology_comparison,
    sweep_tube,
)


def test_sweep_tube_shapes():
    t = np.linspace(0, 4 * np.pi, 100)
    centerline = np.stack([
        np.cos(t) * 10, np.sin(t) * 10, t * 2,
        1.5 + 0.5 * np.sin(t * 3),
    ], axis=1)
    verts, faces = sweep_tube(centerline, n_circle_pts=16)
    assert verts.shape == (100 * 16, 3)
    assert faces.shape[1] == 3


def test_graph_plot_uses_only_supplied_edges_and_equal_axes():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    edges = np.array([[0, 1], [1, 2]])
    segments = edge_segments(points, edges)
    assert segments.shape == (2, 2, 3)
    figure = plot_given_topology_comparison(
        points, points, edges, [-2, -1, -3], [2, 1, 3], sample="unit")
    for axis in figure.axes:
        spans = np.array([
            np.diff(axis.get_xlim())[0],
            np.diff(axis.get_ylim())[0],
            np.diff(axis.get_zlim())[0],
        ])
        np.testing.assert_allclose(spans, spans[0])
        np.testing.assert_allclose(axis.get_box_aspect(), axis.get_box_aspect()[0])
    import matplotlib.pyplot as plt
    plt.close(figure)


if __name__ == "__main__":
    test_sweep_tube_shapes()
    print("All tests passed.")
