"""Runs locally on M4 -- no GPU needed."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coronarycl.metrics import chamfer_l2, overlap_metric


def test_chamfer_l2_zero_for_identical_sets():
    pts = np.random.randn(50, 3)
    assert chamfer_l2(pts, pts) == 0.0


def test_overlap_metric_full_for_identical_sets():
    pts = np.random.randn(50, 3)
    assert overlap_metric(pts, pts, d=0.01) == 1.0


def test_overlap_metric_decreases_with_noise():
    pts = np.random.randn(50, 3)
    noisy = pts + np.random.randn(50, 3) * 5.0
    assert overlap_metric(noisy, pts, d=0.1) < overlap_metric(pts, pts, d=0.1)


if __name__ == "__main__":
    test_chamfer_l2_zero_for_identical_sets()
    test_overlap_metric_full_for_identical_sets()
    test_overlap_metric_decreases_with_noise()
    print("All tests passed.")
