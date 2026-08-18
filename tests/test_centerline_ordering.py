from src.coronarycl.centerline import extract_centerline
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _synthetic_bifurcation_volume():
    vol = np.zeros((60, 60, 60), dtype=bool)

    def draw_tube(vol, p0, p1, r, n=400):
        for t in np.linspace(0, 1, n):
            c = p0 + t * (p1 - p0) + \
                np.array([6*np.sin(3*t), 4*np.cos(2*t), 0])
            x, y, z = c.astype(int)
            vol[max(0, x-r):x+r, max(0, y-r):y+r, max(0, z-r):z+r] = True
        return vol
    p0, pb = np.array([5., 30., 30.]), np.array([35., 30., 30.])
    p1, p2 = np.array([55., 15., 20.]), np.array([55., 45., 40.])
    vol = draw_tube(vol, p0, pb, 2)
    vol = draw_tube(vol, pb, p1, 2)
    vol = draw_tube(vol, pb, p2, 2)
    return vol


def test_reorder_preserves_all_points():
    vol = _synthetic_bifurcation_volume()
    out = extract_centerline(vol)
    raw_n = np.argwhere(__import__("skimage.morphology", fromlist=[
                        "skeletonize"]).skeletonize(vol)).shape[0]
    assert out.shape[0] == raw_n


def test_reorder_improves_locality():
    vol = _synthetic_bifurcation_volume()
    out = extract_centerline(vol)
    xyz = out[:, :3]
    jumps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    frac_close = np.mean(jumps <= np.sqrt(3))  # true 26-connected step
    assert frac_close > 0.9, f"expected >90% path-local steps, got {frac_close:.2f}"


if __name__ == "__main__":
    test_reorder_preserves_all_points()
    test_reorder_improves_locality()
    print("All tests passed.")
