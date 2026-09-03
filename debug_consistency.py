"""Diagnose the Dataset v2 projection-consistency failure.

Runs four independent tests and tells you which hypothesis is correct.

  T1  3D containment: does the GT centerline lie inside the GT 3D vessel mask?
      (no projection involved -- isolates upstream centerline/mask mismatch)
  T2  DLT self-check: RMS reprojection error of the calibration markers,
      argmax localisation vs intensity-centroid localisation.
  T3  Ground-truth projection: render a centerline-ONLY volume through TIGRE
      with the same geometry, then measure the distance from P-predicted
      pixels to the actually-rendered centerline pixels. This isolates P
      from every other convention.
  T4  Brute force: all 48 axis-permutation x sign combinations of the mm
      coordinates, reporting which (if any) yields high consistency.

Usage:
    python debug_consistency.py --raw_dir DIR --centerline_dir DIR --n 3
"""
import argparse, glob, itertools, os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, center_of_mass
from scipy.spatial import cKDTree
import tigre

DET_N = 512


def build_geo(shape, spacing, view):
    geo = tigre.geometry(mode="cone", nVoxel=np.array(shape), default=True)
    geo.dVoxel = np.array(spacing, dtype=np.float32)
    geo.sVoxel = geo.dVoxel * geo.nVoxel
    geo.DSO, geo.DSD = view["DSO"], view["DSD"]
    geo.nDetector = np.array([DET_N, DET_N])
    geo.dDetector = np.array([view["det_spacing"]] * 2, dtype=np.float32)
    geo.sDetector = geo.dDetector * geo.nDetector
    geo.offOrigin = np.array(view["offOrigin"], dtype=np.float32)
    return geo


def _blob_centre(proj, mode):
    """Locate the projected marker. 'argmax' = original (biased on a plateau);
    'centroid' = intensity-weighted centre of the blob (sub-pixel)."""
    if mode == "argmax":
        r, c = np.unravel_index(np.argmax(proj), proj.shape)
        return float(c), float(r)
    thr = 0.5 * proj.max()
    m = proj >= thr
    if m.sum() == 0:
        r, c = np.unravel_index(np.argmax(proj), proj.shape)
        return float(c), float(r)
    r, c = center_of_mass(proj * m)
    return float(c), float(r)


def calibrate_P(shape, spacing, view, mode="centroid", n_points=20, cal_n=64,
                seed=0, return_err=False):
    sVoxel = np.array(shape) * np.array(spacing)
    cal_shape = (cal_n, cal_n, cal_n)
    cal_spacing = sVoxel / cal_n
    geo = build_geo(cal_shape, cal_spacing, view)
    angles = np.array([[np.radians(view["alpha"]), np.radians(view["beta"]), 0.0]], np.float32)

    rng = np.random.default_rng(seed)
    centre = np.array(cal_shape) / 2.0
    corr = []
    for _ in range(n_points):
        idx = rng.integers(6, np.array(cal_shape) - 6)
        vol = np.zeros(cal_shape, dtype=np.float32)
        vol[tuple(idx)] = 1.0
        proj = tigre.Ax(vol, geo, angles)[0]
        if proj.max() <= 0:
            continue
        x, y = _blob_centre(proj, mode)
        corr.append(((idx + 0.5 - centre) * cal_spacing, x, y))

    A = []
    for (X, x, y) in corr:
        Xh = np.array([*X, 1.0])
        A.append(np.concatenate([Xh, np.zeros(4), -x * Xh]))
        A.append(np.concatenate([np.zeros(4), Xh, -y * Xh]))
    _, _, Vt = np.linalg.svd(np.array(A))
    P = (Vt[-1].reshape(3, 4) / Vt[-1].reshape(3, 4)[-1, -1]).astype(np.float64)

    if not return_err:
        return P
    pts = np.array([c[0] for c in corr])
    obs = np.array([[c[1], c[2]] for c in corr])
    pred = project_points(P, pts)
    rms = float(np.sqrt(((pred - obs) ** 2).sum(1).mean()))
    return P, rms


def project_points(P, pts_mm):
    h = np.hstack([pts_mm, np.ones((len(pts_mm), 1))])
    uvw = (P @ h.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--centerline_dir", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    ids = sorted(int(os.path.basename(f).split(".")[0])
                 for f in glob.glob(os.path.join(args.raw_dir, "*.img.nii.gz")))
    view = dict(alpha=30.0, beta=0.0, DSD=1000.0, DSO=760.0,
                det_spacing=0.2779, offOrigin=np.zeros(3, np.float32))

    done = 0
    for cid in ids:
        if done >= args.n:
            break
        lp = os.path.join(args.raw_dir, f"{cid}.label.nii.gz")
        cp = os.path.join(args.centerline_dir, f"{cid}_centerline.npy")
        if not (os.path.exists(lp) and os.path.exists(cp)):
            continue
        nii = nib.load(lp)
        mask = (np.asarray(nii.get_fdata()) > 0.5)
        spacing = np.array(nii.header.get_zooms()[:3], float)
        shape = mask.shape
        cl = np.load(cp)
        vox = cl[:, :3]
        print(f"\n########## case {cid}  shape={shape} spacing={np.round(spacing,3)} ##########")

        # ---------------- T1: 3D containment ----------------
        vi = np.round(vox).astype(int)
        ok = ((vi >= 0) & (vi < np.array(shape))).all(1)
        m3 = binary_dilation(mask, iterations=1)
        inside = np.zeros(len(vi), bool)
        inside[ok] = m3[vi[ok, 0], vi[ok, 1], vi[ok, 2]]
        print(f"T1 3D containment (centerline inside dilated mask): {inside.mean():.3f}"
              f"   [in-bounds {ok.mean():.3f}]")
        if inside.mean() < 0.9:
            print("   -> centerline and mask DISAGREE in 3D. Projection is not the problem.")
            print("      Check axis order / orientation used when centerlines were built.")

        # ---------------- T2: DLT self-check ----------------
        P_arg, rms_arg = calibrate_P(shape, spacing, view, "argmax", return_err=True)
        P_cen, rms_cen = calibrate_P(shape, spacing, view, "centroid", return_err=True)
        print(f"T2 DLT reprojection RMS:  argmax {rms_arg:.2f}px   centroid {rms_cen:.2f}px")

        # ---------------- T3: TIGRE-rendered centerline vs P ----------------
        geo = build_geo(shape, spacing, view)
        angles = np.array([[np.radians(view["alpha"]), np.radians(view["beta"]), 0.0]], np.float32)
        clvol = np.zeros(shape, dtype=np.float32)
        clvol[vi[ok, 0], vi[ok, 1], vi[ok, 2]] = 1.0
        rendered = tigre.Ax(clvol, geo, angles)[0] > 1e-6
        true_px = np.argwhere(rendered)          # (row, col)
        if len(true_px) == 0:
            print("T3 rendered centerline is EMPTY — geometry/volume mismatch."); done += 1; continue
        tree = cKDTree(true_px[:, ::-1])          # -> (col, row)
        mm = (vox + 0.5 - np.array(shape) / 2.0) * spacing
        for name, P in (("argmax", P_arg), ("centroid", P_cen)):
            uv = project_points(P, mm)
            d, _ = tree.query(uv)
            print(f"T3 P[{name:8s}] median dist to rendered centerline: {np.median(d):7.2f} px"
                  f"   (90th {np.percentile(d,90):.1f})")

        # ---------------- T4: brute-force axis/sign conventions ----------------
        best = []
        maskproj = tigre.Ax(mask.astype(np.float32), geo, angles)[0] > 1e-6
        tolmask = binary_dilation(maskproj, iterations=2)
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product([1, -1], repeat=3):
                cand = (vox + 0.5 - np.array(shape) / 2.0) * spacing
                cand = cand[:, perm] * np.array(signs)
                uv = project_points(P_cen, cand)
                c = np.round(uv[:, 0]).astype(int); r = np.round(uv[:, 1]).astype(int)
                ib = (c >= 0) & (c < DET_N) & (r >= 0) & (r < DET_N)
                hit = np.zeros(len(uv), bool)
                hit[ib] = tolmask[r[ib], c[ib]]
                best.append((hit.mean(), perm, signs))
        best.sort(reverse=True)
        print("T4 top-3 coordinate conventions (consistency, axis-perm, signs):")
        for s, p, g in best[:3]:
            print(f"     {s:.3f}   perm={p}  signs={g}")
        print(f"     identity (0,1,2)/(+,+,+) = "
              f"{[s for s,p,g in best if p==(0,1,2) and g==(1,1,1)][0]:.3f}")
        done += 1


if __name__ == "__main__":
    main()
