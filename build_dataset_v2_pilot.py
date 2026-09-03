"""Dataset v2 PILOT (Plan A) — 10 cases + hard projection-consistency gate.

Pipeline per case:
    3D vessel mask (ImageCAS .label.nii.gz)
      -> TIGRE cone-beam projection, DeepCA-style geometry
      -> binarize
      -> store {2D binary projections, poses, 3D centerline GT (mm), radius, topology}

Gate (Step 9): project the 3D GT centerline with each calibrated pose and
verify it lands on the corresponding 2D vessel mask. If the mean consistency
is below --min_consistency the script REFUSES to declare success.

Patient-level split is computed once over ALL case ids BEFORE any projection,
so every sample carries its split label and no patient can straddle splits.

Requires CUDA (TIGRE). Run on Kaggle, same env as the original drr.py.

Usage:
    python build_dataset_v2_pilot.py \
        --raw_dir /path/to/imagecas_raw \
        --centerline_dir /path/to/centerlines \
        --out_dir ./dataset_v2_pilot \
        --n 10
"""
import argparse, glob, json, os, time
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
import tigre

# ---------------- DeepCA geometry (Wang et al., WACV 2025, Table 3) ----------------
DET_N = 512
DET_SPACING_RANGE = (0.2769, 0.2789)
V1_DSD_RANGE, V2_DSD_RANGE = (970.0, 1010.0), (1050.0, 1070.0)
V1_DSO_RANGE, V2_DSO_JITTER = (745.0, 785.0), 3.0
V1_PRIMARY, V1_SECONDARY = (18.0, 42.0), (-8.0, 8.0)
V2_PRIMARY, V2_SECONDARY = (-8.0, 8.0), (18.0, 42.0)
MOTION_ROT_DEG, MOTION_TRANS_MM = 10.0, 8.0


def sample_geometry(rng):
    """One DeepCA-style two-view acquisition. View 2 also gets the rigid
    motion perturbation (DeepCA protocol)."""
    det_sp = rng.uniform(*DET_SPACING_RANGE)
    dso1 = rng.uniform(*V1_DSO_RANGE)
    rot = rng.uniform(-MOTION_ROT_DEG, MOTION_ROT_DEG, size=3)
    trans = rng.uniform(-MOTION_TRANS_MM, MOTION_TRANS_MM, size=3)
    v1 = dict(alpha=rng.uniform(*V1_PRIMARY), beta=rng.uniform(*V1_SECONDARY),
              DSD=rng.uniform(*V1_DSD_RANGE), DSO=dso1,
              det_spacing=det_sp, offOrigin=np.zeros(3, np.float32))
    v2 = dict(alpha=rng.uniform(*V2_PRIMARY) + rot[0],
              beta=rng.uniform(*V2_SECONDARY) + rot[1],
              DSD=rng.uniform(*V2_DSD_RANGE),
              DSO=dso1 + rng.uniform(-V2_DSO_JITTER, V2_DSO_JITTER),
              det_spacing=det_sp, offOrigin=np.array(trans, np.float32))
    return [v1, v2], dict(motion_rot_deg=rot.tolist(), motion_trans_mm=trans.tolist())


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


def calibrate_P(shape, spacing, view, n_points=12, cal_n=64, seed=0):
    """DLT calibration against TIGRE's real Ax(), as in the original pipeline.
    IMPORTANT: the calibration phantom is built with the SAME physical extent
    (sVoxel) as the real volume, so P transfers exactly; only the sampling
    resolution is coarser (cheap). World coords = mm relative to volume centre.
    """
    sVoxel = np.array(shape) * np.array(spacing)
    cal_shape = (cal_n, cal_n, cal_n)
    cal_spacing = sVoxel / cal_n
    geo = build_geo(cal_shape, cal_spacing, view)
    angles = np.array([[np.radians(view["alpha"]), np.radians(view["beta"]), 0.0]], np.float32)

    rng = np.random.default_rng(seed)
    centre = np.array(cal_shape) / 2.0
    corr = []
    for _ in range(n_points):
        idx = rng.integers(8, np.array(cal_shape) - 8)
        vol = np.zeros(cal_shape, dtype=np.float32)
        vol[tuple(idx)] = 1.0
        proj = tigre.Ax(vol, geo, angles)[0]
        row, col = np.unravel_index(np.argmax(proj), proj.shape)
        corr.append(((idx - centre) * cal_spacing, col, row))

    A = []
    for (X, x, y) in corr:
        Xh = np.array([*X, 1.0])
        A.append(np.concatenate([Xh, np.zeros(4), -x * Xh]))
        A.append(np.concatenate([np.zeros(4), Xh, -y * Xh]))
    _, _, Vt = np.linalg.svd(np.array(A))
    P = Vt[-1].reshape(3, 4)
    return (P / P[-1, -1]).astype(np.float32)


def project_points(P, pts_mm):
    """mm (relative to volume centre) -> (col, row) pixel coords."""
    h = np.hstack([pts_mm, np.ones((len(pts_mm), 1))])
    uvw = (P @ h.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--centerline_dir", required=True)
    ap.add_argument("--out_dir", default="./dataset_v2_pilot")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tol_px", type=float, default=2.0,
                    help="pixel tolerance (mask dilation) for the consistency gate")
    ap.add_argument("--min_consistency", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- patient-level split over ALL ids, BEFORE any projection ----
    all_ids = sorted(int(os.path.basename(f).split(".")[0])
                     for f in glob.glob(os.path.join(args.raw_dir, "*.img.nii.gz")))
    rs = np.random.default_rng(args.seed).permutation(len(all_ids))
    shuffled = [all_ids[i] for i in rs]
    n_tr = int(0.8 * len(shuffled)); n_va = int(0.1 * len(shuffled))
    split_of = {}
    for i, cid in enumerate(shuffled):
        split_of[cid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    json.dump({"train": [c for c in shuffled if split_of[c] == "train"],
               "val":   [c for c in shuffled if split_of[c] == "val"],
               "test":  [c for c in shuffled if split_of[c] == "test"]},
              open(os.path.join(args.out_dir, "case_splits_v2.json"), "w"), indent=2)
    print(f"split over {len(all_ids)} patients -> "
          f"{sum(v=='train' for v in split_of.values())}/"
          f"{sum(v=='val' for v in split_of.values())}/"
          f"{sum(v=='test' for v in split_of.values())}\n")

    rng = np.random.default_rng(args.seed)
    rows, times = [], []
    done = 0
    for cid in all_ids:
        if done >= args.n:
            break
        label_path = os.path.join(args.raw_dir, f"{cid}.label.nii.gz")
        cl_path = os.path.join(args.centerline_dir, f"{cid}_centerline.npy")
        if not (os.path.exists(label_path) and os.path.exists(cl_path)):
            continue
        t0 = time.time()

        nii = nib.load(label_path)
        mask = (np.asarray(nii.get_fdata()) > 0.5).astype(np.float32)
        spacing = np.array(nii.header.get_zooms()[:3], dtype=np.float64)
        shape = mask.shape
        if mask.sum() < 100:
            print(f"{cid}: SKIP (empty mask)"); continue

        # centerline: voxel index -> mm relative to volume centre (same convention as P)
        cl = np.load(cl_path)
        cl_vox, radius_vox, topo = cl[:, :3], cl[:, 3], cl[:, 4]
        cl_mm = (cl_vox - np.array(shape) / 2.0) * spacing
        radius_mm = radius_vox * float(np.mean(spacing))  # EDT was computed in voxels

        views, motion = sample_geometry(rng)
        images, poses, cons = [], [], []
        for k, view in enumerate(views):
            geo = build_geo(shape, spacing, view)
            angles = np.array([[np.radians(view["alpha"]), np.radians(view["beta"]), 0.0]], np.float32)
            proj = tigre.Ax(mask, geo, angles)[0]
            binary = (proj > 1e-6)
            P = calibrate_P(shape, spacing, view, seed=args.seed + k)
            images.append(binary); poses.append(P)

            # ---- Step 9 gate: GT centerline must project onto the vessel mask ----
            uv = project_points(P, cl_mm)
            col = np.round(uv[:, 0]).astype(int); row = np.round(uv[:, 1]).astype(int)
            inb = (col >= 0) & (col < DET_N) & (row >= 0) & (row < DET_N)
            tol = binary_dilation(binary, iterations=int(args.tol_px)) if args.tol_px > 0 else binary
            hit = np.zeros(len(uv), bool)
            hit[inb] = tol[row[inb], col[inb]]
            cons.append(float(hit.mean()))

        dt = time.time() - t0; times.append(dt)
        rows.append(dict(case=cid, split=split_of[cid], consistency=cons,
                         n_points=int(len(cl_mm)), secs=round(dt, 1)))
        np.savez_compressed(
            os.path.join(args.out_dir, f"{cid}.npz"),
            images=np.stack(images).astype(np.uint8),
            poses=np.stack(poses).astype(np.float32),
            centerline_mm=cl_mm.astype(np.float32),
            radius_mm=radius_mm.astype(np.float32),
            topology=topo.astype(np.uint8),
            volume_shape=np.array(shape), spacing=spacing.astype(np.float32),
            split=split_of[cid],
            geometry=json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                  for k, v in vw.items()} for vw in views]),
            motion=json.dumps(motion),
        )
        print(f"{cid} [{split_of[cid]}]: consistency v1 {cons[0]:.3f} | v2 {cons[1]:.3f} "
              f"| {len(cl_mm)} pts | {dt:.1f}s", flush=True)
        done += 1

    if not rows:
        print("No cases processed — check --raw_dir / --centerline_dir."); return

    allc = np.array([c for r in rows for c in r["consistency"]])
    mean_t = float(np.mean(times))
    print("\n" + "=" * 64)
    print(f"cases {len(rows)} | consistency mean {allc.mean():.3f} "
          f"(min {allc.min():.3f}, max {allc.max():.3f})   [tol {args.tol_px}px]")
    print(f"per-case {mean_t:.1f}s  ->  ETA for 1000 cases: {mean_t*1000/3600:.1f} h")
    if allc.mean() >= args.min_consistency:
        print(f"GATE PASSED (>= {args.min_consistency}) — geometry/poses/centerline agree.")
    else:
        print(f"GATE FAILED (< {args.min_consistency}) — DO NOT TRAIN ON THIS DATASET.")
        print("  Likely causes: axis-order mismatch, wrong centre convention,")
        print("  offOrigin sign, or P not transferring from the calibration phantom.")
    print("=" * 64)
    json.dump(rows, open(os.path.join(args.out_dir, "pilot_report.json"), "w"), indent=2)
    print("saved pilot_report.json + case_splits_v2.json + per-case npz")


if __name__ == "__main__":
    main()
