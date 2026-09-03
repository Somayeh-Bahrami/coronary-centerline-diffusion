"""Dataset v2 PILOT v2 (Plan A) — fixes the three causes the diagnostic found.

Changes vs v1:
  FIX 1  DLT marker localisation: intensity centroid, not np.argmax.
         (argmax on a flat-topped blob is biased to its top-left corner:
          measured RMS 2.7-3.4 px -> 0.14-1.04 px)
  FIX 2  Split RCA / LCA by 3D connected component, one sample per vessel.
         The combined tree spans ~110 mm and cannot fit the DeepCA detector
         FOV (99-115 mm at isocentre); a single coronary system can.
  FIX 3  Crop to a ~96 mm cube centred on the vessel, matching DeepCA's
         90-105 mm reconstruction volume, so nothing projects off-detector.

Per sample:
    vessel mask (one coronary system)
      -> crop to CROP_MM cube about its centroid
      -> TIGRE cone-beam projection, DeepCA geometry
      -> binarize
      -> {2D binary projections, poses, 3D centerline (mm), radius, topology}

Patient-level split is computed over all patient ids BEFORE anything else, and
both vessels of a patient inherit the same split (no leakage).

Requires CUDA (TIGRE).

Usage:
    python build_dataset_v2_pilot_v2.py \
        --raw_dir DIR --centerline_dir DIR --out_dir ./dataset_v2_pilot_v2 --n 10
"""
import argparse, glob, json, os, time
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, center_of_mass, label as cc_label
import tigre

# ---------------- DeepCA geometry (Wang et al., WACV 2025, Table 3) ----------------
DET_N = 512
DET_SPACING_RANGE = (0.2769, 0.2789)
V1_DSD_RANGE, V2_DSD_RANGE = (970.0, 1010.0), (1050.0, 1070.0)
V1_DSO_RANGE, V2_DSO_JITTER = (745.0, 785.0), 3.0
V1_PRIMARY, V1_SECONDARY = (18.0, 42.0), (-8.0, 8.0)
V2_PRIMARY, V2_SECONDARY = (-8.0, 8.0), (18.0, 42.0)
MOTION_ROT_DEG, MOTION_TRANS_MM = 10.0, 8.0
CROP_MM_DEFAULT = 96.0          # DeepCA reconstruction volume is 90-105 mm
MIN_COMPONENT_FRAC = 0.05       # ignore specks below 5% of the mask


def sample_geometry(rng):
    det_sp = rng.uniform(*DET_SPACING_RANGE)
    dso1 = rng.uniform(*V1_DSO_RANGE)
    rot = rng.uniform(-MOTION_ROT_DEG, MOTION_ROT_DEG, size=3)
    trans = rng.uniform(-MOTION_TRANS_MM, MOTION_TRANS_MM, size=3)
    v1 = dict(alpha=float(rng.uniform(*V1_PRIMARY)), beta=float(rng.uniform(*V1_SECONDARY)),
              DSD=float(rng.uniform(*V1_DSD_RANGE)), DSO=float(dso1),
              det_spacing=float(det_sp), offOrigin=np.zeros(3, np.float32))
    v2 = dict(alpha=float(rng.uniform(*V2_PRIMARY) + rot[0]),
              beta=float(rng.uniform(*V2_SECONDARY) + rot[1]),
              DSD=float(rng.uniform(*V2_DSD_RANGE)),
              DSO=float(dso1 + rng.uniform(-V2_DSO_JITTER, V2_DSO_JITTER)),
              det_spacing=float(det_sp), offOrigin=np.array(trans, np.float32))
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


def _blob_centre(proj):
    """FIX 1 — intensity-weighted centroid of the marker blob (sub-pixel).
    np.argmax returns the first index on a flat plateau, a systematic bias."""
    thr = 0.5 * proj.max()
    m = proj >= thr
    if m.sum() == 0:
        r, c = np.unravel_index(np.argmax(proj), proj.shape)
        return float(c), float(r)
    r, c = center_of_mass(proj * m)
    return float(c), float(r)


def calibrate_P(shape, spacing, view, n_points=20, cal_n=64, seed=0, return_err=False):
    """DLT against TIGRE's real Ax(). Calibration phantom has the SAME physical
    extent (sVoxel) as the volume being projected, so P transfers exactly.
    World coords = mm relative to the volume centre."""
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
        x, y = _blob_centre(proj)
        corr.append(((idx + 0.5 - centre) * cal_spacing, x, y))
    if len(corr) < 6:
        raise RuntimeError("DLT calibration failed: too few visible markers")

    A = []
    for (X, x, y) in corr:
        Xh = np.array([*X, 1.0])
        A.append(np.concatenate([Xh, np.zeros(4), -x * Xh]))
        A.append(np.concatenate([np.zeros(4), Xh, -y * Xh]))
    _, _, Vt = np.linalg.svd(np.array(A))
    P = Vt[-1].reshape(3, 4)
    P = (P / P[-1, -1]).astype(np.float64)
    if not return_err:
        return P
    pts = np.array([c[0] for c in corr]); obs = np.array([[c[1], c[2]] for c in corr])
    rms = float(np.sqrt(((project_points(P, pts) - obs) ** 2).sum(1).mean()))
    return P, rms


def project_points(P, pts_mm):
    h = np.hstack([pts_mm, np.ones((len(pts_mm), 1))])
    uvw = (P @ h.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def side_labels(affine, comp_centroids):
    """Name components by anatomical side using the NIfTI axis codes, rather
    than assuming an array orientation. Falls back to size rank."""
    try:
        codes = nib.aff2axcodes(affine)          # e.g. ('L','A','S') or ('R','A','S')
    except Exception:
        return None, None
    ax = next((i for i, c in enumerate(codes) if c in ("L", "R")), None)
    if ax is None:
        return None, None
    # increasing index along this axis moves toward codes[ax]
    toward = codes[ax]
    vals = [c[ax] for c in comp_centroids]
    order = np.argsort(vals)                      # ascending index
    names = [None] * len(vals)
    # the component further toward 'L' is the LEFT coronary system
    left_first = (toward == "L")
    ranked = order[::-1] if left_first else order
    for rank, ci in enumerate(ranked):
        names[ci] = "LCA" if rank == 0 else "RCA"
    return names, ax


def split_components(mask, centerline_vox, affine):
    """FIX 2 — one sample per coronary system (connected component)."""
    lab, n = cc_label(mask, structure=np.ones((3, 3, 3), int))
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = [i for i in range(1, n + 1) if sizes[i] >= MIN_COMPONENT_FRAC * mask.sum()]
    keep = sorted(keep, key=lambda i: -sizes[i])[:2]
    if not keep:
        return []

    # assign centerline points to components (dilate to catch boundary points)
    vi = np.clip(np.round(centerline_vox).astype(int), 0, np.array(mask.shape) - 1)
    pt_lab = lab[vi[:, 0], vi[:, 1], vi[:, 2]]
    if (pt_lab == 0).any():
        lab_d = lab.copy()
        for _ in range(2):
            grow = binary_dilation(lab_d > 0) & (lab_d == 0)
            if not grow.any():
                break
            from scipy.ndimage import grey_dilation
            lab_d = np.maximum(lab_d, grey_dilation(lab_d, size=(3, 3, 3)) * grow)
        pt_lab = lab_d[vi[:, 0], vi[:, 1], vi[:, 2]]

    centroids = [np.array(center_of_mass(lab == i)) for i in keep]
    names, _ = side_labels(affine, centroids)
    out = []
    for j, i in enumerate(keep):
        sel = pt_lab == i
        if sel.sum() < 50:
            continue
        out.append(dict(comp_id=int(i), name=(names[j] if names else f"c{j}"),
                        mask=(lab == i), point_sel=sel, n_vox=int(sizes[i])))
    return out


def crop_box(vox, shape, spacing, crop_mm):
    """FIX 3 — a crop_mm box covering the vessel.

    Centred on the centerline BOUNDING-BOX centre (not the centroid, which is
    pulled toward dense proximal regions and under-covers distal branches), and
    SHIFTED rather than clamped at volume edges so the box keeps its full size.
    """
    shape = np.array(shape)
    bb_lo, bb_hi = vox.min(0), vox.max(0)
    centre = (bb_lo + bb_hi) / 2.0
    half_vox = np.ceil((crop_mm / 2.0) / np.array(spacing)).astype(int)
    size = np.minimum(2 * half_vox, shape)          # only shrink if the volume is smaller
    lo = np.round(centre).astype(int) - size // 2
    lo = np.clip(lo, 0, np.maximum(shape - size, 0))  # shift, don't shrink
    return lo, lo + size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--centerline_dir", required=True)
    ap.add_argument("--out_dir", default="./dataset_v2_pilot_v2")
    ap.add_argument("--n", type=int, default=10, help="number of PATIENTS")
    ap.add_argument("--crop_mm", type=float, default=CROP_MM_DEFAULT)
    ap.add_argument("--tol_px", type=float, default=2.0)
    ap.add_argument("--min_consistency", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- patient-level split over ALL ids, before anything else ----
    all_ids = sorted(int(os.path.basename(f).split(".")[0])
                     for f in glob.glob(os.path.join(args.raw_dir, "*.img.nii.gz")))
    perm = np.random.default_rng(args.seed).permutation(len(all_ids))
    shuffled = [all_ids[i] for i in perm]
    n_tr, n_va = int(0.8 * len(shuffled)), int(0.1 * len(shuffled))
    split_of = {c: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, c in enumerate(shuffled)}
    json.dump({s: [c for c in shuffled if split_of[c] == s] for s in ("train", "val", "test")},
              open(os.path.join(args.out_dir, "case_splits_v2.json"), "w"), indent=2)
    print(f"patient split over {len(all_ids)}: "
          f"{sum(v=='train' for v in split_of.values())}/"
          f"{sum(v=='val' for v in split_of.values())}/"
          f"{sum(v=='test' for v in split_of.values())}  "
          f"(both vessels of a patient share a split)\n")

    rng = np.random.default_rng(args.seed)
    rows, times, fov_warn = [], [], 0
    done = 0
    for cid in all_ids:
        if done >= args.n:
            break
        lp = os.path.join(args.raw_dir, f"{cid}.label.nii.gz")
        cp = os.path.join(args.centerline_dir, f"{cid}_centerline.npy")
        if not (os.path.exists(lp) and os.path.exists(cp)):
            continue
        t0 = time.time()
        nii = nib.load(lp)
        mask_full = np.asarray(nii.get_fdata()) > 0.5
        spacing = np.array(nii.header.get_zooms()[:3], float)
        shape = mask_full.shape
        cl = np.load(cp)
        if mask_full.sum() < 100 or len(cl) < 50:
            continue

        comps = split_components(mask_full, cl[:, :3], nii.affine)
        if not comps:
            print(f"{cid}: SKIP (no usable component)"); continue

        for comp in comps:
            vox = cl[comp["point_sel"], :3]
            bbox_mm = (vox.max(0) - vox.min(0)) * spacing   # true vessel extent
            lo, hi = crop_box(vox, shape, spacing, args.crop_mm)
            sl = tuple(slice(l, h) for l, h in zip(lo, hi))
            cmask = comp["mask"][sl].astype(np.float32)
            cshape = np.array(cmask.shape)
            sVoxel = cshape * spacing
            if cmask.sum() < 50:
                continue

            # centerline -> mm relative to the CROP centre (matches calibration)
            in_crop = ((vox >= lo) & (vox < hi)).all(1)
            vox_c = vox[in_crop]
            cl_mm = (vox_c - lo + 0.5 - cshape / 2.0) * spacing
            radius_mm = cl[comp["point_sel"], 3][in_crop] * float(np.mean(spacing))
            topo = cl[comp["point_sel"], 4][in_crop]

            views, motion = sample_geometry(rng)
            images, poses, cons, rmss = [], [], [], []
            for k, view in enumerate(views):
                fov = (DET_N * view["det_spacing"]) / (view["DSD"] / view["DSO"])
                if sVoxel.max() > fov:
                    fov_warn += 1
                geo = build_geo(cshape, spacing, view)
                angles = np.array([[np.radians(view["alpha"]), np.radians(view["beta"]), 0.0]],
                                  np.float32)
                binary = tigre.Ax(cmask, geo, angles)[0] > 1e-6
                P, rms = calibrate_P(cshape, spacing, view, seed=args.seed + k, return_err=True)
                images.append(binary); poses.append(P); rmss.append(rms)

                uv = project_points(P, cl_mm)
                col = np.round(uv[:, 0]).astype(int); row = np.round(uv[:, 1]).astype(int)
                ib = (col >= 0) & (col < DET_N) & (row >= 0) & (row < DET_N)
                tol = binary_dilation(binary, iterations=int(args.tol_px)) if args.tol_px > 0 else binary
                hit = np.zeros(len(uv), bool)
                hit[ib] = tol[row[ib], col[ib]]
                cons.append(float(hit.mean()))

            sid = f"{cid}_{comp['name']}"
            np.savez_compressed(
                os.path.join(args.out_dir, f"{sid}.npz"),
                images=np.stack(images).astype(np.uint8),
                poses=np.stack(poses).astype(np.float32),
                centerline_mm=cl_mm.astype(np.float32),
                radius_mm=radius_mm.astype(np.float32),
                topology=topo.astype(np.uint8),
                patient_id=cid, vessel=comp["name"], split=split_of[cid],
                crop_lo=lo.astype(np.int32), crop_shape=cshape.astype(np.int32),
                spacing=spacing.astype(np.float32), sVoxel=sVoxel.astype(np.float32),
                geometry=json.dumps([{k2: (v2.tolist() if isinstance(v2, np.ndarray) else v2)
                                      for k2, v2 in vw.items()} for vw in views]),
                motion=json.dumps(motion),
            )
            rows.append(dict(sample=sid, patient=cid, vessel=comp["name"],
                             split=split_of[cid], consistency=cons,
                             dlt_rms_px=[round(r, 2) for r in rmss],
                             pts_kept=int(in_crop.sum()), pts_total=int(len(vox)),
                             frac_in_crop=float(in_crop.mean()),
                             vessel_bbox_mm=[round(float(b), 1) for b in bbox_mm],
                             sVoxel_mm=[round(float(s), 1) for s in sVoxel]))
            flag = "  <-- vessel exceeds crop" if bbox_mm.max() > args.crop_mm else ""
            print(f"{sid:>12} [{split_of[cid]:>5}]: cons v1 {cons[0]:.3f} v2 {cons[1]:.3f} | "
                  f"DLT rms {rmss[0]:.2f}/{rmss[1]:.2f}px | in-crop {in_crop.mean():.3f} "
                  f"({in_crop.sum()}/{len(vox)} pts) | bbox {np.round(bbox_mm,0)} "
                  f"| vol {np.round(sVoxel,0)}{flag}", flush=True)
        times.append(time.time() - t0)
        done += 1

    if not rows:
        print("No samples produced — check paths."); return

    allc = np.array([c for r in rows for c in r["consistency"]])
    fic = np.array([r["frac_in_crop"] for r in rows])
    rmsall = np.array([v for r in rows for v in r["dlt_rms_px"]])
    per_patient = float(np.mean(times))
    print("\n" + "=" * 70)
    print(f"patients {done} | samples {len(rows)} ({len(rows)/max(done,1):.1f} vessels/patient)")
    print(f"consistency  mean {allc.mean():.3f}  min {allc.min():.3f}  max {allc.max():.3f}   [tol {args.tol_px}px]")
    print(f"DLT rms      mean {rmsall.mean():.2f}px  max {rmsall.max():.2f}px")
    print(f"centerline kept inside {args.crop_mm:.0f}mm crop: mean {fic.mean():.3f}  min {fic.min():.3f}")
    bb = np.array([r["vessel_bbox_mm"] for r in rows])
    n_over = int((bb.max(1) > args.crop_mm).sum())
    print(f"vessel extent (max axis): mean {bb.max(1).mean():.1f}mm  max {bb.max(1).max():.1f}mm  "
          f"| {n_over}/{len(rows)} exceed the {args.crop_mm:.0f}mm crop")
    if fov_warn:
        print(f"WARNING: {fov_warn} view(s) had crop extent > detector FOV — raise DSO / lower crop_mm")
    print(f"per-patient {per_patient:.1f}s  ->  ETA 1000 patients: {per_patient*1000/3600:.1f} h")
    if allc.mean() >= args.min_consistency:
        print(f"GATE PASSED (>= {args.min_consistency}). Dataset v2 geometry is validated.")
    else:
        print(f"GATE FAILED (< {args.min_consistency}). Do not train.")
        print("  If frac_in_crop is low -> vessel exceeds the crop; raise --crop_mm (watch FOV).")
        print("  If frac_in_crop ~1 but consistency low -> pose/convention issue; rerun debug_consistency.py.")
    print("=" * 70)
    json.dump(rows, open(os.path.join(args.out_dir, "pilot_report.json"), "w"), indent=2)
    print("saved pilot_report.json + case_splits_v2.json + per-sample npz")


if __name__ == "__main__":
    main()
