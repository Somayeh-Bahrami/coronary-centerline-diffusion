"""Dataset v3 (Plan A) — single isotropic frame, GT regenerated after crop.

This builder REUSES the repo's validated modules rather than duplicating them,
so there is exactly one implementation of each piece of logic:

    src.coronarycl.centerline     _build_adjacency, _traversal_order,
                                  _classify_topology, LABEL_*  (DFS ordering,
                                  53% -> 98.6% consecutive-pair adjacency)
    src.coronarycl.preprocessing  pad_centerline, compute_centerline_norm_stats
    src.coronarycl.splits         make_case_level_split, write_splits

Only the v3-specific steps live here: vessel-mask projection, DeepCA geometry,
centroid DLT, RCA/LCA split, crop, isotropic resampling, and the QC gate.

Order of operations (each step feeds the next, one coordinate frame throughout):

    ImageCAS <id>.label.nii.gz
      -> RCA / LCA split (3D connected components, side from NIfTI affine)
      -> 96 mm crop, bbox-centred, SHIFTED (not shrunk) at volume edges
      -> resample ONCE to isotropic (default 0.35 mm)      [pure upsampling]
      -> skeletonize  +  distance_transform_edt(sampling=iso)  -> radius in mm
      -> topology labels  +  DFS traversal ordering  (centerline.py)
      -> centerline mm relative to the isotropic volume centre = isocentre
      -> TIGRE projection of THAT SAME isotropic volume (DeepCA geometry)
      -> centroid-DLT pose calibration
      -> hard gate, pad to fixed length (preprocessing.py), save

The same isotropic volume is used for BOTH the projections and the
skeletonization, so mask / centerline / radius / projection / pose share one
grid, one centre and one motion instance by construction.

Normalisation stats are computed in a second pass over TRAIN patients only and
written next to the split; samples store RAW mm (the transform is applied at
load time).

Requires CUDA (TIGRE). Run from the repository root so `src.coronarycl`
imports resolve.

Usage:
    python build_dataset_v3.py --raw_dir DIR --out_dir ./dataset_v3 --n 10
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import (binary_dilation, center_of_mass,
                           distance_transform_edt, label as cc_label, zoom)
from skimage.morphology import skeletonize
import tigre

sys.path.insert(0, os.getcwd())          # repo root -> `src.coronarycl`
from src.coronarycl.centerline import (          # noqa: E402
    _traversal_order, _classify_topology,
    LABEL_ENDPOINT, LABEL_REGULAR, LABEL_BIFURCATION,
)
from src.coronarycl.preprocessing import (       # noqa: E402
    pad_centerline, compute_centerline_norm_stats,
)
from src.coronarycl.splits import (              # noqa: E402
    make_case_level_split, write_splits,
)

# ---------------- DeepCA geometry (Wang et al., WACV 2025, Table 3) ----------------
DET_N = 512
DET_SPACING_RANGE = (0.2769, 0.2789)
V1_DSD_RANGE, V2_DSD_RANGE = (970.0, 1010.0), (1050.0, 1070.0)
V1_DSO_RANGE, V2_DSO_JITTER = (745.0, 785.0), 3.0
V1_PRIMARY, V1_SECONDARY = (18.0, 42.0), (-8.0, 8.0)
V2_PRIMARY, V2_SECONDARY = (-8.0, 8.0), (18.0, 42.0)
MOTION_ROT_DEG, MOTION_TRANS_MM = 10.0, 8.0
CROP_MM_DEFAULT, ISO_MM_DEFAULT = 96.0, 0.35
MIN_COMPONENT_FRAC = 0.05
_CONN26 = np.ones((3, 3, 3), int)


# ======================= geometry / projection =======================
def sample_geometry(rng, motion_2d=False):
    """One DeepCA-style two-view acquisition; view 2 also carries the rigid
    motion perturbation (DeepCA protocol: +/-10 deg rotation, +/-8 mm
    translation).

    `motion_2d=True` zeroes the out-of-plane translation component. DeepCA's
    paper and Table 3 state only "translations +/-8 mm" without a per-axis
    breakdown, and their projections_simulation.py could not be inspected
    here, so whether their translation is 2- or 3-axis is UNVERIFIED. The
    default keeps 3-axis translation; set --motion_2d if you confirm from
    their source that the out-of-plane component is zero.
    """
    det_sp = rng.uniform(*DET_SPACING_RANGE)
    dso1 = rng.uniform(*V1_DSO_RANGE)
    # Only the two rotation components that are actually applied (primary /
    # secondary angulation) are sampled -- an earlier version drew a third
    # and recorded it in the metadata without ever using it.
    rot = rng.uniform(-MOTION_ROT_DEG, MOTION_ROT_DEG, size=2)
    trans = rng.uniform(-MOTION_TRANS_MM, MOTION_TRANS_MM, size=3)
    if motion_2d:
        trans[2] = 0.0
    v1 = dict(alpha=float(rng.uniform(*V1_PRIMARY)), beta=float(rng.uniform(*V1_SECONDARY)),
              DSD=float(rng.uniform(*V1_DSD_RANGE)), DSO=float(dso1),
              det_spacing=float(det_sp), offOrigin=np.zeros(3, np.float32))
    v2 = dict(alpha=float(rng.uniform(*V2_PRIMARY) + rot[0]),
              beta=float(rng.uniform(*V2_SECONDARY) + rot[1]),
              DSD=float(rng.uniform(*V2_DSD_RANGE)),
              DSO=float(dso1 + rng.uniform(-V2_DSO_JITTER, V2_DSO_JITTER)),
              det_spacing=float(det_sp), offOrigin=np.array(trans, np.float32))
    return [v1, v2], dict(motion_rot_deg=rot.tolist(),        # [primary, secondary]
                          motion_trans_mm=trans.tolist(), motion_2d=bool(motion_2d))


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
    """Sub-pixel marker location: intensity-weighted centroid of the blob.
    Returns None when the marker is not visible (caller skips it).

    np.argmax returns the FIRST index of a flat-topped plateau -- its
    top-left corner -- a systematic multi-pixel bias. That was the v1 pose
    error (DLT RMS 2.7-3.4 px vs 0.14-1.04 px with the centroid).
    """
    peak = proj.max()
    if not np.isfinite(peak) or peak <= 0:
        return None
    m = proj >= 0.5 * peak
    row, col = center_of_mass(proj * m)
    if not (np.isfinite(row) and np.isfinite(col)):
        row, col = np.unravel_index(np.argmax(proj), proj.shape)
    return float(col), float(row)


def project_points(P, pts_mm):
    """mm (relative to the volume centre) -> (col, row) pixel coordinates."""
    h = np.hstack([pts_mm, np.ones((len(pts_mm), 1))])
    uvw = (P @ h.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def calibrate_P(shape, spacing, view, n_points=20, cal_n=64, seed=0):
    """DLT against TIGRE's real Ax(). The calibration phantom has the SAME
    physical extent (sVoxel) as the volume being projected, so P transfers
    exactly; only the sampling resolution is coarser."""
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
        centre_px = _blob_centre(tigre.Ax(vol, geo, angles)[0])
        if centre_px is None:
            continue
        x, y = centre_px
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
    pts = np.array([c[0] for c in corr])
    obs = np.array([[c[1], c[2]] for c in corr])
    rms = float(np.sqrt(((project_points(P, pts) - obs) ** 2).sum(1).mean()))
    return P, rms


# ======================= vessel split / crop / resample =======================
def side_labels(affine, centroids):
    """Name components by anatomical side from the NIfTI axis codes rather
    than assuming an array orientation; None -> fall back to size rank."""
    try:
        codes = nib.aff2axcodes(affine)
    except Exception:
        return None
    ax = next((i for i, c in enumerate(codes) if c in ("L", "R")), None)
    if ax is None:
        return None
    vals = [c[ax] for c in centroids]
    order = np.argsort(vals)
    ranked = order[::-1] if codes[ax] == "L" else order
    names = [None] * len(vals)
    for rank, ci in enumerate(ranked):
        names[ci] = "LCA" if rank == 0 else "RCA"
    return names


def split_components(mask, affine):
    """One sample per coronary system (3D connected component)."""
    lab, n = cc_label(mask, structure=_CONN26)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = [i for i in range(1, n + 1) if sizes[i] >= MIN_COMPONENT_FRAC * mask.sum()]
    keep = sorted(keep, key=lambda i: -sizes[i])[:2]
    if not keep:
        return []
    centroids = [np.array(center_of_mass(lab == i)) for i in keep]
    names = side_labels(affine, centroids)
    return [dict(name=(names[j] if names else f"c{j}"), mask=(lab == i),
                 n_vox=int(sizes[i])) for j, i in enumerate(keep)]


def crop_box(mask, shape, spacing, crop_mm):
    """crop_mm box on the vessel's bounding-box centre; shifted, never shrunk."""
    shape = np.array(shape)
    idx = np.argwhere(mask)
    centre = (idx.min(0) + idx.max(0)) / 2.0
    half = np.ceil((crop_mm / 2.0) / np.array(spacing)).astype(int)
    size = np.minimum(2 * half, shape)
    lo = np.round(centre).astype(int) - size // 2
    lo = np.clip(lo, 0, np.maximum(shape - size, 0))
    return lo, lo + size


def to_isotropic(mask_crop, spacing, iso):
    """Nearest-neighbour resample to an isotropic grid. With iso <= min(spacing)
    this is pure upsampling, so no thin vessel can be lost."""
    factors = np.array(spacing, float) / float(iso)
    out = zoom(mask_crop.astype(np.uint8), factors, order=0,
               grid_mode=True, mode="nearest")
    return out.astype(bool)


# ======================= centerline GT =======================
def centerline_from_iso(iso_mask, iso):
    """Skeleton + radius(mm) + topology + DFS order on the isotropic grid.

    Skeletonisation runs on the vessel's tight sub-box (identical result, far
    cheaper than the full crop). Ordering and topology come from
    src.coronarycl.centerline so there is one implementation, not two.
    """
    idx = np.argwhere(iso_mask)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, np.array(iso_mask.shape))
    sub = iso_mask[tuple(slice(l, h) for l, h in zip(lo, hi))]

    skel = skeletonize(sub)
    if skel.sum() < 20:
        return None
    dist_mm = distance_transform_edt(sub, sampling=(iso, iso, iso))   # radius in mm
    sub_coords = np.argwhere(skel)              # sub-box frame (ordering reference)
    radii = dist_mm[skel]
    topo = _classify_topology(skel)
    order = _traversal_order(sub_coords, radii)          # repo implementation
    n_comp = int(cc_label(skel, structure=_CONN26)[1])
    coords = (sub_coords + lo)[order]                    # back to iso-grid indices
    return coords, radii[order], topo[order], n_comp


def adjacency_fraction(coords):
    """QC: fraction of consecutive ordered pairs that are 26-connected."""
    if len(coords) < 2:
        return 1.0
    return float((np.abs(np.diff(coords, axis=0)).max(1) <= 1).mean())


# ======================= main =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="./dataset_v3")
    ap.add_argument("--n", type=int, default=10, help="number of PATIENTS")
    ap.add_argument("--crop_mm", type=float, default=CROP_MM_DEFAULT)
    ap.add_argument("--iso", type=float, default=ISO_MM_DEFAULT)
    ap.add_argument("--max_points", type=int, default=4000)
    ap.add_argument("--tol_px", type=float, default=2.0)
    ap.add_argument("--min_consistency", type=float, default=0.95,
                    help="per-sample, per-view minimum; samples below are REJECTED")
    ap.add_argument("--min_adjacency", type=float, default=0.95,
                    help="per-sample minimum ordering adjacency")
    ap.add_argument("--max_reject_frac", type=float, default=0.02,
                    help="gate fails if more than this fraction of samples is rejected")
    ap.add_argument("--motion_2d", action="store_true",
                    help="zero the out-of-plane motion translation (see sample_geometry)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- patient-level 80/10/10 split, before anything else (splits.py) ----
    all_ids = sorted(int(os.path.basename(f).split(".")[0])
                     for f in glob.glob(os.path.join(args.raw_dir, "*.label.nii.gz")))
    splits = make_case_level_split(all_ids, val_frac=0.10, test_frac=0.10, seed=args.seed)
    write_splits(splits, Path(args.out_dir) / "case_splits_v3.json")
    split_of = {c: s for s in ("train", "val", "test") for c in splits[s]}
    print(f"patient split over {len(all_ids)}: {len(splits['train'])}/"
          f"{len(splits['val'])}/{len(splits['test'])}  (both vessels share a split)\n")

    rng = np.random.default_rng(args.seed)
    rows, times, failures, rejected, done = [], [], [], [], 0
    n_comp_hist = []          # raw connected-component count per patient
    for cid in all_ids:
        if done >= args.n:
            break
        lp = os.path.join(args.raw_dir, f"{cid}.label.nii.gz")
        if not os.path.exists(lp):
            continue
        t0 = time.time()
        nii = nib.load(lp)
        mask_full = np.asarray(nii.get_fdata()) > 0.5
        spacing = np.array(nii.header.get_zooms()[:3], float)
        if mask_full.sum() < 100:
            continue

        _lab, _nraw = cc_label(mask_full, structure=_CONN26)
        _sz = np.bincount(_lab.ravel()); _sz[0] = 0
        _big = int((_sz >= MIN_COMPONENT_FRAC * mask_full.sum()).sum())
        n_comp_hist.append(_big)
        if _big > 2:
            failures.append((str(cid), f"{_big} significant components -- only the 2 "
                                       f"largest kept, {_big - 2} discarded"))
        for comp in split_components(mask_full, nii.affine):
            sid = f"{cid}_{comp['name']}"
            lo, hi = crop_box(comp["mask"], mask_full.shape, spacing, args.crop_mm)
            cmask = comp["mask"][tuple(slice(l, h) for l, h in zip(lo, hi))]
            mask_kept = float(cmask.sum()) / float(comp["mask"].sum())
            if cmask.sum() < 50:
                failures.append((sid, "empty crop")); continue

            iso_mask = to_isotropic(cmask, spacing, args.iso)
            iso_shape = np.array(iso_mask.shape)
            iso_sp = np.array([args.iso] * 3)
            sVoxel = iso_shape * iso_sp

            cl = centerline_from_iso(iso_mask, args.iso)
            if cl is None:
                failures.append((sid, "skeleton too small")); continue
            coords, radii_mm, topo, n_comp = cl
            # mm relative to the isotropic volume centre == isocentre == calibration origin
            cl_mm = (coords + 0.5 - iso_shape / 2.0) * iso_sp
            adj = adjacency_fraction(coords)

            r_ok = bool(np.isfinite(radii_mm).all() and (radii_mm > 0).all())
            inside = float(iso_mask[coords[:, 0], coords[:, 1], coords[:, 2]].mean())
            n_end = int((topo == LABEL_ENDPOINT).sum())

            # ---- pad at build time (preprocessing.py); raises if too long ----
            gt = np.concatenate([cl_mm, radii_mm[:, None], topo[:, None]], 1).astype(np.float32)
            try:
                padded, pmask = pad_centerline(gt, args.max_points)
            except ValueError:
                failures.append((sid, f"{len(gt)} points > --max_points {args.max_points}"))
                continue

            views, motion = sample_geometry(rng, motion_2d=args.motion_2d)
            images, poses, cons, rmss = [], [], [], []
            fov_ok = True
            for k, view in enumerate(views):
                fov = (DET_N * view["det_spacing"]) / (view["DSD"] / view["DSO"])
                if sVoxel.max() > fov:
                    fov_ok = False
                geo = build_geo(iso_shape, iso_sp, view)
                angles = np.array([[np.radians(view["alpha"]), np.radians(view["beta"]), 0.0]],
                                  np.float32)
                binary = tigre.Ax(iso_mask.astype(np.float32), geo, angles)[0] > 1e-6
                P, rms = calibrate_P(iso_shape, iso_sp, view, seed=args.seed + k)
                images.append(binary); poses.append(P); rmss.append(rms)

                uv = project_points(P, cl_mm)
                col = np.round(uv[:, 0]).astype(int); row = np.round(uv[:, 1]).astype(int)
                ib = (col >= 0) & (col < DET_N) & (row >= 0) & (row < DET_N)
                tolm = (binary_dilation(binary, iterations=int(args.tol_px))
                        if args.tol_px > 0 else binary)
                hit = np.zeros(len(uv), bool); hit[ib] = tolm[row[ib], col[ib]]
                cons.append(float(hit.mean()))

            # ---- PER-SAMPLE QC: reject before saving, never on a mean ----
            bad = []
            if min(cons) < args.min_consistency:
                bad.append(f"consistency {min(cons):.3f} < {args.min_consistency}")
            if adj < args.min_adjacency:
                bad.append(f"adjacency {adj:.3f} < {args.min_adjacency}")
            if not r_ok:
                bad.append("radius non-finite or <= 0")
            if max(rmss) >= 1.0:
                bad.append(f"DLT rms {max(rmss):.2f}px >= 1.0")
            if not fov_ok:
                bad.append("crop exceeds detector FOV")
            if bad:
                rejected.append((sid, "; ".join(bad)))
                print(f"{sid:>12} [{split_of[cid]:>5}]: REJECTED -- {'; '.join(bad)}", flush=True)
                continue

            np.savez_compressed(
                os.path.join(args.out_dir, f"{sid}.npz"),
                images=np.stack(images).astype(np.uint8),
                poses=np.stack(poses).astype(np.float32),
                centerline=padded, centerline_mask=pmask,   # x_mm,y_mm,z_mm,radius_mm,topology
                n_points=len(gt), patient_id=cid, vessel=comp["name"], split=split_of[cid],
                iso_mm=args.iso, iso_shape=iso_shape.astype(np.int32),
                sVoxel=sVoxel.astype(np.float32), crop_lo=lo.astype(np.int32),
                native_spacing=spacing.astype(np.float32),
                geometry=json.dumps([{a: (b.tolist() if isinstance(b, np.ndarray) else b)
                                      for a, b in vw.items()} for vw in views]),
                motion=json.dumps(motion),
            )
            rows.append(dict(sample=sid, patient=cid, vessel=comp["name"], split=split_of[cid],
                             consistency=cons, dlt_rms=[round(r, 3) for r in rmss],
                             n_points=int(len(gt)), adjacency=round(adj, 3), radius_ok=r_ok,
                             radius_mm=[round(float(radii_mm.min()), 2),
                                        round(float(radii_mm.max()), 2)],
                             skeleton_inside=round(inside, 4), n_endpoints=n_end,
                             n_skel_components=n_comp, mask_kept=round(mask_kept, 3),
                             fov_ok=fov_ok))
            print(f"{sid:>12} [{split_of[cid]:>5}]: cons {cons[0]:.3f}/{cons[1]:.3f} | "
                  f"rms {rmss[0]:.2f}/{rmss[1]:.2f}px | pts {len(gt)} | adj {adj:.3f} | "
                  f"r {radii_mm.min():.2f}-{radii_mm.max():.2f}mm | mask_kept {mask_kept:.3f} | "
                  f"comps {n_comp}{'' if fov_ok else '  FOV!'}", flush=True)
        times.append(time.time() - t0); done += 1

    if not rows:
        print("No samples produced."); return

    # ---- normalisation stats from TRAIN patients only (preprocessing.py) ----
    tr = [r for r in rows if r["split"] == "train"]
    if tr:
        mm_centerlines = {}
        for r in tr:
            z = np.load(os.path.join(args.out_dir, f"{r['sample']}.npz"))
            mm_centerlines[r["sample"]] = z["centerline"][z["centerline_mask"]]
        stats = compute_centerline_norm_stats(mm_centerlines, list(mm_centerlines))
        stats.update(n_train_samples=len(tr), iso_mm=args.iso,
                     crop_mm=args.crop_mm, max_points=args.max_points)
        json.dump(stats, open(os.path.join(args.out_dir, "norm_stats_v3.json"), "w"), indent=2)
        print(f"\nnorm stats (train only, n={len(tr)}): "
              f"coord_std {np.round(stats['coord_std'], 2)} "
              f"radius {stats['radius_mean']:.2f}+/-{stats['radius_std']:.2f} mm")

    allc = np.array([c for r in rows for c in r["consistency"]])
    rmsall = np.array([v for r in rows for v in r["dlt_rms"]])
    adjall = np.array([r["adjacency"] for r in rows])
    npts = np.array([r["n_points"] for r in rows])
    kept = np.array([r["mask_kept"] for r in rows])
    print("\n" + "=" * 72)
    print(f"patients {done} | samples {len(rows)} ({len(rows)/max(done,1):.1f} vessels/patient)")
    print(f"T3 consistency   mean {allc.mean():.3f}  min {allc.min():.3f}   [tol {args.tol_px}px]")
    print(f"T2 DLT rms       mean {rmsall.mean():.2f}px  max {rmsall.max():.2f}px")
    print(f"T1 skel inside   min {min(r['skeleton_inside'] for r in rows):.4f}")
    print(f"ordering adj     mean {adjall.mean():.3f}  min {adjall.min():.3f}  (target > 0.95)")
    print(f"radius sane      {sum(r['radius_ok'] for r in rows)}/{len(rows)} samples")
    print(f"points/vessel    mean {npts.mean():.0f}  max {npts.max()}  (max_points {args.max_points})")
    print(f"mask kept in crop mean {kept.mean():.3f}  min {kept.min():.3f}")
    print(f"skeleton comps   mean {np.mean([r['n_skel_components'] for r in rows]):.1f} (1 = clean tree)")
    if n_comp_hist:
        _h = np.array(n_comp_hist)
        print(f"vessel comps/pt  mean {_h.mean():.2f}  max {_h.max()}  "
              f"| patients with >2: {(_h > 2).sum()}")
    if any(not r["fov_ok"] for r in rows):
        print(f"WARNING: {sum(not r['fov_ok'] for r in rows)} sample(s) exceeded detector FOV")
    for s, why in failures:
        print(f"  FAILURE {s}: {why}")
    per = float(np.mean(times))
    print(f"per-patient {per:.1f}s  ->  ETA 1000 patients: {per*1000/3600:.1f} h")
    n_attempt = len(rows) + len(rejected)
    reject_frac = len(rejected) / max(n_attempt, 1)
    print(f"rejected         {len(rejected)}/{n_attempt} samples "
          f"({reject_frac:.1%}; limit {args.max_reject_frac:.1%})")
    for s_, why in rejected:
        print(f"  REJECTED {s_}: {why}")
    # Every SAVED sample already passed the per-sample QC above, so these
    # minima are guarantees, not averages. The gate only has to check that
    # too few samples were thrown away and that nothing was truncated.
    ok = (reject_frac <= args.max_reject_frac
          and allc.min() >= args.min_consistency
          and adjall.min() >= args.min_adjacency
          and rmsall.max() < 1.0
          and all(r["radius_ok"] for r in rows)
          and not failures)
    print("GATE PASSED — dataset v3 validated." if ok else
          "GATE FAILED — inspect the failing metric above; do not train.")
    print("=" * 72)
    json.dump(rows, open(os.path.join(args.out_dir, "pilot_report_v3.json"), "w"), indent=2)
    print("saved pilot_report_v3.json + case_splits_v3.json + norm_stats_v3.json + per-sample npz")


if __name__ == "__main__":
    main()
