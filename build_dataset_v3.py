"""Dataset v3 (Plan A) — single isotropic frame, GT regenerated after crop.

Order of operations (each step feeds the next, one coordinate frame throughout):

    ImageCAS <id>.label.nii.gz
      -> RCA / LCA split (3D connected components, side from NIfTI affine)
      -> 96 mm crop, bbox-centred, SHIFTED (not shrunk) at volume edges
      -> resample ONCE to isotropic (default 0.35 mm)      [pure upsampling]
      -> skeletonize  +  distance_transform_edt(sampling=iso)  -> radius in mm
      -> topology labels  +  DFS traversal ordering (reused from centerline.py)
      -> centerline mm relative to the isotropic volume centre = isocentre
      -> TIGRE projection of THAT SAME isotropic volume (DeepCA geometry)
      -> centroid-DLT pose calibration
      -> hard gate, pad to fixed length, save

The same isotropic volume is used for BOTH the projections and the
skeletonization, so mask / centerline / radius / projection / pose share one
grid, one centre and one motion instance by construction.

Normalisation stats are computed in a second pass over TRAIN patients only and
written next to the split; samples store RAW mm (the transform is applied at
load time).

Requires CUDA (TIGRE).

Usage:
    python build_dataset_v3.py --raw_dir DIR --out_dir ./dataset_v3 --n 10
"""
import argparse, glob, json, os, time
import numpy as np
import nibabel as nib
from scipy.ndimage import (binary_dilation, center_of_mass, convolve,
                           distance_transform_edt, label as cc_label, zoom)
from skimage.morphology import skeletonize
import tigre

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

LABEL_ENDPOINT, LABEL_REGULAR, LABEL_BIFURCATION = 0, 1, 2
_NEIGHBOR_KERNEL = np.ones((3, 3, 3)); _NEIGHBOR_KERNEL[1, 1, 1] = 0
_NEIGHBOR_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     for dz in (-1, 0, 1) if not (dx == dy == dz == 0)]


# ======================= geometry / projection =======================
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
    """Intensity-weighted centroid (sub-pixel). np.argmax is biased to the
    top-left of a flat-topped blob — that was the v1 pose error."""
    m = proj >= 0.5 * proj.max()
    if m.sum() == 0:
        r, c = np.unravel_index(np.argmax(proj), proj.shape)
        return float(c), float(r)
    r, c = center_of_mass(proj * m)
    return float(c), float(r)


def project_points(P, pts_mm):
    h = np.hstack([pts_mm, np.ones((len(pts_mm), 1))])
    uvw = (P @ h.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def calibrate_P(shape, spacing, view, n_points=20, cal_n=64, seed=0):
    sVoxel = np.array(shape) * np.array(spacing)
    cal_shape = (cal_n, cal_n, cal_n); cal_spacing = sVoxel / cal_n
    geo = build_geo(cal_shape, cal_spacing, view)
    angles = np.array([[np.radians(view["alpha"]), np.radians(view["beta"]), 0.0]], np.float32)
    rng = np.random.default_rng(seed); centre = np.array(cal_shape) / 2.0
    corr = []
    for _ in range(n_points):
        idx = rng.integers(6, np.array(cal_shape) - 6)
        vol = np.zeros(cal_shape, dtype=np.float32); vol[tuple(idx)] = 1.0
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
    P = Vt[-1].reshape(3, 4); P = (P / P[-1, -1]).astype(np.float64)
    pts = np.array([c[0] for c in corr]); obs = np.array([[c[1], c[2]] for c in corr])
    rms = float(np.sqrt(((project_points(P, pts) - obs) ** 2).sum(1).mean()))
    return P, rms


# ======================= vessel split / crop / resample =======================
def side_labels(affine, centroids):
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
    lab, n = cc_label(mask, structure=np.ones((3, 3, 3), int))
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = [i for i in range(1, n + 1) if sizes[i] >= MIN_COMPONENT_FRAC * mask.sum()]
    keep = sorted(keep, key=lambda i: -sizes[i])[:2]
    if not keep:
        return []
    centroids = [np.array(center_of_mass(lab == i)) for i in keep]
    names = side_labels(affine, centroids)
    return [dict(name=(names[j] if names else f"c{j}"), mask=(lab == i),
                 n_vox=int(sizes[i])) for j, i in enumerate(keep)]


def crop_box(mask, shape, spacing, crop_mm):
    """96 mm box on the vessel's bounding-box centre; shifted, never shrunk."""
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
    out = zoom(mask_crop.astype(np.uint8), factors, order=0, grid_mode=True, mode="nearest")
    return out.astype(bool)


# ======================= centerline GT (reused from centerline.py) =======================
def _build_adjacency(coords):
    index_of = {tuple(c): i for i, c in enumerate(coords.astype(np.int64))}
    adjacency = [[] for _ in range(len(coords))]
    for i, c in enumerate(coords.astype(np.int64)):
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            j = index_of.get((c[0] + dx, c[1] + dy, c[2] + dz))
            if j is not None:
                adjacency[i].append(j)
    return adjacency


def _traversal_order(coords, radii):
    """DFS pre-order from the largest-radius endpoint; components largest
    first. Verbatim logic from src/coronarycl/centerline.py (53% -> 98.6%
    consecutive-pair adjacency)."""
    n = len(coords)
    adjacency = _build_adjacency(coords)
    degree = np.array([len(a) for a in adjacency])
    unvisited, components = set(range(n)), []
    while unvisited:
        start = next(iter(unvisited)); stack, seen, comp = [start], {start}, []
        while stack:
            node = stack.pop(); comp.append(node)
            for nb in adjacency[node]:
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        components.append(comp); unvisited -= seen
    components.sort(key=len, reverse=True)
    order = []
    for comp in components:
        endpoints = [i for i in comp if degree[i] <= 1]
        root = max(endpoints if endpoints else comp, key=lambda i: radii[i])
        stack, seen, comp_order = [root], {root}, []
        while stack:
            node = stack.pop(); comp_order.append(node)
            for nb in sorted(adjacency[node]):
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        order.extend(comp_order)
    order = np.array(order, dtype=np.int64)
    assert len(order) == n and len(set(order.tolist())) == n
    return order, len(components)


def _classify_topology(skel):
    cnt = convolve(skel.astype(np.uint8), _NEIGHBOR_KERNEL, mode="constant", cval=0)[skel]
    lab = np.full(cnt.shape, LABEL_REGULAR, np.int64)
    lab[cnt <= 1] = LABEL_ENDPOINT
    lab[cnt >= 3] = LABEL_BIFURCATION
    return lab


def centerline_from_iso(iso_mask, iso):
    """Skeleton + radius(mm) + topology + DFS order, on the isotropic grid.
    Skeletonisation runs on the vessel's tight sub-box (identical result,
    far cheaper than the full 275^3 volume)."""
    idx = np.argwhere(iso_mask)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, np.array(iso_mask.shape))
    sub = iso_mask[tuple(slice(l, h) for l, h in zip(lo, hi))]

    skel = skeletonize(sub)
    if skel.sum() < 20:
        return None
    dist_mm = distance_transform_edt(sub, sampling=(iso, iso, iso))   # radius in mm
    coords = np.argwhere(skel) + lo                                   # back to iso-grid indices
    radii = dist_mm[skel]
    topo = _classify_topology(skel)
    order, n_comp = _traversal_order(np.argwhere(skel), radii)
    return coords[order], radii[order], topo[order], n_comp


def adjacency_fraction(coords):
    """QC: fraction of consecutive ordered pairs that are 26-connected."""
    if len(coords) < 2:
        return 1.0
    d = np.abs(np.diff(coords, axis=0)).max(1)
    return float((d <= 1).mean())


def pad_to(arr, max_len):
    n = len(arr)
    out = np.zeros((max_len, arr.shape[1]), np.float32)
    m = np.zeros(max_len, bool)
    out[:n] = arr[:max_len]; m[:min(n, max_len)] = True
    return out, m


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
    ap.add_argument("--min_consistency", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_ids = sorted(int(os.path.basename(f).split(".")[0])
                     for f in glob.glob(os.path.join(args.raw_dir, "*.label.nii.gz")))
    perm = np.random.default_rng(args.seed).permutation(len(all_ids))
    shuffled = [all_ids[i] for i in perm]
    n_tr, n_va = int(0.8 * len(shuffled)), int(0.1 * len(shuffled))
    split_of = {c: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, c in enumerate(shuffled)}
    json.dump({s: [c for c in shuffled if split_of[c] == s] for s in ("train", "val", "test")},
              open(os.path.join(args.out_dir, "case_splits_v3.json"), "w"), indent=2)
    print(f"patient split over {len(all_ids)}: "
          f"{sum(v=='train' for v in split_of.values())}/{sum(v=='val' for v in split_of.values())}/"
          f"{sum(v=='test' for v in split_of.values())}  (both vessels share a split)\n")

    rng = np.random.default_rng(args.seed)
    rows, times, failures, done = [], [], [], 0
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

            # ---- QC: radius sane, skeleton inside mask ----
            r_ok = bool(np.isfinite(radii_mm).all() and (radii_mm > 0).all())
            inside = float(iso_mask[coords[:, 0], coords[:, 1], coords[:, 2]].mean())
            n_end = int((topo == LABEL_ENDPOINT).sum())

            views, motion = sample_geometry(rng)
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
                tolm = binary_dilation(binary, iterations=int(args.tol_px)) if args.tol_px > 0 else binary
                hit = np.zeros(len(uv), bool); hit[ib] = tolm[row[ib], col[ib]]
                cons.append(float(hit.mean()))

            gt = np.concatenate([cl_mm, radii_mm[:, None], topo[:, None]], 1).astype(np.float32)
            padded, pmask = pad_to(gt, args.max_points)
            if len(gt) > args.max_points:
                failures.append((sid, f"truncated {len(gt)}>{args.max_points}"))

            np.savez_compressed(
                os.path.join(args.out_dir, f"{sid}.npz"),
                images=np.stack(images).astype(np.uint8),
                poses=np.stack(poses).astype(np.float32),
                centerline=padded, centerline_mask=pmask,   # cols: x_mm,y_mm,z_mm,radius_mm,topology
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
                             n_points=int(len(gt)), adjacency=round(adj, 3),
                             radius_ok=r_ok, radius_mm=[round(float(radii_mm.min()), 2),
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

    # ---- second pass: normalisation stats from TRAIN patients only ----
    tr = [r for r in rows if r["split"] == "train"]
    if tr:
        pts, rads = [], []
        for r in tr:
            z = np.load(os.path.join(args.out_dir, f"{r['sample']}.npz"))
            g = z["centerline"][z["centerline_mask"]]
            pts.append(g[:, :3]); rads.append(g[:, 3])
        pts = np.concatenate(pts); rads = np.concatenate(rads)
        stats = {"coord_mean": pts.mean(0).tolist(), "coord_std": pts.std(0).tolist(),
                 "radius_mean": float(rads.mean()), "radius_std": float(rads.std()),
                 "n_train_samples": len(tr), "iso_mm": args.iso, "crop_mm": args.crop_mm,
                 "max_points": args.max_points}
        json.dump(stats, open(os.path.join(args.out_dir, "norm_stats_v3.json"), "w"), indent=2)
        print(f"\nnorm stats (train only, n={len(tr)}): "
              f"coord_std {np.round(stats['coord_std'],2)} radius {stats['radius_mean']:.2f}"
              f"+/-{stats['radius_std']:.2f} mm")

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
    if any(not r["fov_ok"] for r in rows):
        print(f"WARNING: {sum(not r['fov_ok'] for r in rows)} sample(s) exceeded detector FOV")
    for s, why in failures:
        print(f"  FAILURE {s}: {why}")
    per = float(np.mean(times))
    print(f"per-patient {per:.1f}s  ->  ETA 1000 patients: {per*1000/3600:.1f} h")
    ok = (allc.mean() >= args.min_consistency and rmsall.max() < 1.0
          and adjall.mean() > 0.95 and all(r["radius_ok"] for r in rows)
          and npts.max() <= args.max_points)
    print("GATE PASSED — dataset v3 validated." if ok else
          "GATE FAILED — inspect the failing metric above; do not train.")
    print("=" * 72)
    json.dump(rows, open(os.path.join(args.out_dir, "pilot_report_v3.json"), "w"), indent=2)
    print("saved pilot_report_v3.json + case_splits_v3.json + norm_stats_v3.json + per-sample npz")


if __name__ == "__main__":
    main()
