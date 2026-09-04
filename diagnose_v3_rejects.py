"""Why did v3 samples fail the consistency check?

Splits every missed centerline point into:
  (a) OFF-DETECTOR  -> projects outside the 512x512 frame (input image is
                       clipped too; the crop cube's rotated silhouette is
                       bigger than the detector)
  (b) OFF-MASK      -> lands on the detector but not on the vessel silhouette
                       (tolerance / thin-projection issue)

and reports the detector size that WOULD have contained every point.

Usage (from the repo root):
    python diagnose_v3_rejects.py --pilot_dir /kaggle/working/dataset_v3_pilot
"""
import argparse, glob, json, os
import numpy as np
from scipy.ndimage import binary_dilation

DET_N = 512


def project_points(P, pts_mm):
    h = np.hstack([pts_mm, np.ones((len(pts_mm), 1))])
    uvw = (P @ h.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_dir", required=True)
    ap.add_argument("--tol_px", type=float, default=2.0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.pilot_dir, "*.npz")))
    if not files:
        print("no npz found"); return

    rows, need_px = [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        sid = os.path.basename(f)[:-4]
        cl = z["centerline"][z["centerline_mask"]][:, :3]
        det_sp = json.loads(str(z["geometry"]))[0]["det_spacing"]
        for k in range(2):
            P = z["poses"][k].astype(np.float64)
            img = z["images"][k].astype(bool)
            uv = project_points(P, cl)
            col = np.round(uv[:, 0]).astype(int); row = np.round(uv[:, 1]).astype(int)
            inb = (col >= 0) & (col < DET_N) & (row >= 0) & (row < DET_N)
            tol = binary_dilation(img, iterations=int(args.tol_px)) if args.tol_px > 0 else img
            hit = np.zeros(len(uv), bool)
            hit[inb] = tol[row[inb], col[inb]]

            miss = ~hit
            off_det = miss & ~inb
            off_mask = miss & inb
            # how large would the detector have to be to contain every point?
            r = np.abs(uv - (DET_N / 2.0)).max()
            need_px.append(2 * r)
            rows.append(dict(sample=sid, view=k, consistency=float(hit.mean()),
                             off_detector=float(off_det.mean()),
                             off_mask=float(off_mask.mean()),
                             need_det_px=float(2 * r),
                             need_det_mm=float(2 * r * det_sp)))
            flag = "  <-- FAILS" if hit.mean() < 0.95 else ""
            print(f"{sid:>12} v{k}: cons {hit.mean():.3f} | off-detector {off_det.mean():6.3f}"
                  f" | off-mask {off_mask.mean():6.3f} | needs {2*r:6.0f} px "
                  f"({2*r*det_sp:6.1f} mm){flag}")

    od = np.array([r["off_detector"] for r in rows])
    om = np.array([r["off_mask"] for r in rows])
    nd = np.array([r["need_det_px"] for r in rows])
    nm = np.array([r["need_det_mm"] for r in rows])
    print("\n" + "=" * 70)
    print(f"views {len(rows)} | mean off-detector {od.mean():.4f} | mean off-mask {om.mean():.4f}")
    print(f"detector needed: max {nd.max():.0f} px ({nm.max():.1f} mm)  "
          f"| p95 {np.percentile(nd,95):.0f} px ({np.percentile(nm,95):.1f} mm)"
          f"  [current 512 px]")
    if od.sum() > om.sum():
        need_sp = nm.max() / DET_N
        print("VERDICT: dominated by OFF-DETECTOR -> the 2D inputs are clipped.")
        print(f"  Fix A: keep 512 px, set detector spacing to >= {need_sp:.4f} mm "
              f"(vs DeepCA 0.2779) -> deviates from DeepCA, keeps image size.")
        print(f"  Fix B: keep 0.2779 mm, raise DET_N to >= {int(np.ceil(nd.max()/64)*64)} px "
              f"-> matches DeepCA optics, larger images.")
        print( "  Fix C: shrink --crop_mm until the rotated cube fits (loses more vessel).")
    else:
        print("VERDICT: dominated by OFF-MASK -> geometry is fine, silhouette/tolerance is not.")
        print("  Fix: raise --tol_px to 3, or check thin distal vessels surviving projection.")
    print("=" * 70)
    json.dump(rows, open(os.path.join(args.pilot_dir, "reject_diagnosis.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
