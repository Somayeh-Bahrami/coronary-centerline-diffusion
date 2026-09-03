"""Step 0 verification: is coronary lumen HU actually elevated in 3D,
before blaming DRR projection physics for low 2D vessel contrast.

Run where the RAW ImageCAS files live (Kaggle / wherever *.img.nii.gz +
*.label.nii.gz are), NOT on the packaged npz files (those don't have HU).

Usage:
    python check_hu_contrast_3d.py /path/to/imagecas_raw --n 15
"""
import argparse, glob, os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion

def case_stats(img_path, label_path, ring_iters=3):
    hu = nib.load(img_path).get_fdata().astype(np.float32)
    mask = nib.load(label_path).get_fdata() > 0.5
    if mask.sum() < 50:
        return None
    # erode 1 vox so we sample lumen core, not partial-volume edge
    core = binary_erosion(mask, iterations=1) if mask.sum() > 200 else mask
    if core.sum() < 20:
        core = mask
    # local background ring just outside the vessel (myocardium/fat), not whole thorax
    ring = binary_dilation(mask, iterations=ring_iters) & ~binary_dilation(mask, iterations=1)
    if ring.sum() < 20:
        return None
    vess_hu = hu[core]
    bg_hu = hu[ring]
    pooled_std = np.sqrt((vess_hu.var() + bg_hu.var()) / 2) + 1e-6
    effect = (vess_hu.mean() - bg_hu.mean()) / pooled_std
    return dict(
        vess_mean=float(vess_hu.mean()), vess_std=float(vess_hu.std()),
        bg_mean=float(bg_hu.mean()), bg_std=float(bg_hu.std()),
        effect_size=float(effect), n_vess_vox=int(mask.sum()),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()

    img_files = sorted(glob.glob(os.path.join(args.raw_dir, "*.img.nii.gz")))[:args.n]
    rows = []
    for img_path in img_files:
        cid = os.path.basename(img_path).split(".")[0]
        label_path = os.path.join(args.raw_dir, f"{cid}.label.nii.gz")
        if not os.path.exists(label_path):
            continue
        try:
            r = case_stats(img_path, label_path)
        except Exception as e:
            print(f"{cid}: FAILED {e}")
            continue
        if r is None:
            continue
        r["case"] = cid
        rows.append(r)
        print(f"{cid}: vessel HU {r['vess_mean']:.1f}±{r['vess_std']:.1f}  "
              f"bg HU {r['bg_mean']:.1f}±{r['bg_std']:.1f}  "
              f"effect size {r['effect_size']:.2f}  (n_vox={r['n_vess_vox']})")

    if rows:
        es = np.array([r["effect_size"] for r in rows])
        print("\n" + "="*60)
        print(f"n={len(rows)} cases | mean effect size {es.mean():.2f} "
              f"(median {np.median(es):.2f}, min {es.min():.2f}, max {es.max():.2f})")
        print("Rule of thumb: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")
        print("="*60)
        import json
        json.dump(rows, open("hu_contrast_3d_results.json", "w"), indent=2)
        print("saved hu_contrast_3d_results.json")
    else:
        print("No valid cases processed — check raw_dir path / file naming.")

if __name__ == "__main__":
    main()
