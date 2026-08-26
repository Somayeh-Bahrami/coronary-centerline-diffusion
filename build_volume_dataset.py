"""Stage 0 (Option B: X-rays -> 3D volume -> centerline).

Build fixed-size SOFT vessel-volume targets from the ImageCAS segmentation
masks, for the Stage-1 (X-ray -> volume) diffusion model.

For each case:
  * load the vessel segmentation mask (<id>.label.nii.gz),
  * resample it to a fixed TARGET^3 grid that spans the SAME physical extent
    as the original volume -- so the projection matrices you already have
    (which map centered-mm -> detector pixels) still project onto this grid,
  * store it as a SOFT occupancy in [0, 1] (diffusion trains far better on a
    continuous target than a hard 0/1 mask, and soft/dilated occupancy is
    robust to how thin coronary vessels are at 128^3),
  * save a per-case .npz with the volume + the geometry metadata needed later
    to relate voxel indices to physical (centered-mm) coordinates.

Run this where the ImageCAS masks live (same machine/dir that ran your DRR
generation and preprocessing). It does NOT need a GPU.

Usage:
    python build_volume_dataset.py \
        --raw_dir   /path/to/imagecas/labels \
        --splits    /path/to/case_splits.json \
        --out_dir   /path/to/output/volumes_128 \
        --target 128
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, binary_dilation, gaussian_filter


def _fit_exact(vol, target):
    """zoom() can land at target +/- 1 per axis; crop/pad to exactly target^3."""
    out = np.zeros((target, target, target), dtype=vol.dtype)
    s = [min(vol.shape[i], target) for i in range(3)]
    out[:s[0], :s[1], :s[2]] = vol[:s[0], :s[1], :s[2]]
    return out


def build_volume(mask_path, target=128, dilate_iter=1, blur_sigma=0.6):
    """Return (soft_volume[target^3] float32 in [0,1], orig_shape, spacing_mm, svoxel_mm)."""
    nii = nib.load(str(mask_path))
    spacing = np.array(nii.header.get_zooms()[:3], dtype=np.float32)
    # binary vessel mask
    mask = np.asarray(nii.get_fdata()) > 0
    orig_shape = np.array(mask.shape[:3], dtype=np.int32)

    # Dilate slightly first so 1-voxel-thin vessels survive the ~4x downsample.
    if dilate_iter > 0:
        mask = binary_dilation(mask, iterations=dilate_iter)

    factors = target / orig_shape.astype(np.float32)
    vol = zoom(mask.astype(np.float32), factors,
               order=1)       # linear -> soft [0,1]
    vol = _fit_exact(vol, target)

    if blur_sigma > 0:
        vol = gaussian_filter(vol, blur_sigma)

    vmax = float(vol.max())
    if vmax > 0:
        # normalize to [0,1]
        vol = vol / vmax

    svoxel = (orig_shape.astype(np.float32) *
              spacing)         # physical extent (mm)
    return vol.astype(np.float32), orig_shape, spacing, svoxel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True,
                    help="Dir with <id>.label.nii.gz ImageCAS masks.")
    ap.add_argument("--splits", required=True,
                    help="case_splits.json (train/val/test id lists).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target", type=int, default=128)
    ap.add_argument("--dilate_iter", type=int, default=1)
    ap.add_argument("--blur_sigma", type=float, default=0.6)
    ap.add_argument("--mask_suffix", default=".label.nii.gz")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.splits) as f:
        splits = json.load(f)
    all_ids = [cid for split in ("train", "val", "test")
               for cid in splits.get(split, [])]
    split_of = {cid: sp for sp in ("train", "val", "test")
                for cid in splits.get(sp, [])}
    print(f"{len(all_ids)} cases across splits; target={args.target}^3")

    t0 = time.time()
    occ_fracs, done, missing = [], 0, []
    for i, cid in enumerate(all_ids):
        mask_path = raw_dir / f"{cid}{args.mask_suffix}"
        if not mask_path.exists():
            missing.append(cid)
            continue
        vol, orig_shape, spacing, svoxel = build_volume(
            mask_path, args.target, args.dilate_iter, args.blur_sigma)
        occ = float((vol > 0.05).mean())
        occ_fracs.append(occ)
        np.savez_compressed(
            out_dir / f"{cid}.npz",
            # (T,T,T) float32 [0,1]
            volume=vol,
            orig_shape=orig_shape,                       # original voxel dims
            spacing=spacing,                             # original mm/voxel
            # physical extent (mm) = shape*spacing
            svoxel=svoxel,
            target=np.int32(args.target),
            split=split_of[cid],
        )
        done += 1
        if i % 50 == 0:
            print(
                f"  [{i}/{len(all_ids)}] {cid}: occ={occ:.4f}  {time.time()-t0:.0f}s")

    print(
        f"\nDONE: {done} volumes written to {out_dir}  ({time.time()-t0:.0f}s)")
    if occ_fracs:
        of = np.array(occ_fracs)
        print(f"occupancy fraction (>0.05): mean={of.mean():.4f}  "
              f"min={of.min():.4f}  max={of.max():.4f}")
        n_empty = int((of < 1e-4).sum())
        print(f"cases with ~empty volume (vessels lost!): {n_empty}  "
              f"-- if >0, raise --dilate_iter or lower --blur_sigma")
    if missing:
        print(f"MISSING masks for {len(missing)} ids, e.g. {missing[:5]} "
              f"-- check --raw_dir / --mask_suffix")


if __name__ == "__main__":
    main()
