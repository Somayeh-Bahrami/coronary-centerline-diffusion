"""Quick visual sanity check of the 128^3 vessel volumes (Stage 0 output).
Saves a max-intensity projection (three views) per case so you can confirm
they look like coronary trees. No GPU needed.

Usage:
    python check_volumes.py --vol_dir data/processed/volumes_128 --n 4
"""
import matplotlib.pyplot as plt
import argparse
import glob
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("--vol_dir", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--out", default="volume_checks.png")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.vol_dir, "*.npz")))[:args.n]
assert files, f"no .npz in {args.vol_dir}"

fig, axes = plt.subplots(len(files), 3, figsize=(9, 3*len(files)))
if len(files) == 1:
    axes = axes[None, :]
for r, f in enumerate(files):
    v = np.load(f)["volume"]                     # (128,128,128)
    cid = Path(f).stem
    for c, axis in enumerate([2, 1, 0]):         # MIP along z, y, x
        mip = v.max(axis=axis)
        axes[r, c].imshow(mip.T, cmap="hot", origin="lower")
        axes[r, c].set_title(
            f"{cid} — view {c}  (occ={float((v > 0.05).mean()):.4f})", fontsize=8)
        axes[r, c].axis("off")
plt.tight_layout()
plt.savefig(args.out, dpi=110)
print("saved", args.out,
      "— open it and confirm each row looks like a branching vessel tree")
