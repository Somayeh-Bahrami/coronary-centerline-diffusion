"""Step 2.1 completion — full 20-case classical baseline average.
CPU-only, M4. Independent of Step 2.3 training -- run in parallel.

Usage:
    python run_baseline_eval.py
"""
import json
from pathlib import Path

import numpy as np
import nibabel as nib

from src.coronarycl.models.baseline import extract_2d_centerline, epipolar_baseline
from src.coronarycl.metrics import evaluate_case, ground_truth_to_mm

CENTERLINE_DIR = Path("data/processed/centerlines")
DRR_DIR = Path("data/processed/DRR_Generation/Total")
RAW_DIR = Path("data/raw")
SPLITS_DIR = Path("data/splits")


def run_one(case_id: int) -> dict:
    drr = np.load(
        DRR_DIR / f"case_{case_id}_projections.npz", allow_pickle=True)
    masks, poses = drr["masks"], drr["poses"]

    view0 = extract_2d_centerline(masks[0])
    view1 = extract_2d_centerline(masks[1])
    pred = epipolar_baseline([view0, view1], [poses[0], poses[1]])

    gt_voxel = np.load(CENTERLINE_DIR / f"{case_id}_centerline.npy")
    img = nib.load(str(RAW_DIR / f"{case_id}.label.nii.gz"))
    gt_mm = ground_truth_to_mm(gt_voxel, img.shape, img.header.get_zooms()[:3])
    return evaluate_case(pred, gt_mm)


def main():
    with open(SPLITS_DIR / "case_splits.json") as f:
        val_ids = json.load(f)["val"]

    results = {}
    for cid in val_ids:
        try:
            results[cid] = run_one(cid)
            print(f"case {cid}: chamfer_l2={results[cid]['chamfer_l2']:.2f}mm")
        except Exception as e:
            print(f"case {cid}: FAILED -- {e}")

    vals = [r["chamfer_l2"] for r in results.values()]
    print(f"\n{len(vals)}/{len(val_ids)} cases succeeded")
    print(f"Mean Chamfer L2: {np.mean(vals):.2f}mm (std {np.std(vals):.2f}; "
          f"prior single-case value was 23.35mm)")

    with open("baseline_val_results.json", "w") as f:
        json.dump({"per_case": {str(k): v for k, v in results.items()},
                   "mean_chamfer_l2": float(np.mean(vals))}, f, indent=2)


if __name__ == "__main__":
    main()
