"""Step 1.3 -- ID consistency check + train/val/test split. Runs
locally on M4 (no GPU needed).

Case-level split ONLY -- never split by view, both views of a case must
stay together to avoid leakage (follows 3DGR-CAR's, MICCAI 2024, split
logic on the same ImageCAS lineage).
"""

import json
from pathlib import Path

import numpy as np


def get_verified_case_ids(centerline_dir: Path, drr_dir: Path) -> list:
    """ID consistency check: confirm every case has both a centerline
    (Step 1.1) and a DRR projection set (Step 1.2), and return only the
    IDs present in both. Raises if either side has cases the other is
    missing, since packaging later would silently fail on a case
    missing one half of its data.

    Does NOT assume case IDs form a contiguous range -- ImageCAS's
    numbering is not guaranteed contiguous, so IDs are derived from the
    files actually present rather than range(1, n+1).
    """
    centerline_ids = {int(f.stem.split('_')[0]) for f in Path(
        centerline_dir).glob("*_centerline.npy")}
    drr_ids = {int(f.stem.split('_')[1]) for f in Path(
        drr_dir).glob("case_*_projections.npz")}

    centerline_only = sorted(centerline_ids - drr_ids)
    drr_only = sorted(drr_ids - centerline_ids)
    if centerline_only or drr_only:
        raise RuntimeError(
            f"ID mismatch between centerlines and DRR projections -- "
            f"centerline-only: {centerline_only[:10]}, drr-only: {drr_only[:10]}. "
            f"Fix Step 1.1/1.2 before splitting."
        )

    return sorted(centerline_ids)


def make_case_level_split(case_ids, val_frac=0.02, test_frac=0.02, seed=0):
    """Split by case ID (not by view). Default ~960/20/20 on 1000 cases."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(case_ids))
    rng.shuffle(ids)

    n_val = max(1, int(len(ids) * val_frac))
    n_test = max(1, int(len(ids) * test_frac))

    val_ids = ids[:n_val].tolist()
    test_ids = ids[n_val:n_val + n_test].tolist()
    train_ids = ids[n_val + n_test:].tolist()

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def write_splits(splits: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
