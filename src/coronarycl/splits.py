"""Step 1.3 — train/val/test split. Runs locally on M4 (no GPU needed).

Case-level split ONLY — never split by view, both views of a case must
stay together to avoid leakage (follows 3DGR-CAR's, MICCAI 2024, split
logic on the same ImageCAS lineage).
"""

import json
from pathlib import Path

import numpy as np


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
