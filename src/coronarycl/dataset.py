"""Step 1.3 -- per-case packaging and PyTorch Dataset/DataLoader.
Runs locally on M4, no GPU needed.
"""

import json
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    voxel_to_mm,
    normalize_centerline,
    pad_centerline,
    normalize_image,
)

# global max centerline length across all 1000 cases (see EDA, Step 1.3)
MAX_LEN = 2884


def package_case(case_id, centerline_dir: Path, drr_dir: Path, raw_dir: Path,
                 norm_stats: dict, img_norm_stats: dict, out_dir: Path, split_name: str):
    """Combine one case's centerline + DRR projections into a single
    packaged .npz file: padded/normalized centerline + mask, normalized
    images, vessel masks, and projection matrices.
    """
    spacing = nib.load(
        str(raw_dir / f"{case_id}.label.nii.gz")).header.get_zooms()[:3]
    centerline_raw = np.load(centerline_dir / f"{case_id}_centerline.npy")
    centerline_mm = voxel_to_mm(centerline_raw, spacing)
    centerline_normed = normalize_centerline(centerline_mm, norm_stats)
    centerline_padded, centerline_mask = pad_centerline(
        centerline_normed, MAX_LEN)

    drr = np.load(
        drr_dir / f"case_{case_id}_projections.npz", allow_pickle=True)
    images_normed = normalize_image(
        drr["images"], img_norm_stats["clip_min"], img_norm_stats["clip_max"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{case_id}.npz",
        centerline=centerline_padded,                  # (MAX_LEN, 5) float32
        centerline_mask=centerline_mask,                # (MAX_LEN,) bool
        # (2, 512, 512), normalized [0,1]
        images=images_normed.astype(np.float32),
        vessel_masks=drr["masks"].astype(np.float32),    # (2, 512, 512)
        poses=drr["poses"].astype(np.float32),           # (2, 3, 4)
        split=split_name,
    )


def package_all(centerline_dir, drr_dir, raw_dir, splits_dir, out_dir):
    """Package every case in every split. Reads splits + normalization
    stats from splits_dir (produced by the EDA/normalization steps),
    writes one .npz per case to out_dir.
    """
    centerline_dir, drr_dir, raw_dir = Path(
        centerline_dir), Path(drr_dir), Path(raw_dir)
    splits_dir, out_dir = Path(splits_dir), Path(out_dir)

    with open(splits_dir / "case_splits.json") as f:
        splits = json.load(f)
    with open(splits_dir / "normalization_stats.json") as f:
        norm_stats = json.load(f)
    with open(splits_dir / "image_norm_stats.json") as f:
        img_norm_stats = json.load(f)

    for split_name, ids in splits.items():
        for cid in ids:
            package_case(cid, centerline_dir, drr_dir, raw_dir,
                         norm_stats, img_norm_stats, out_dir, split_name)
        print(f"{split_name}: packaged {len(ids)} cases")

    print("Total packaged files:", len(list(out_dir.glob("*.npz"))))


class CoronaryCenterlineDataset(Dataset):
    """Reads pre-packaged per-case .npz files (see package_all above)."""

    def __init__(self, packaged_dir, case_ids):
        self.packaged_dir = Path(packaged_dir)
        self.case_ids = case_ids

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        d = np.load(self.packaged_dir / f"{case_id}.npz", allow_pickle=True)

        return {
            # (MAX_LEN, 5)
            "centerline": torch.from_numpy(d["centerline"]).float(),
            # (MAX_LEN,)
            "centerline_mask": torch.from_numpy(d["centerline_mask"]).bool(),
            # (2, 512, 512)
            "images": torch.from_numpy(d["images"]).float(),
            # (2, 512, 512)
            "vessel_masks": torch.from_numpy(d["vessel_masks"]).float(),
            # (2, 3, 4)
            "poses": torch.from_numpy(d["poses"]).float(),
            "case_id": case_id,
        }
