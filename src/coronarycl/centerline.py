"""Step 1.1 — centerline + radius extraction from ImageCAS segmentations.
Runs locally on M4 (CPU-only, no GPU needed).

Classical skeletonization directly on ImageCAS's expert-annotated
segmentation mask — deterministic, no learned model needed since
ground-truth segmentation is already given.
"""

from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def extract_centerline(volume: np.ndarray) -> np.ndarray:
    """Skeletonize a binary/segmented vessel volume and attach a radius
    at each centerline voxel via the distance transform.

    Args:
        volume: binary (or thresholded) 3D array, vessel = True.

    Returns:
        (N, 4) array of (x, y, z, radius) in voxel coordinates.
    """
    skeleton = skeletonize(volume)
    dist = distance_transform_edt(volume)
    coords = np.argwhere(skeleton)
    radii = dist[skeleton]
    return np.concatenate([coords, radii[:, None]], axis=1)


def extract_case(label_path: Path) -> np.ndarray:
    """Load one ImageCAS `<case>.label.nii.gz` segmentation and extract
    its centerline."""
    seg = nib.load(label_path).get_fdata() > 0.5
    return extract_centerline(seg)


def extract_all(raw_dir: Path, out_dir: Path, case_ids=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(raw_dir.glob("*.label.nii.gz"))
    if case_ids:
        label_files = [f for f in label_files if any(str(c) in f.stem for c in case_ids)]

    for f in label_files:
        centerline = extract_case(f)
        out_path = out_dir / f"{f.stem.split('.')[0]}_centerline.npy"
        np.save(out_path, centerline)
        print(f"{f.stem}: {centerline.shape[0]} centerline points -> {out_path}")
