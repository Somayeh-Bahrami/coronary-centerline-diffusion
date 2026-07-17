"""Step 1.1 — centerline + radius extraction from ImageCAS segmentations.
Runs locally on MacBook M4 (CPU-only, no GPU needed).

Classical skeletonization directly on ImageCAS's expert-annotated
segmentation mask — deterministic, no learned model needed since
ground-truth segmentation is already given.
"""

from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import convolve, distance_transform_edt
from skimage.morphology import skeletonize

# Branch/topology label values (5th column of the output array).
LABEL_ENDPOINT = 0     # 1 skeleton neighbor  -- tip of a vessel branch
LABEL_REGULAR = 1      # 2 skeleton neighbors -- ordinary point along a branch
LABEL_BIFURCATION = 2  # 3+ skeleton neighbors -- branch point (vessel splits)

_NEIGHBOR_KERNEL = np.ones((3, 3, 3))
_NEIGHBOR_KERNEL[1, 1, 1] = 0  # don't count the voxel itself


def _classify_topology(skeleton: np.ndarray) -> np.ndarray:
    """For each skeleton voxel, count its 26-connected skeleton neighbors
    and classify it as an endpoint, regular point, or bifurcation.

    Returns an array of labels aligned with np.argwhere(skeleton) order.
    """
    neighbor_count = convolve(skeleton.astype(np.uint8), _NEIGHBOR_KERNEL,
                              mode="constant", cval=0)
    counts_at_skeleton = neighbor_count[skeleton]

    labels = np.full(counts_at_skeleton.shape, LABEL_REGULAR, dtype=np.int64)
    labels[counts_at_skeleton <= 1] = LABEL_ENDPOINT
    labels[counts_at_skeleton >= 3] = LABEL_BIFURCATION
    return labels


def extract_centerline(volume: np.ndarray) -> np.ndarray:
    """Skeletonize a binary/segmented vessel volume, attach a radius at
    each centerline voxel via the distance transform, and label each
    point's local topology (endpoint / regular / bifurcation).

    Args:
        volume: binary (or thresholded) 3D array, vessel = True.

    Returns:
        (N, 5) array of (x, y, z, radius, branch_label) in voxel
        coordinates. branch_label is one of LABEL_ENDPOINT (0),
        LABEL_REGULAR (1), LABEL_BIFURCATION (2).
    """
    skeleton = skeletonize(volume)
    dist = distance_transform_edt(volume)
    coords = np.argwhere(skeleton)
    radii = dist[skeleton]
    branch_labels = _classify_topology(skeleton)
    return np.concatenate(
        [coords, radii[:, None], branch_labels[:, None]], axis=1
    )


def extract_case(label_path: Path) -> np.ndarray:
    """Load one ImageCAS `<case>.label.nii.gz` segmentation and extract
    its centerline."""
    seg = nib.load(label_path).get_fdata() > 0.5
    return extract_centerline(seg)


def extract_all(raw_dir: Path, out_dir: Path, case_ids=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(raw_dir.glob("*.label.nii.gz"))
    if case_ids:
        case_ids = set(case_ids)
        label_files = [f for f in label_files
                       if int(f.stem.split('.')[0]) in case_ids]

    for f in label_files:
        centerline = extract_case(f)
        out_path = out_dir / f"{f.stem.split('.')[0]}_centerline.npy"
        np.save(out_path, centerline)
        print(
            f"{f.stem}: {centerline.shape[0]} centerline points -> {out_path}")
