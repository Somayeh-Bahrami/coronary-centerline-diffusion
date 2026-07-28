import numpy as np
python
"""Step 1.3 -- preprocessing: coordinate normalization, padding, and
image intensity normalization. Runs locally on M4, no GPU needed.

All normalization statistics are computed from the TRAINING split only
(see docs/work_breakdown.md Step 1.3), then applied uniformly to
train/val/test -- this avoids leaking val/test information into the
transform.
"""


def voxel_to_mm(centerline_arr: np.ndarray, voxel_spacing) -> np.ndarray:
    """Convert (x,y,z) voxel indices to physical mm. Radius (col 3) is
    also converted, using the mean spacing as an isotropic
    approximation, since radius isn't a single-axis quantity. Topology
    label (col 4) passes through unchanged.
    """
    out = centerline_arr.copy()
    out[:, 0] *= voxel_spacing[0]
    out[:, 1] *= voxel_spacing[1]
    out[:, 2] *= voxel_spacing[2]
    out[:, 3] *= np.mean(voxel_spacing)
    return out


def compute_centerline_norm_stats(mm_centerlines: dict, train_ids: list) -> dict:
    """Compute coordinate/radius normalization stats from the training
    split only. mm_centerlines: {case_id: (N,5) array in physical mm}.
    """
    train_points = np.concatenate(
        [mm_centerlines[cid][:, :4] for cid in train_ids], axis=0
    )
    return {
        "coord_mean": train_points[:, :3].mean(axis=0).tolist(),
        "coord_std": train_points[:, :3].std(axis=0).tolist(),
        "radius_mean": float(train_points[:, 3].mean()),
        "radius_std": float(train_points[:, 3].std()),
    }


def normalize_centerline(mm_arr: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Apply train-derived normalization stats to one case's centerline."""
    out = mm_arr.copy()
    coord_mean = np.array(norm_stats["coord_mean"])
    coord_std = np.array(norm_stats["coord_std"])
    out[:, :3] = (out[:, :3] - coord_mean) / coord_std
    out[:, 3] = (out[:, 3] - norm_stats["radius_mean"]) / \
        norm_stats["radius_std"]
    # column 4 (topology label) untouched -- categorical, not continuous
    return out


def pad_centerline(arr: np.ndarray, max_len: int):
    """Pad a (N, 5) centerline array to (max_len, 5), with a boolean
    mask marking which rows are real data vs. padding.
    """
    n = arr.shape[0]
    if n > max_len:
        raise ValueError(f"Array length {n} exceeds max_len {max_len} -- "
                         f"raise max_len instead of truncating.")
    padded = np.zeros((max_len, arr.shape[1]), dtype=np.float32)
    mask = np.zeros(max_len, dtype=bool)
    padded[:n] = arr
    mask[:n] = True
    return padded, mask


def compute_image_norm_stats(sample_images: np.ndarray) -> dict:
    """Percentile-clipped normalization bounds, computed from a sample
    of training images. Percentile clipping (not plain min-max) is used
    since a small number of extreme-value outlier pixels otherwise skew
    the scale.
    """
    p1, p99 = np.percentile(sample_images, [1, 99])
    return {"clip_min": float(p1), "clip_max": float(p99)}


def normalize_image(img: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """Clip to [clip_min, clip_max] (from training data), then scale to [0,1]."""
    clipped = np.clip(img, clip_min, clip_max)
    return (clipped - clip_min) / (clip_max - clip_min)
