"""v3.2-compatible dataset loader.

src/coronarycl/dataset.py CANNOT read this builder's output.  Four
incompatibilities, in increasing order of how quietly they fail:

  1. FILENAMES.  It reads f"{case_id}.npz"; the builder writes
     f"{patient_id}_{LCA|RCA}.npz" -- one sample per coronary system.
     -> KeyError/FileNotFoundError. Loud.

  2. vessel_masks.  __getitem__ requires that key; v3.2 does not write it
     (the projected label mask is no longer stored separately -- `images`
     IS the binary vessel silhouette).
     -> KeyError. Loud.

  3. NORMALISATION.  package_case() ran voxel_to_mm() then
     normalize_centerline() then pad_centerline().  v3.2 stores RAW mm,
     already padded, and writes the stats to norm_stats_v3.json for the
     loader to apply.  Running the old path would (a) double-convert via
     voxel_to_mm and (b) feed unnormalised millimetres to the model.
     -> SILENT. Loss scale explodes; nothing raises.

  4. `poses` SEMANTICS.  Same key, different meaning: v3.2 `poses` is the
     NOMINAL scanner geometry with the simulated motion removed, so
     projecting the GT centerline through poses[1] does NOT land on
     images[1].  That mismatch is the motion-compensation task.  Pre-v3.1
     builds baked the motion into `poses`, which made the task trivial.
     -> SILENT and experiment-invalidating. This is why every sample carries
        builder_version and why this loader asserts on it.

PADDING TRAP: normalize_centerline() rescales columns 0-3 of every row it is
given, including padded rows.  The old pipeline normalised BEFORE padding, so
padded rows were exactly 0.  Here the array arrives already padded, so this
loader normalises ONLY the rows flagged by centerline_mask and leaves padded
rows at exactly 0.  Any masked loss is unaffected either way, but code that
assumes "padding == 0" would silently break if you normalised the whole array.

Usage:
    from dataset_v3_1 import CoronaryCenterlineDatasetV31, list_samples
    ds = CoronaryCenterlineDatasetV31("./dataset_v3_1", split="train")
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

EXPECTED_BUILDER_MAJOR = 3          # accepts 3.x, rejects pre-versioned output


def list_samples(packaged_dir, split=None):
    """Every sample .npz in `packaged_dir`, optionally filtered to one split.

    Reads the split from each file rather than from case_splits_v3.json, so a
    sample can never be silently placed in the wrong split by a stale json.
    """
    packaged_dir = Path(packaged_dir)
    out = []
    for f in sorted(packaged_dir.glob("*.npz")):
        with np.load(f, allow_pickle=True) as z:
            if split is None or str(z["split"]) == split:
                out.append(f.name[:-4])
    if not out:
        raise FileNotFoundError(
            f"no samples in {packaged_dir}" + (f" for split={split!r}" if split else ""))
    return out


class CoronaryCenterlineDatasetV31(Dataset):
    """Reads per-sample .npz written by build_dataset_v3.py (builder >= 3.1).

    Returns per item:
        images            (2, 512, 512) float32, binary vessel silhouettes 0/1
        centerline        (max_points, 5) float32 -- normalised x,y,z,radius
                          + RAW topology label in column 4; padded rows are 0
        centerline_mask   (max_points,) bool
        poses             (2, 3, 4) float32  NOMINAL scanner geometry  <- model input
        poses_render      (2, 3, 4) float32  motion-carrying           <- eval/QC ONLY
        patient_id, vessel, iso_mm, n_points

    `normalize=False` returns raw millimetres, which is what you want for
    computing Chamfer distance in physical units at evaluation time.
    """

    def __init__(self, packaged_dir, split=None, sample_ids=None,
                 norm_stats=None, normalize=True, return_render_poses=True):
        self.dir = Path(packaged_dir)
        self.ids = sample_ids if sample_ids is not None else list_samples(self.dir, split)
        self.normalize = normalize
        self.return_render_poses = return_render_poses
        if normalize:
            if norm_stats is None:
                p = self.dir / "norm_stats_v3.json"
                if not p.exists():
                    raise FileNotFoundError(
                        f"{p} missing -- v3.2 stores RAW mm and needs these stats. "
                        f"Pass normalize=False only if you intend raw millimetres.")
                norm_stats = json.load(open(p))
            self.coord_mean = np.array(norm_stats["coord_mean"], np.float32)
            self.coord_std = np.array(norm_stats["coord_std"], np.float32)
            self.radius_mean = float(norm_stats["radius_mean"])
            self.radius_std = float(norm_stats["radius_std"])
        self._check_version(self.ids[0])

    def _check_version(self, sid):
        with np.load(self.dir / f"{sid}.npz", allow_pickle=True) as z:
            if "builder_version" not in z:
                raise RuntimeError(
                    f"{sid}.npz has no builder_version -- it predates v3.1, so its "
                    f"`poses` almost certainly carry the simulated motion. Training on "
                    f"it removes the motion-compensation task. Rebuild with "
                    f"build_dataset_v3.py (v3.2+).")
            ver = str(z["builder_version"])
            if int(ver.split(".")[0]) != EXPECTED_BUILDER_MAJOR:
                raise RuntimeError(f"builder_version {ver} not supported by this loader")
            self.builder_version = ver

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        with np.load(self.dir / f"{sid}.npz", allow_pickle=True) as d:
            cl = d["centerline"].astype(np.float32)          # RAW mm, already padded
            mask = d["centerline_mask"].astype(bool)
            if self.normalize:
                # Only the VALID rows -- see the PADDING TRAP note above.
                v = cl[mask]
                v[:, :3] = (v[:, :3] - self.coord_mean) / self.coord_std
                v[:, 3] = (v[:, 3] - self.radius_mean) / self.radius_std
                cl = np.zeros_like(cl)
                cl[mask] = v
            item = {
                "images": torch.from_numpy(d["images"].astype(np.float32)),
                "centerline": torch.from_numpy(cl),
                "centerline_mask": torch.from_numpy(mask),
                "poses": torch.from_numpy(d["poses"].astype(np.float32)),
                "patient_id": int(d["patient_id"]),
                "vessel": str(d["vessel"]),
                "iso_mm": float(d["iso_mm"]),
                "n_points": int(d["n_points"]),
                "sample": sid,
            }
            if self.return_render_poses:
                # QC / visualisation only. Feeding this to the model tells it
                # the exact synthetic motion.
                item["poses_render"] = torch.from_numpy(d["poses_render"].astype(np.float32))
        return item

    def denormalize(self, cl_normed):
        """Normalised (…,5) -> millimetres. Use before any physical-unit metric."""
        out = np.array(cl_normed, np.float32, copy=True)
        out[..., :3] = out[..., :3] * self.coord_std + self.coord_mean
        out[..., 3] = out[..., 3] * self.radius_std + self.radius_mean
        return out
