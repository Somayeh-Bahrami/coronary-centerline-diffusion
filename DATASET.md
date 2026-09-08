# Dataset

## ImageCAS

[ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT)
(Zeng et al., *Computerized Medical Imaging and Graphics*, 2023) is a
large-scale public benchmark of coronary CT angiography (CCTA) volumes,
each with an expert-annotated coronary artery segmentation mask.

- **Size:** 1000 3D CCTA volumes, acquired on a Siemens 128-slice
  dual-source CT scanner.
- **Per-case files:** `<case>.img.nii.gz` (the CT volume) and
  `<case>.label.nii.gz` (the binary coronary artery segmentation).
- **Resolution:** 512×512×(206-275) voxels; in-plane resolution
  0.29-0.43 mm², inter-slice spacing 0.25-0.45 mm.
- **Access:** distributed under a data-use agreement via the official
  repo above (Google Drive / Baidu links) — not an anonymous scripted
  download. See `scripts/download_imagecas.py` for where to plug in
  your authorized access method once obtained.

## What we derive from it (this project doesn't use ImageCAS directly)

This project needs 2D projections + 3D centerline ground truth, not
the raw CCTA volumes themselves:

Both steps below are performed by `build_dataset_v3.py` in a single pass, on
one isotropic grid, so mask / centerline / radius / projection / pose share one
coordinate frame by construction. (Earlier versions split this across
`prepare_centerlines.py` and a separate `src/coronarycl/drr.py` projection
module; both have been removed — the two-frame design was the source of the
v1/v2 projection-consistency failures.)

1. **Centerline + radius ground truth** — each `<case>.label.nii.gz` is split
   into coronary systems (3D connected components, side from the NIfTI
   affine), cropped to a 96 mm cube, resampled to an isotropic grid, then
   skeletonized. Radius comes from `distance_transform_edt(sampling=iso)`, so
   it is in millimetres. Output per sample: an `(N, 5)`
   `(x, y, z, radius, topology)` array in raw mm, DFS-ordered.
2. **Synthetic 2D X-ray projections** — 2 binary vessel silhouettes per sample,
   projected with TIGRE (Biguri et al., 2016) through DeepCA's two-view
   geometry (512² detector, ~0.278 mm pixels), with DeepCA's rigid motion
   perturbation (±10° rotation, ±8 mm two-axis translation) applied to view 2
   only. Each view carries two 3×4 projection matrices: `poses` (nominal
   scanner geometry, motion removed — the model input) and `poses_render`
   (motion included — validation only).

Both `data/raw/` and `data/processed/` are gitignored — regenerate
locally rather than committing.

## Splits

Patient-level split (never split by view or by vessel — both projections and
both coronary systems of one patient stay in the same split, or the model leaks
information across train/val/test). Default 80/10/10 over the available
patients, written by `build_dataset_v3.py` to `case_splits_v3.json` in the
output directory alongside the samples.

## Class / severity balance

Not yet characterized for this project. TODO once centerline
extraction (Step 1.1) is complete: report the distribution of vessel
radius / stenosis severity across cases, since this affects whether
stress-test subsets (per `docs/work_breakdown.md` Step 3.1) are
representative.
