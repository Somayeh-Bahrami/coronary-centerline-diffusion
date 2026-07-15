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

1. **Centerline + radius ground truth** — extracted locally from each
   `<case>.label.nii.gz` via 3D skeletonization
   (`prepare_centerlines.py`, Step 1.1). Output: one `(N, 4)`
   `(x, y, z, radius)` array per case in `data/processed/centerlines/`.
2. **Synthetic 2D X-ray projections (DRRs)** — generated on Colab from
   each `<case>.img.nii.gz` volume via cone-beam projection, with a
   simulated non-rigid motion perturbation between the two views
   (`notebooks/colab_drr_generation.ipynb`, Step 1.2). Output: 2
   projection images + their projection matrices per case in
   `data/processed/projections/`.

Both `data/raw/` and `data/processed/` are gitignored — regenerate
locally rather than committing.

## Splits

Case-level split (never split by view — both projections of one case
must stay in the same split, or the model leaks information across
train/val/test). Default ~960/20/20 on the full 1000 cases, generated
by `make_splits.py` (Step 1.3) into `data/splits/case_splits.json`.

## Class / severity balance

Not yet characterized for this project. TODO once centerline
extraction (Step 1.1) is complete: report the distribution of vessel
radius / stenosis severity across cases, since this affects whether
stress-test subsets (per `docs/work_breakdown.md` Step 3.1) are
representative.
