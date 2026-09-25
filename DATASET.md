# Dataset

## Source data: ImageCAS

[ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT)
contains 1,000 coronary CT angiography volumes with expert coronary-artery
segmentation masks (Zeng et al., *Computerized Medical Imaging and Graphics*,
2023).

The original ImageCAS data are subject to their own access terms and are not
redistributed by this repository. Obtain authorized access from the official
dataset provider and retain the original `<case>.img.nii.gz` and
`<case>.label.nii.gz` names.

## Derived reconstruction dataset

Builder v3.4 creates a controlled synthetic benchmark for two-view 3D
centerline reconstruction in one physical coordinate frame:

1. load and orient each expert segmentation using the NIfTI affine;
2. identify LCA and RCA connected components without silently discarding
   additional valid components;
3. crop each coronary system to a **105 mm** cube, shifting at image borders;
4. resample to a single isotropic grid;
5. skeletonize and calculate physical radius with an anisotropy-aware EDT;
6. order the centerline as a depth-first tree traversal;
7. render two 512 x 512 binary vessel projections using TIGRE and DeepCA-style
   geometry;
8. apply the view-2 rigid-motion perturbation on the intended axes;
9. estimate and validate projection matrices with held-out DLT markers;
10. apply strict per-sample and per-view quality gates; and
11. package samples, patient splits, and training-only normalization.

The final DFS-ordered dataset contains 1,694 vessel samples:

| Split | Samples |
|---|---:|
| Train | 1,350 |
| Validation | 177 |
| Test | 167 |

Patients are assigned 800/100/100 to train/validation/test before vessel-level
acceptance. Both coronary systems from a patient always share the same split.
The final validation analysis contains 177 vessels from 96 represented
patients. The held-out test samples have not been accessed for model selection
or reporting.

## Sample format

Each `<patient>_LCA.npz` or `<patient>_RCA.npz` contains:

- two binary projection images;
- nominal 3 x 4 projection matrices used as model conditioning;
- motion-aware render matrices retained for validation only;
- a padded centerline and validity mask;
- centerline columns `(x, y, z, radius, topology)`;
- sample/patient identity and vessel side; and
- geometry and builder metadata.

Coordinates and radius are in millimetres. Coordinates are relative to the
crop isocentre. Padding rows are zero and excluded through the validity mask.

`poses` is the nominal scanner geometry with simulated motion removed.
`poses_render` is the geometry actually used to render the motion-perturbed
projection and must never be supplied to the reconstruction model.

## Exact build command

The final dataset and current builder default use a 105 mm crop. It is passed
explicitly here so the build command records the protocol clearly:

```bash
python build_dataset_v3.py \
  --raw_dir /path/to/imagecas \
  --out_dir data/processed/ds105_full \
  --n 1000 \
  --crop_mm 105 \
  --iso auto \
  --seed 0
```

Do not train unless the builder prints `GATE PASSED`. Oversized centerlines,
failed geometry, split skeletons, insufficient detector coverage, invalid
radii, and other hard-gate failures are rejected rather than truncated or
silently retained.

The builder writes one NPZ per accepted coronary system plus:

- `case_splits_v3.json`;
- `norm_stats_v3.json`; and
- `pilot_report_v3.json`.

Normalization statistics are calculated exclusively from valid centerline
points belonging to training patients.

## Canonical graph edges

Topology-aware evaluation never infers edges from predicted coordinates. A
canonical edge cache is generated from the packaged ground-truth trees and
verified against dataset fingerprints:

```bash
python scripts/build_edge_cache.py \
  --data data/processed/ds105_full \
  --out artifacts/dfs_edge_cache
```

The output must be separate from the dataset. Generated caches are
reproducible artifacts and are intentionally excluded from Git.

## Derived representation datasets

Two controlled representation variants were derived without changing images,
geometry, splits, normalization, or anatomical point values.

### Exact optimal linear ordering

`scripts/leaf_bound_check.py` computes an exact constructive permutation that
maximizes graph edges represented by consecutive sequence positions.
`scripts/build_optimal_order_dataset.py` applies it to all five centerline
columns and remaps the canonical graph. The independent verification script
checks every sample, all metadata, graph fingerprints, and source JSON files.

### Branch-token representation

`scripts/build_branch_token_dataset.py` produces an edge-balanced traversal
with explicit branch-return tokens. This is the single preregistered tree-aware
representation tested after optimal ordering failed its advancement rule.

These variants represent the same anatomy; they are not additional patients
or independent datasets.

## Integrity and leakage controls

The final pipeline enforces:

- patient-level splitting;
- training-only normalization;
- strict per-view projection consistency;
- finite held-out DLT error below the configured limit;
- complete point-count preservation (no downsampling or truncation);
- connected ground-truth skeletons;
- finite positive physical radii;
- detector and mask-retention coverage;
- motion reaching the renderer;
- consistent padding and masks; and
- fingerprints linking derived datasets and edge caches to their sources.

## Reproducibility hashes

Hashes for final datasets, edge caches, checkpoints, and evidence packages are
recorded in [`manifests/final_artifacts.sha256`](manifests/final_artifacts.sha256).
The large artifacts themselves are not committed to Git.

## Limitations

- Projections are synthetic binary vessel silhouettes derived from CCTA, not
  clinical invasive angiograms.
- Evaluation supplies the ground-truth point count and graph topology.
- Connectivity metrics test whether predicted nodes preserve the supplied
  tree; the model does not recover topology autonomously.
- Current predictions are not claimed to be ready for clinical WSS or FFR.
  Graph-native generation or explicit validated reconnection is required
  before returning to that downstream objective.
