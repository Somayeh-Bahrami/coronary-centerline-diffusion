# Counting Is Not Connecting

## Topology-aware evaluation of two-view coronary centerline diffusion

This repository contains the data-generation, conditional-diffusion, and
topology-aware evaluation code for reconstructing 3D coronary centerlines
from two synthetic X-ray angiographic projections.

The project began as a reconstruction stage for downstream coronary
hemodynamics. During validation, we found that good point-set accuracy does
not necessarily produce a connected vascular tree. The completed study
therefore focuses on this mismatch: it measures geometric accuracy and graph
connectivity separately and tests two controlled representation changes.

The manuscript is currently being finalized and has **not yet been submitted
or accepted**. All reported model comparisons are validation-set analyses;
the held-out test set has not been accessed.

## Study overview

### Data

The source data are the expert coronary-artery segmentation masks from
[ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT).
`build_dataset_v3.py` converts the masks into paired reconstruction samples:

- separate LCA and RCA coronary systems;
- a 105 mm crop and isotropic resampling;
- 3D centerline coordinates and physical radius in millimetres;
- two 512 x 512 binary vessel projections rendered with TIGRE;
- nominal projection matrices supplied to the model and motion-aware matrices
  retained only for validation;
- patient-level train/validation/test assignments; and
- strict geometry, projection, coverage, connectivity, padding, and radius
  quality-control gates.

The final packaged dataset contains 1,694 vessel samples:

| Split | Vessel samples |
|---|---:|
| Train | 1,350 |
| Validation | 177 |
| Test | 167 |

The patient assignment is 800/100/100. Both coronary systems from one patient
remain in the same split. Normalization statistics are calculated from the
training split only.

See [DATASET.md](DATASET.md) for the complete data description and access
requirements. ImageCAS data are not redistributed by this repository.

### Model

The model is a conditional diffusion model over ordered centerline nodes
`(x, y, z, radius)`. Its denoiser is a padding-aware 1D convolutional
encoder-decoder with bottleneck self-attention and **no U-Net skip
connections**. Two vessel projections and their nominal projection matrices
are encoded as conditioning tokens and incorporated through positional-query
cross-attention. The final experiments use epsilon prediction, self-
conditioning, classifier-free conditioning dropout, and a hidden dimension of
384.

### Experimental arms

All principal comparisons use the same 50,000-step training budget and the
same validation protocol.

1. **DFS ordering:** the original depth-first linearization of the coronary
   tree.
2. **Optimal linear ordering:** an exact constructive ordering that minimizes
   the number of graph edges that cannot be adjacent in a one-dimensional
   sequence.
3. **Branch-token representation:** the DFS node ordering augmented with four
   normalized per-node topology channels — branch id, parent branch id, parent
   attachment index, and within-branch index — giving an eight-channel node
   record `(x, y, z, radius, branch_id, parent_branch_id,
   parent_attach_index, within_branch_index)`. The node ordering itself is
   unchanged; the topology fields are additional generation targets, decoded
   at inference by a fixed deterministic decoder that never receives
   ground-truth edges. See
   [STAGE2_FROZEN_REPRESENTATION.md](STAGE2_FROZEN_REPRESENTATION.md) and
   `src/coronarycl/branch_token_tree.py`.

Ordering and branch-token experiments were preregistered in
[experiment_protocol.md](experiment_protocol.md) and
[STAGE2_FROZEN_REPRESENTATION.md](STAGE2_FROZEN_REPRESENTATION.md).

### Evaluation

Sampler and protocol choices are frozen on validation data. Evaluation uses
100-step DDIM sampling, guidance 2.0, and five fixed sampling seeds.

Point-set metrics:

- summed, unsquared symmetric Chamfer-L2 in millimetres;
- HD95;
- overlap within 1, 2, and 5 mm; and
- radius error and correlation.

Topology-aware metrics, computed using the supplied ground-truth graph:

- fraction of broken edges at a multiple of ground-truth edge length;
- largest connected-component fraction (LCC);
- reconstructed-to-ground-truth tree-length ratio;
- severed mass; and
- crop violations.

The topology is supplied for analysis; this repository does not claim
autonomous topology recovery. Sampling also uses the ground-truth point count.

## Main finding

Point-set accuracy and vascular connectivity dissociate, and they do so in
both directions.

Optimal linear ordering reduces the broken-edge fraction and tree-length
inflation, yet the largest connected-component fraction **falls** rather than
improves, from 25.7% to 18.6% — a paired per-patient change of −6.5 to
−7.6 pp measured against each of three DFS training seeds, with all three
confidence intervals excluding zero. Fewer broken edges therefore did not
produce a better-connected vessel: the remaining breaks fall closer to the
root, so each one severs a larger share of the tree (0.156 versus 0.105, about
50% more). The arm does not pass the preregistered advancement rule.

The branch-token representation moves in the opposite direction on geometry.
Chamfer changed by −0.4 to −3.1 mm against the three DFS seeds, inside the
seed-to-seed range, while connectivity collapsed: the broken-edge fraction rose
to 38.5%, LCC fell to 14.0%, and the tree-length ratio rose from 3.26 to 8.21.
This result applies to the continuous branch-token encoding specified here, not
to topology-aware methods in general.

Both LCC effects exceed the DFS training-seed spread by more than tenfold,
whereas the sign of the Chamfer difference depends on which seed is used as the
comparator. Two interventions with opposite effects on the point metric thus
support the same conclusion: plausible point clouds are not sufficient for a
centerline intended for graph-dependent hemodynamic analysis, and Chamfer
distance alone cannot establish that a reconstructed tree is usable downstream.

The appropriate next step is a graph-native generator or an explicit,
validated topology-recovery/reconnection stage before downstream WSS or FFR
estimation. This repository does **not** claim that its current predictions
are ready for clinical hemodynamic use.

## Repository layout

```text
build_dataset_v3.py                 Dataset construction and hard QC
train.py                            Training entry point
ddim_eval.py                        DDIM and topology-aware validation
cond_sensitivity.py                 Patient-aware conditioning sensitivity
visualize_predictions.py            Equal-axis prediction visualization

src/coronarycl/                     Dataset, model, trainer, sampler, metrics
scripts/build_edge_cache.py         Canonical graph-edge cache builder
scripts/leaf_bound_check.py         Exact linear-ordering bound and permutation
scripts/build_optimal_order_dataset.py
scripts/verify_optimal_order_dataset.py
scripts/build_branch_token_dataset.py

configs/h384_ordering_50k_dfs.yaml
configs/h384_ordering_50k_optimal.yaml
configs/h384_branch_token_stage2.yaml
tests/                              Unit and integration tests
```

Generated datasets, edge caches, checkpoints, predictions, and analysis
outputs should be stored outside Git or in ignored artifact directories.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the frozen validation-study environment, use
`requirements-repro.txt`. Select the CPU or CUDA PyTorch 2.11.0 wheel that
matches the target system.

Dataset rendering additionally requires a CUDA-compatible installation of
[TIGRE](https://github.com/CERN/TIGRE). TIGRE is not installed by
`requirements.txt`.

## Dataset construction

The paper dataset and current builder default use a **105 mm** crop. The crop
is passed explicitly below so the command records the protocol clearly:

```bash
python build_dataset_v3.py \
  --raw_dir /path/to/imagecas \
  --out_dir data/processed/ds105_full \
  --n 1000 \
  --crop_mm 105 \
  --iso auto \
  --seed 0
```

Do not train unless the final hard gate passes. Build the canonical edge cache
outside the dataset directory:

```bash
python scripts/build_edge_cache.py \
  --data data/processed/ds105_full \
  --out artifacts/dfs_edge_cache
```

## Representation datasets

### Exact optimal ordering

```bash
python scripts/leaf_bound_check.py \
  --data data/processed/ds105_full \
  --edge-cache artifacts/dfs_edge_cache \
  --out artifacts/ordering_analysis_v1

python scripts/build_optimal_order_dataset.py \
  --source data/processed/ds105_full \
  --permutations artifacts/ordering_analysis_v1/optimal_ordering_v1.npz \
  --source-edge-cache artifacts/dfs_edge_cache \
  --out data/processed/ds105_optimal_v1 \
  --edge-out artifacts/optimal_edge_cache

python scripts/verify_optimal_order_dataset.py --help
```

Run the verification command with the corresponding source, derived-dataset,
permutation, and edge-cache paths before training.

### Branch-token representation

```bash
python scripts/build_branch_token_dataset.py \
  --source data/processed/ds105_full \
  --source-edge-cache artifacts/dfs_edge_cache \
  --out data/processed/ds105_branch_token_v1 \
  --edge-out artifacts/branch_token_edge_cache
```

## Training

Run a quick test first, using a separate checkpoint directory if a production
run already exists:

```bash
python train.py --config configs/h384_ordering_50k_dfs.yaml --quick-test
```

The three frozen 50k arms are launched with:

```bash
python train.py --config configs/h384_ordering_50k_dfs.yaml
python train.py --config configs/h384_ordering_50k_optimal.yaml
python train.py --config configs/h384_branch_token_stage2.yaml
```

Before running, update only environment-specific dataset, checkpoint, and edge-
cache paths. Do not change the frozen scientific hyperparameters when
reproducing the comparison.

## Validation evaluation

Example for the DFS arm:

```bash
python ddim_eval.py \
  --ckpt /path/to/dfs_checkpoint.pt \
  --data data/processed/ds105_full \
  --edge-cache artifacts/dfs_edge_cache \
  --split val \
  --steps 100 \
  --guidance 2.0 \
  --seeds 104729,130363,155921,181081,205019 \
  --bounds physical \
  --precision fp32 \
  --out artifacts/dfs_val.json \
  --save-pred artifacts/dfs_val_predictions.npz
```

`ddim_eval.py` refuses test-set evaluation unless `--allow-test` is explicitly
provided. Do not use that option while developing, selecting checkpoints, or
tuning the protocol.

Conditioning sensitivity can be measured in joint image-and-pose mode and in
image-only mode:

```bash
python cond_sensitivity.py \
  --data data/processed/ds105_full \
  --ckpt dfs=/path/to/dfs_checkpoint.pt \
  --out-dir artifacts/sensitivity_joint \
  --seeds 104729,130363,155921,181081,205019 \
  --shuffle-mode joint \
  --precision fp32

python cond_sensitivity.py \
  --data data/processed/ds105_full \
  --ckpt dfs=/path/to/dfs_checkpoint.pt \
  --out-dir artifacts/sensitivity_images \
  --seeds 104729,130363,155921,181081,205019 \
  --shuffle-mode images \
  --precision fp32
```

## Tests

```bash
python -m pytest -q
```

The current audited repository state contains 81 passing tests.

## Reproducibility and release status

- Patient-level splitting prevents cross-patient leakage.
- Normalization statistics are training-only.
- Checkpoints record the prediction parameterization and run signature.
- Evaluation records dataset, edge-cache, checkpoint, sampler, and seed
  metadata.
- SHA-256 hashes for frozen external artifacts are recorded in
  [`manifests/final_artifacts.sha256`](manifests/final_artifacts.sha256).
- Per-sample validation records and paired per-patient differences for the
  three 50k arms are committed in
  [`results/validation_50k_v1/`](results/validation_50k_v1/). Running
  `python results/validation_50k_v1/verify_table1.py` re-derives every paired
  difference from the per-sample files. The DFS records there are one training
  seed; the remaining two seed replicates behind the reported mean ± SD are
  archived outside Git and listed in the artifact manifest.
- The final manuscript, checkpoint release, and permanent artifact links will
  be added after the submission package is frozen.
- The test split remains untouched at the current project stage.

## License and citation

The source code is released under the [MIT License](LICENSE). Citation
metadata are provided in [CITATION.cff](CITATION.cff). The manuscript citation
will be added after submission.

## Authors

Somayeh Bahrami, advised by Prof. Bo Zhu, Georgia Institute of Technology.