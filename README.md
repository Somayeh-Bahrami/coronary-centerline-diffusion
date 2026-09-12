# Diffusion-Based 3D Coronary Centerline Reconstruction from Sparse-View X-Ray Angiography

Reconstructing the 3D coronary artery centerline from 2 sparse,
non-simultaneous 2D X-ray angiography projections via a conditional
diffusion model, Phase 1 of a two-phase project toward real-time,
wire-free intraoperative hemodynamic assessment (FFR, WSS, blood
velocity).

CS 6999, Georgia Tech. Advisor: Prof. Bo Zhu.

## Proposal

See [proposal.pdf](Proposal.pdf) for the full project proposal
Step-by-step task tracking lives in [docs/work_breakdown.md](docs/work_breakdown.md).

## Setup

```bash
pip install -r requirements.txt
```

Runs on CUDA and CPU; device is auto-detected
(`src/coronarycl/config.py`). Dataset generation and full-scale training
require a CUDA GPU — both are run on Kaggle Notebooks (P100). TIGRE has no
pip package and must be built from source there:

```bash
git clone --depth 1 https://github.com/CERN/TIGRE.git
pip install ./TIGRE/Python
```

## Dataset

See [DATASET.md](DATASET.md) for an overview of ImageCAS (1000 CCTA
volumes, expert-annotated segmentation masks). Dataset v3 is built by a
single script, `build_dataset_v3.py`, which does every step in one
coordinate frame: RCA/LCA split, 96 mm crop, isotropic resample,
skeletonization (centerline + radius in mm), binary vessel-mask projection
through DeepCA's two-view geometry via TIGRE (Biguri et al., 2016), DLT pose
calibration, and a hard QC gate.

```bash
python build_dataset_v3.py --raw_dir <imagecas_raw> --out_dir ./dataset_v3_1 --n 20
```

It writes one `.npz` per coronary system (`<patient>_LCA.npz` /
`<patient>_RCA.npz`), a patient-level 80/10/10 split (both vessels of a
patient stay in the same split), and train-only normalization stats. Read it
with `src/coronarycl/dataset_v3_1.py`. Run the 20-patient pilot and check the
gate before building all 1000. `data/` is gitignored.

**Poses:** `poses` is the NOMINAL scanner geometry with the simulated motion
removed, following DeepCA's protocol — so projecting the ground-truth
centerline through `poses[1]` does *not* land on `images[1]`. That mismatch is
the motion-compensation task. `poses_render` carries the motion and is for
validation and visualization only.

## Model

A conditional diffusion model (1D-UNet denoiser over centerline
nodes), following AortaDiff's (arXiv:2507.13404) centerline-diffusion
design, conditioned on both projections and their projection matrices
via cross-attention. A classical, non-learned epipolar-constraint
baseline (`src/coronarycl/models/baseline.py`) is implemented
alongside it to establish a reconstruction-quality floor.

```bash
python train.py --config configs/h384_200k.yaml         # clean 200k run — CUDA/BF16
python train.py --config configs/default.yaml --quick-test   # local M4 sanity check
```

Initial hyperparameters follow AortaDiff's reported setup (Adam,
β₁=0.9/β₂=0.99, LR 1×10⁻³, T=1000); batch size and training length are
tuned empirically for this dataset's scale rather than copied
directly (AortaDiff trained on 18 cases with 3D-volume conditioning,
versus ~800 training patients with 2D-projection conditioning here).

## Evaluation

Sampler/checkpoint settings are selected on VAL only. Evaluation reports the
project's summed, unsquared Chamfer-L2 convention, HD95, Ot(1/2/5 mm),
given-topology continuity and tree length, and crop violations. The canonical
DDIM sampler is padding-aware and uses physical per-channel bounds. TEST stays
locked until the complete protocol is frozen.

```bash
python ddim_eval.py --ckpt outputs/h384_200k/checkpoints/best.pt \
  --data data/processed/ds105_full --split val --steps 50 --guidance 1.0 \
  --out outputs/h384_200k/val_s50_g1.json --save-pred outputs/h384_200k/val_s50_g1.npz
python cond_sensitivity.py --data data/processed/ds105_full \
  --ckpt h384=outputs/h384_200k/checkpoints/best.pt \
  --out-dir outputs/h384_200k/sensitivity_joint --shuffle-mode joint
```

## Fine-tuning

If real (non-simultaneous) ICA projections become available, fine-tune
the trained checkpoint on them to close the DRR-to-real sim-to-real
gap:

```bash
python finetune.py --checkpoint outputs/model.pt --real-data-dir data/real_ica/
```

## Tests

```bash
python -m pytest tests/
```

## References

See [references.bib](references.bib).

## Author

Somayeh Bahrami — advised by Prof. Bo Zhu, Georgia Tech.
