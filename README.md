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
(`src/coronarycl/config.py`). DRR generation and full-scale training
require a CUDA GPU (Google Colab) — see **Model** below. For those
steps:

```bash
pip install -r requirements-colab.txt
```

## Dataset

See [DATASET.md](DATASET.md) for an overview of ImageCAS (1000 CCTA
volumes, expert-annotated segmentation masks) and the 3-step
preparation pipeline: 
(1) ground-truth centerline + radius extraction
via skeletonization.
(2) synthetic 2D projections via DeepDRR (Unberath et al., 2018) with DeepCA's (Wang et al., WACV 2025) motion-simulation protocol.
(3) case-level train/val/test split (960/20/20, following 3DGR-CAR's training-set scale).

```bash
python prepare_centerlines.py --config configs/default.yaml
python make_splits.py --config configs/default.yaml --n-cases 1000
```

DRR generation runs on Colab only — see
`notebooks/colab_drr_generation.ipynb`. `data/` is gitignored.

## Model

A conditional diffusion model (1D-UNet denoiser over centerline
nodes), following AortaDiff's (arXiv:2507.13404) centerline-diffusion
design, conditioned on both projections and their projection matrices
via cross-attention. A classical, non-learned epipolar-constraint
baseline (`src/coronarycl/models/baseline.py`) is implemented
alongside it to establish a reconstruction-quality floor.

```bash
python train.py --config configs/default.yaml           # full run — needs Colab
python train.py --config configs/default.yaml --quick-test   # local M4 sanity check
```

Initial hyperparameters follow AortaDiff's reported setup (Adam,
β₁=0.9/β₂=0.99, LR 1×10⁻³, T=1000); batch size and training length are
tuned empirically for this dataset's scale rather than copied
directly (AortaDiff trained on 18 cases with 3D-volume conditioning,
versus ~960 cases with 2D-projection conditioning here).

## Evaluation

Chamfer L2 distance and a threshold-based overlap metric Ot(d)
(following DeepCA's protocol), reported for the baseline and diffusion
model side by side, plus a stress-test subset (high foreshortening /
vessel overlap) and tube-surface visualizations for clinical review
(predicted vs. ground truth, including visible-stenosis cases).

```bash
python evaluate.py --pred outputs/pred_centerline.npy --gt data/processed/centerlines/case_0001_centerline.npy
python visualize_tube.py --centerline outputs/pred_centerline.npy --output tube.obj
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
