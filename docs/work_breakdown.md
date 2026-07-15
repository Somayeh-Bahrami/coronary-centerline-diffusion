# Phase 1 Work Step Breakdown

Sparse-view coronary centerline reconstruction — 2-view DRR input.
Same content as the Google Doc shared with Prof. Bo Zhu.

Infrastructure note: working machine is a MacBook Pro (Apple Silicon M4,
no CUDA). Each sub-step states whether it runs locally or needs a CUDA
GPU (Google Colab).

## Step 1: Data Preparation

### 1.1 [SMALL] [Deps: NONE] Dataset acquisition & centerline ground-truth extraction
- Download ImageCAS (1000 CCTA volumes, public).
- Extract 3D centerline + radius profile per case via classical
  skeletonization (scikit-image) directly on the provided segmentation
  mask — deterministic, no learned model needed since ground-truth
  segmentation is already given.

**DoD:** For all 1000 cases, a saved (N×4) array per case —
(x, y, z, radius) — plus a branch/topology label; spot-checked visually
on 10 random cases against the CCTA volume.

**How I plan to do it:** Runs fully on my M4 locally — CPU-only, no GPU
needed. scikit-image `skeletonize_3d` + radius estimation via distance
transform. Implemented in `src/coronarycl/centerline.py`, driven by
`prepare_centerlines.py`.

### 1.2 [MEDIUM] [Deps: 1.1] DRR generation — 2 views, correct camera geometry + motion simulation
- Replace the orthographic-rotation+MIP script (confirmed invalid — all
  8 prior views were the same projection axis, verified by >99.9%
  correlation after re-rotation) with a true perspective/cone-beam
  projector.
- Generate 2 non-simultaneous projections per case, with an independent
  small rigid perturbation (±10° rotation, ±8mm translation) applied to
  the second view only, following DeepCA's (Wang et al., WACV 2025)
  exact motion-simulation protocol.
- Save the projection matrix (source/detector pose) alongside every
  image — required as model conditioning input.

**DoD:** 8000→2000 projections regenerated (2 views × 1000 cases) with
paired projection-matrix files; verified NOT to be in-plane rotations
of one view.

**How I plan to do it:** DeepDRR / TIGRE-based cone-beam projection
requires CUDA — cannot run on M4 locally. Plan: run this step on Google
Colab (T4 GPU), then download projections + projection matrices to
laptop for everything downstream. Implemented in
`src/coronarycl/drr.py`, driven by
`notebooks/colab_drr_generation.ipynb`.

### 1.3 [SMALL] [Deps: 1.1, 1.2] Train / val / test split & data packaging
- Case-level split (never split by view) to avoid leakage, following
  3DGR-CAR's (MICCAI 2024) split logic on the same dataset.
- Package into final tensors (images + projection matrices + centerline
  ground truth).

**DoD:** Fixed train/val/test case-ID lists saved (e.g. ~960/20/20
split); one packaged dataset file per split, loadable in one line.

**How I plan to do it:** Runs fully on my M4 locally — no GPU needed,
just array bookkeeping. Implemented in `src/coronarycl/splits.py`,
driven by `make_splits.py`.

## Step 2: 3D Centerline Reconstruction Model

### 2.1 [MEDIUM] [Deps: 1.3] Deterministic baseline reproduction
- Reproduce a simple deterministic baseline (e.g. epipolar-constraint
  centerline matching) to get a floor number before the generative
  model.

**DoD:** Baseline Chamfer L2 distance reported on the val set.

**How I plan to do it:** Prototype and run at small scale on my M4
(CPU); if too slow, move full run to Colab. Implemented in
`src/coronarycl/models/baseline.py`.

### 2.2 [LARGE] [Deps: 1.3] Conditional diffusion model — design & implementation
- Centerline-native diffusion architecture (1D-UNet or GNN denoiser
  over centerline nodes), following AortaDiff's (2025, arXiv:2507.13404)
  centerline-diffusion design.
- Condition on both 2D projections + their projection matrices via
  cross-attention, so the model can reason about epipolar geometry
  between the two views.

**DoD:** Model forward/backward pass runs end-to-end on a small dummy
batch without errors; architecture diagram + short design doc.

**How I plan to do it:** Write and debug the architecture locally on my
M4 using the PyTorch MPS backend (small batch, CPU/MPS fallback) — this
part does not require CUDA. Full-scale training moves to Step 2.3.
Implemented in `src/coronarycl/models/diffusion.py`.

### 2.3 [MEDIUM] [Deps: 2.2] Model training
- Train on the full packaged dataset from Step 1.3.

**DoD:** Training curve (loss vs. epoch) saved; checkpoint that reaches
at least baseline (2.1) performance on val set.

**How I plan to do it:** Full-dataset training needs a CUDA GPU for
reasonable speed — plan to use Google Colab (or a rented GPU if Colab's
free tier is insufficient). M4 used only for local unit tests on a tiny
subset before committing a full run. Implemented in
`src/coronarycl/trainer.py`, driven by `train.py`.

## Step 3: Evaluation & Iteration

### 3.1 [SMALL] [Deps: 2.3] Quantitative evaluation
- Chamfer L2 distance between predicted and ground-truth centerlines.
- Threshold-based overlap metric Ot(d), following DeepCA's evaluation
  protocol — useful under motion/deformation.

**DoD:** Table of both metrics on val + test sets, plus a stress-test
subset (foreshortening/overlap cases).

**How I plan to do it:** Runs fully on my M4 locally — lightweight
distance computations, no GPU needed. Implemented in
`src/coronarycl/metrics.py`, driven by `evaluate.py`.

### 3.2 [SMALL] [Deps: 2.3] Qualitative visualization for clinical review
- Generate a tube surface from predicted centerline + radius via a
  simple deterministic geometric sweep (not a learned model) — this is
  how stenosis/shape/foreshortening are shown to Prof. Bo Zhu without
  reintroducing a mesh-generation pipeline.
- Grounded in how minimal lumen diameter and % diameter stenosis are
  already computed clinically from a centerline + radius profile
  (Quantitative Coronary Angiography).

**DoD:** Rendered tube visualizations for 5-10 representative cases,
including at least one visible-stenosis case.

**How I plan to do it:** Runs fully on my M4 locally — simple geometric
operation. Implemented in `src/coronarycl/visualize.py`, driven by
`visualize_tube.py`.

### 3.3 [MEDIUM] [Deps: 3.1] Fine-tuning / domain-gap analysis
- If real (non-simultaneous) ICA projections are available, fine-tune
  on them to close the DRR-to-real sim-to-real gap, as DeepCA and
  3DGR-CAR both note is necessary.

**DoD:** Before/after metric comparison on any available real-data
subset; written note on remaining domain gap.

**How I plan to do it:** Fine-tuning run needs the same CUDA GPU setup
as Step 2.3 (Colab); analysis and plots done locally on my M4. Driven
by `finetune.py`.
