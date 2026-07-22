# Phase 1 Work Step Breakdown

Sparse-view coronary centerline reconstruction — 2-view DRR input.
Same content as the Google Doc shared with Prof. Bo Zhu.

Infrastructure note: working machine is a MacBook Pro (Apple Silicon M4,
no CUDA). Each sub-step states whether it runs locally or needs a CUDA
GPU (Kaggle Notebooks, GPU P100).

## Step 1: Data Preparation

### 1.1 [SMALL] [Deps: NONE] Dataset acquisition & centerline ground-truth extraction — DONE
- Download ImageCAS (1000 CCTA volumes, public, with expert-annotated
  segmentation masks).
- Extract 3D centerline + radius profile per case via classical
  skeletonization (scikit-image) directly on the provided segmentation
  mask — deterministic, no learned model needed since ground-truth
  segmentation is already given.
- Each centerline point additionally carries a branch/topology label
  (endpoint, regular, or bifurcation), since the diffusion model
  (Step 2.2) treats topology as fixed and only denoises node position
  and radius — without this label there would be no way to identify
  branch points at training time.

**DoD:** For all 1000 cases, a saved (N×5) array per case —
(x, y, z, radius, topology label) — spot-checked visually on multiple
random cases against the CCTA volume. **Completed: all 1000 cases
processed.**

**How I plan to do it:** Runs fully on my M4 locally — CPU-only, no GPU
needed. scikit-image skeletonization + radius estimation via distance
transform + topology labeling via skeleton-neighbor counting.
Implemented in `src/coronarycl/centerline.py`, driven by
`prepare_centerlines.py`.

### 1.2 [MEDIUM] [Deps: NONE] DRR generation — 2 views, correct camera geometry + motion simulation — DONE
- Generate synthetic 2D X-ray projections via TIGRE, not DeepDRR.
  DeepDRR's three-class material decomposition (air/soft tissue/bone)
  was found to absorb contrast-enhanced vessel voxels into the generic
  soft-tissue class, yielding only weak vessel-vs-background contrast
  (effect size 0.35, confirmed quantitatively) even after display
  enhancement. TIGRE performs a direct Beer-Lambert line integral over
  Hounsfield-derived attenuation with no material-classification step,
  avoiding this failure mode and matching DeepCA's own toolchain.
- Projections are single, non-subtracted shots, not Digital Subtraction
  Angiography (DSA). DSA was tried and gave stronger measured contrast
  (effect size 0.92), but was reverted: routine coronary angiography is
  normally acquired without background subtraction, since cardiac
  motion causes subtraction misregistration artifacts. Single-shot
  projections keep training data closer to real deployment conditions.
- Generate 2 non-simultaneous projections per case, with an independent
  small rigid perturbation (±10° rotation, ±8mm translation) applied to
  the second view only, following DeepCA's (Wang et al., WACV 2025)
  exact motion-simulation protocol. Both alpha and beta angulation are
  applied (an earlier version silently ignored beta).
- Save a 3×4 projection matrix alongside every image, as a clean
  float32 array (not a dict/object array, which causes PyTorch
  DataLoader issues). Since TIGRE's internal camera-geometry convention
  could not be reliably derived analytically (two independent attempts
  each produced 20+ pixel errors), the matrix is instead obtained
  empirically per view via Direct Linear Transform calibration against
  TIGRE's own forward projections of known synthetic markers, validated
  to sub-pixel accuracy on held-out points.

**DoD:** 2 projections × 1000 cases, each with a paired 3×4
projection-matrix array and vessel-mask projection. **Completed: all 5
batches (1-200, 201-400, 401-600, 601-800, 801-1000) generated,
verified complete (0 missing cases per batch), and spot-checked both
visually and via automated anomaly detection (correct shape, no
NaN/Inf, non-trivial intensity range).**

**Known limitation:** both views are projections of the same static CT
volume — there is no simulated cardiac motion/deformation between the
two views, only a rigid camera perturbation. This is a simplified,
"mostly static" approximation of real 2-view acquisition.

**How I plan to do it:** TIGRE requires building from source (`git
clone` + `pip install .`, no pip package available) — confirmed
working on Kaggle Notebooks (GPU P100). Full generation completed
across all 5 batches; projections + pose arrays downloaded to laptop
for everything downstream. Implemented in `src/coronarycl/drr.py`.

### 1.3 [SMALL] [Deps: 1.1, 1.2] Train / val / test split & data packaging — NEXT
- Case-level split (never split by view) to avoid leakage, following
  3DGR-CAR's (MICCAI 2024) training-set scale (960 of 1000 cases) on
  the same dataset; the remaining 40 cases split evenly between
  validation and test, a default not specified by that paper itself.
- Package into final tensors (images + projection matrices + centerline
  ground truth).

**DoD:** Fixed train/val/test case-ID lists saved (960/20/20); one
packaged dataset file per split, loadable in one line.

**How I plan to do it:** Runs fully on my M4 locally — no GPU needed,
just array bookkeeping. Implemented in `src/coronarycl/splits.py`,
driven by `make_splits.py`.

## Step 2: 3D Centerline Reconstruction Model

### 2.1 [MEDIUM] [Deps: 1.1, 1.2, 1.3] Deterministic baseline (classical epipolar-constraint matching)
- Extract a 2D centerline from each of the 2 DRR projections by
  skeletonizing the synthetic 2D vessel mask (from Step 1.2's
  label-volume projection) — genuinely the same tooling as Step 1.1,
  since both operate on a ground-truth mask rather than raw intensity,
  avoiding a separate learned 2D-segmentation step.
- Match centerline points across the 2 views using the epipolar
  constraint from the saved projection matrices (Step 1.2), then
  triangulate matched pairs to a 3D point cloud.
- This is a classical, non-learned baseline — not a reproduction of a
  specific published system — used to get a floor number before the
  generative model.

**DoD:** Baseline Chamfer L2 distance reported on the val set, against
the Step 1.1 ground-truth centerlines.

**How I plan to do it:** Prototype and run at small scale on my M4
(CPU); if too slow, move full run to Kaggle. Implemented in
`src/coronarycl/models/baseline.py`.

### 2.2 [LARGE] [Deps: NONE] Conditional diffusion model — design & implementation
- Centerline-native diffusion architecture (1D-UNet denoiser over
  centerline nodes), following AortaDiff's (2025, arXiv:2507.13404)
  centerline-diffusion design. Topology (branch connectivity) is held
  fixed; the denoiser only diffuses node position and radius, not tree
  structure.
- Condition on both 2D projections + their projection matrices via
  cross-attention, so the model can reason about epipolar geometry
  between the two views.

**DoD:** Model forward/backward pass runs end-to-end on a small dummy
batch without errors; architecture diagram + short design doc.

**How I plan to do it:** Write and debug the architecture locally on my
M4 using the PyTorch MPS backend (small batch, CPU/MPS fallback) — this
part does not require CUDA. Full-scale training moves to Step 2.3.
Implemented in `src/coronarycl/models/diffusion.py`.

### 2.3 [MEDIUM] [Deps: 2.2, 1.3] Model training
- Train on the full packaged dataset from Step 1.3, using a standard
  diffusion noise-prediction objective. Initial hyperparameters follow
  AortaDiff's reported setup (Adam, β₁=0.9/β₂=0.99, LR 1×10⁻³, T=1000
  diffusion steps) as a starting point — not a validated recipe for
  this project, since AortaDiff trained on 18 cases with 3D-volume
  conditioning, versus 960 cases with 2D-projection conditioning here.
  Batch size is set by GPU memory rather than copied from AortaDiff's
  16, and total training length is determined by early stopping
  against the Step 3.1 validation Chamfer L2 rather than a fixed
  iteration count.

**DoD:** Training curve (loss vs. epoch) saved; checkpoint that reaches
Chamfer L2 distance on the val set at or below the Step 2.1 baseline.

**How I plan to do it:** Full-dataset training needs a CUDA GPU for
reasonable speed — plan to use Kaggle (or a rented GPU if free-tier
quota is insufficient). M4 used only for local unit tests on a tiny
subset before committing a full run. Implemented in
`src/coronarycl/trainer.py`, driven by `train.py`.

## Step 3: Evaluation & Iteration

**Note on subset selection criteria:** Both the stress-test subset
(3.1) and the visible-stenosis cases (3.2) need a selection method not
yet defined elsewhere in this pipeline. Foreshortening/overlap severity
is computable directly from the saved projection matrices (Step 1.2) —
geometric, automatable, no manual review needed. Visible-stenosis
cases are identified by scanning each ground-truth radius profile
(Step 1.1) for a local minimum below a threshold. Both thresholds are
TBD and will be set empirically once Step 1.1's radius-distribution
analysis is available.

### 3.1 [SMALL] [Deps: 2.1, 2.3] Quantitative evaluation
- Chamfer L2 distance between predicted and ground-truth centerlines.
- Threshold-based overlap metric Ot(d) at multiple thresholds
  (1/2/5mm), following DeepCA's evaluation protocol — useful under
  motion/deformation.
- Identify a stress-test subset from val+test cases with high
  foreshortening angle or vessel overlap (see note above).

**DoD:** Table of both metrics on val + test sets, comparing the Step
2.1 baseline against the Step 2.2/2.3 diffusion model side by side,
plus the same comparison restricted to the stress-test subset.

**How I plan to do it:** Runs fully on my M4 locally — lightweight
distance computations, no GPU needed. Implemented in
`src/coronarycl/metrics.py`, driven by `evaluate.py`.

### 3.2 [SMALL] [Deps: 1.1, 2.3] Qualitative visualization for clinical review
- Generate a tube surface from both the predicted centerline (Step
  2.3) and its corresponding ground-truth centerline (Step 1.1) via a
  simple deterministic geometric sweep (not a learned model) —
  rendering both side by side is what lets Bo assess whether the
  reconstruction preserves stenosis-relevant geometry, not just view
  the prediction in isolation. Avoids reintroducing a mesh-generation
  pipeline.
- Grounded in how minimal lumen diameter and % diameter stenosis are
  already computed clinically from a centerline + radius profile
  (Quantitative Coronary Angiography).
- Identify visible-stenosis cases via the ground-truth radius-minimum
  criterion (see note above).

**DoD:** Rendered tube visualizations (predicted + ground truth, side
by side) for 5-10 representative cases, including at least one
identified via the radius-minimum criterion above.

**How I plan to do it:** Runs fully on my M4 locally — simple geometric
operation. Implemented in `src/coronarycl/visualize.py`, driven by
`visualize_tube.py`.

### 3.3 [MEDIUM] [Deps: 2.3, 3.1] Fine-tuning / domain-gap analysis
- If real (non-simultaneous) ICA projections are available, fine-tune
  the Step 2.3 checkpoint on them to close the DRR-to-real sim-to-real
  gap, as DeepCA and 3DGR-CAR both note is necessary.

**DoD:** Before/after Chamfer L2 and Ot(d) comparison (same metrics as
3.1) on any available real-data subset; written note on remaining
domain gap.

**How I plan to do it:** Fine-tuning run needs the same CUDA GPU setup
as Step 2.3 (Kaggle); analysis and plots done locally on my M4. Driven
by `finetune.py`.