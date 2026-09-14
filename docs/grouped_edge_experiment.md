# Group-balanced edge-coherence experiment

This repository contains the exact implementation used for the paired
30,000-step fine-tuning comparison. It starts both arms from the same h384
step-150,000 model-only milestone and creates new optimizer and cosine
scheduler states. It is fine-tuning, not an exact continuation of the original
200,000-step run.

## Scientific scope

- Dataset split: VAL is used for checkpoint/sampler analysis; TEST stays locked.
- Primary point-set metric: unsquared symmetric Chamfer in millimetres,
  `mean(pred->GT) + mean(GT->pred)`.
- Connectivity and length use the given/oracle GT tree topology.
- Sampling supplies the GT node count and preserves GT row correspondence.
- `abs(i-j) > 1` means non-consecutive in the stored serialization; it is not
  an anatomical bifurcation label.
- The grouped experiment changes only the edge-loss aggregation and weights.
  `diffusion.py` and the DDIM update rule are unchanged numerically.

## Frozen arms

- `configs/h384_ft_control_30k.yaml`: epsilon objective only.
- `configs/h384_ft_grouped_30k.yaml`: grouped real-edge auxiliary objective.

Both use h384, batch 16, LR `3e-5`, 1,000-step warmup, 30,000 updates, BF16,
seed 20260911, and the same step-150,000 initialization. Resolve the data,
initial-checkpoint, edge-cache, and output paths before launching either arm.

The grouped coefficients are frozen at:

- consecutive: `0.09476100415945009`
- non-consecutive: `0.029309171820258398`
- outer coherence weight: `1.0`
- Smooth-L1 beta: `0.05` normalized units
- alpha-bar timestep weighting: enabled
- hinge term: disabled

These coefficients came from a gradient calibration at the shared initial
checkpoint. They are experiment-specific, not general recommended defaults.

## Reproduction sequence

1. Run `python -m pytest -q` and
   `python -m src.coronarycl.models.diffusion`.
2. Build the TRAIN-only sidecar with `scripts/build_edge_cache.py`.
3. Verify that every cached edge list equals `topology_tree_edges` for the
   corresponding packaged sample.
4. Resolve the two YAML files into immutable copies with absolute paths.
5. Run 20-step smoke versions in new output directories.
6. Run the control and grouped arms sequentially.
7. Evaluate identical retained checkpoints using the same VAL IDs, five fixed
   sampling seeds, DDIM settings, physical bounds, and FP32 evaluation.
8. Run `scripts/diagnose_grouped_focused.py` for the paired noised-GT and DDIM
   trajectory diagnostics. Do not use TEST during development.

Every report should record the Git commit, configuration, checkpoint SHA-256,
dataset-manifest hashes, sampling seeds, and metric convention.

## Current conclusion

The tested grouped configuration did not produce a reliable endpoint
connectivity improvement. This is a result about this frozen configuration,
not a general rejection of graph-edge losses or proof that the architecture is
the sole cause.
