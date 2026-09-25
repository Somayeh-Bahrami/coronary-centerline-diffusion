# Validation results — 50k paired comparison (v1)

Per-sample validation records and paired per-patient differences for the three
50,000-step arms reported in the manuscript. These are the archived evaluation
outputs; no inference is re-run to produce them.

All numbers are **validation-split only**. The held-out test split has not been
accessed.

## Files

| File | Contents |
|---|---|
| `dfs_val.csv` | Per-sample records, DFS-ordering arm |
| `optimal_val.csv` | Per-sample records, optimal-linear-ordering arm |
| `ebt_val.csv` | Per-sample records, explicit branch-token arm |
| `ord_paired_deltas.csv` | Paired per-patient differences, optimal − DFS |
| `br_paired_deltas.csv` | Paired per-patient differences, branch-token − DFS |
| `verify_table1.py` | Recomputes the manuscript's main table from the files above |

Each per-sample file has 177 rows, one per validation vessel, drawn from 96
patients. Every row is the mean over the five fixed sampling seeds
(`n_seeds = 5`; seeds `104729, 130363, 155921, 181081, 205019`), so the files
contain no per-seed detail.

The delta files carry one row per metric: the mean paired per-patient
difference, its 95% bootstrap confidence interval, and the number of patients
entering the comparison (96). Differences are formed per patient first, then
aggregated, so each arm is compared against the baseline on the same patients.

## Column notes

- `chamfer_l2` — symmetric Chamfer-L2 in millimetres, **summed over the two
  directions and unsquared**. This is roughly twice the value of the more
  common averaged convention; compare against other work with care.
- `broken_edge_fraction_5x`, `edge_continuity_5x`,
  `largest_connected_component_fraction_5x` — computed at a discontinuity
  threshold of five times the ground-truth edge length
  (`curve_metrics(..., discontinuity_factor=5.0)` in
  `src/coronarycl/metrics.py`).
- `tree_length_ratio` — predicted over ground-truth tree length; 1.0 is ideal.
- Topology metrics use the **supplied ground-truth graph**. They measure
  whether predicted node positions keep connected pairs close, not whether the
  model recovers topology autonomously. The branch-token decoder is the only
  component evaluated without ground-truth edges, and that evaluation is
  reported separately.
- Sampling also uses the ground-truth point count.

## Provenance

Each per-sample file corresponds to one checkpoint recorded by SHA-256 in
[`manifests/final_artifacts.sha256`](../../manifests/final_artifacts.sha256):

| File | Checkpoint entry |
|---|---|
| `dfs_val.csv` | `checkpoints/dfs_50k_latest.pt` |
| `optimal_val.csv` | `checkpoints/optimal_50k_latest.pt` |
| `ebt_val.csv` | `checkpoints/branch_token_50k_latest.pt` |

Training configurations are `configs/h384_ordering_50k_dfs.yaml`,
`configs/h384_ordering_50k_optimal.yaml`, and
`configs/h384_branch_token_stage2.yaml`. All three share the frozen settings:
hidden dimension 384, batch 16, 50,000 steps, learning rate 3e-4, BF16,
epsilon prediction, conditioning dropout 0.10, self-conditioning 0.50,
100-step DDIM sampling, guidance 2.0.

## Scope limit

`dfs_val.csv` is the **single seed-matched baseline run** — the arm trained
with `training.seed: 20260911`, the same initialization used by the
optimal-ordering arm. It is the correct baseline for the paired comparisons in
the delta files.

It is *not* the three-seed replicate mean. The seed-replicate evidence is
archived separately as
`packages/dfs_seed_replicates_reproducibility.zip` in the artifact manifest
and is not reproduced by the files in this directory.

## Reproducing the reported table

```bash
python results/validation_50k_v1/verify_table1.py
```

This recomputes per-patient means from the per-sample files and prints the
main-table values. Expected output:

```
arm         chamfer   broken%      LCC%  tree-len
dfs          20.816     4.346     25.77     3.165
optimal      21.903     3.785     18.60     2.556
branch       19.791    38.470     14.01     8.211
```

It also re-derives every paired per-patient difference from the per-sample
files and checks it against the archived delta files. All eight comparisons in
the main table reproduce exactly; the script exits non-zero if any does not.

To regenerate the per-sample files themselves, run `ddim_eval.py` against the
corresponding checkpoint and edge cache as documented in the top-level README.
That requires the checkpoints and packaged datasets, which are distributed
outside Git and verified against the artifact manifest.
