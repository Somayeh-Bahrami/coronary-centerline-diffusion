# Frozen Experimental Protocol: Optimal Ordering and Tree-Aware Reconstruction

## 1. Scope

The project reconstructs a 3D coronary centerline and physical vessel radius
from two synthetic X-ray projections and their acquisition geometry. The
reconstructed geometry is intended for future hemodynamic estimation.

Only two remaining method opportunities are permitted:

1. optimal linear ordering of the existing centerline nodes;
2. one branch/tree-aware representation, only if optimal ordering fails its
   preregistered advancement rule.

No additional loss, sampler, prediction-target, hidden-dimension, or generic
hyperparameter experiments will be opened.

## 2. Frozen Reference State

- Source commit: `bb91023df2b9df7069dda9e58dbb9ccbfdfcd37e`
- Development branch: `codex/optimal-ordering-study`
- Reference model: hidden dimension 384, epsilon prediction
- Reference checkpoint: h384 step 150,000
- Reference checkpoint SHA256:
  `d16ba2d76b026864ab2134fbdfae1f1e3c8588b28e3da355769c3623e80aede1`
- Dataset builder: version 3.4
- Split: patient-level train/validation/test split
- Validation is used for all development and model selection.
- Test remains locked until the final method and evaluation protocol are frozen.

The existing packaged dataset, projections, poses, split files, and
normalization statistics must not be overwritten.

## 3. Primary Hypothesis

The current depth-first-search ordering is not the optimal one-dimensional
serialization of the coronary tree. Replacing it with a constructive optimal
linear ordering may improve learned global connectivity while preserving
point-set accuracy and conditioning use.

This hypothesis concerns the effect of retraining with a different
serialization. Merely permuting an existing prediction does not change its
tree length or connectedness when graph edges are remapped correctly.

## 4. Stage 1: Constructive Optimal Ordering

The ordering implementation must return an explicit permutation, not only the
maximum achievable number of consecutive graph edges.

For every sample, validation must confirm that:

- every valid node appears exactly once;
- coordinates, radius, and topology label remain attached to the same node;
- the unordered coordinate set is unchanged;
- the number of consecutive real graph edges equals the dynamic-programming
  optimum;
- graph topology and connectedness are unchanged;
- padding remains zero and the valid-node mask remains correct.

The implementation must pass path, star, random-tree brute-force, root
invariance, and real-sample tests.

## 5. Stage 1.5: CPU-Only Mechanism Screening

Before any GPU training, classify every real graph edge into four transition
groups:

1. consecutive under both DFS and optimal ordering;
2. non-consecutive under DFS but consecutive under optimal ordering (rescued);
3. consecutive under DFS but non-consecutive under optimal ordering
   (sacrificed);
4. non-consecutive under both orderings.

Using frozen step-150,000 validation predictions, report for each group:

- edge count and fraction;
- predicted-to-ground-truth edge-length ratio;
- broken-edge fraction;
- contribution to total tree-length inflation;
- LCA and RCA results separately;
- patient-bootstrap 95% confidence intervals.

Also report an explicitly labelled optimistic counterfactual estimate in which
rescued edges behave like currently consecutive edges after retraining, while
the cost of sacrificed edges is included. This estimate is a screening
analysis, not a prediction of the retrained model.

Proceed to the ordering pilot only if rescued edges carry materially more
breakage or length inflation than sacrificed edges and the optimistic estimate
supports at least a 10% reduction in tree-length ratio. Otherwise, reject the
ordering mechanism and move directly to Stage 2.

## 6. Derived Optimal-Order Dataset

If Stage 1.5 passes, create a separate derived dataset. Do not rerun CCTA
cropping, isotropic resampling, TIGRE rendering, or pose calibration.

Only valid centerline rows may be permuted. Images, poses, patient splits, and
geometric metadata must remain byte-identical where applicable. Normalization
statistics remain unchanged because permutation does not change their values.

The derived dataset must have a new version identifier, manifest, hashes, QA
report, and edge cache. It must never overwrite the DFS-ordered dataset.

## 7. Stage 1 Paired 50k Pilot

Train both arms from random initialization:

- Arm A: current DFS-ordered dataset;
- Arm B: optimal-order dataset.

The two arms must use identical source code, initialization seed, architecture,
optimizer, learning-rate schedule, batch size, training steps, conditioning,
precision, sampler, validation cases, and evaluation seeds. The sole intended
difference is centerline row ordering.

## 8. Frozen Validation Metrics

Evaluate using the same five fixed sampling seeds and patient-aware summaries:

- Chamfer L2 in millimetres;
- HD95 in millimetres;
- overlap at 1, 2, and 5 mm;
- real-edge continuity and broken-edge fraction;
- largest connected component fraction;
- predicted-to-ground-truth tree-length ratio;
- radius MAE, RMSE, bias, and correlation;
- out-of-crop fraction;
- conditioning sensitivity;
- LCA/RCA-stratified results;
- equal-axis graph-valid visualizations.

## 9. Stage 1 Advancement Rule

Optimal ordering advances only if all of the following hold on validation:

- LCC improves beyond the patient-bootstrap uncertainty interval;
- broken-edge fraction decreases;
- tree-length ratio moves at least 10% toward 1.0;
- Chamfer L2 and HD95 degrade by no more than 5%;
- radius accuracy does not materially degrade;
- conditioning sensitivity is preserved.

If the rule fails, no 200k optimal-order run is allowed. Proceed to Stage 2.

## 10. Stage 2: One Branch/Tree-Aware Method

Stage 2 is permitted only after Stage 1 fails. Exactly one representation must
be specified before implementation, including node coordinates, radius,
branch membership, parent-child relationships, within-branch order, and how
topology is obtained at inference. Ground-truth topology must not be supplied
at inference unless the task is explicitly redefined as reconstruction given
topology.

The method must pass representation reconstruction, padding, forward/backward,
inference-without-ground-truth-topology, 20-step smoke, and single-case overfit
tests before a paired 50k comparison with the frozen sequence baseline.

The same validation metrics and advancement principles used for Stage 1 apply.
If Stage 2 fails, method development stops and the diagnostic paper is written.

## 11. Final Confirmation

Only the selected method and the frozen baseline enter final confirmation.
Run three prespecified training seeds for each arm at 200,000 steps. An existing
run may count only if its code, data, configuration, and seed match exactly.

Checkpoint and sampler selection are performed on validation only. After the
protocol is frozen, evaluate the baseline and selected method once on the
locked test set. No tuning is permitted after test evaluation.

## 12. Hemodynamic-Readiness Interpretation

No model is described as hemodynamics-ready solely because of Chamfer or HD95.
The conclusion must also consider global connectivity, branch preservation,
tree length, physical radius accuracy, crop validity, and whether a usable
lumen representation can be constructed. Direct WSS/FFR validation is a
separate downstream phase and is not inferred from geometry metrics alone.

## 13. Stop Rule

After Stage 1 and, if required, Stage 2, no third method direction is allowed.
The evidence is frozen, artifacts and hashes are archived, and manuscript
writing begins.
