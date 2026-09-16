# Stage 2 — Frozen Representation

Method name: Explicit Branch-Token Tree Representation (EBT)

## Node record
Each valid output row represents one coronary node:
[x, y, z, radius, branch_id, parent_branch_id, parent_attach_index, within_branch_index].

For a sample with padded capacity M, encode branch_id, parent_branch_id,
parent_attach_index, and within_branch_index as their non-negative integer value
divided by (M - 1). Branch IDs are 1..B; 0 is reserved for no-parent.
At inference, multiply by (M - 1), round to the nearest integer, then decode.
Padding is all-zero and is excluded by the existing valid-node mask.

## Canonical target construction
Use the ground-truth tree from the matching frozen edge cache. The canonical root is the valid node with the smallest original raw row index.
Split a branch at every bifurcation and terminal leaf. Assign branch IDs by
deterministic depth-first traversal from that root; sibling ties use the
existing canonical node ordering. Each node belongs to exactly one branch. An attachment/bifurcation node belongs
to its parent branch only; each child branch begins at the adjacent child node
and its first node is connected to that parent attachment. Within each branch,
nodes are ordered from its first owned node to its terminal end.

For the root branch:
parent_branch_id = 0 and parent_attach_index = 0.

For every other branch:
parent_branch_id identifies its parent branch and parent_attach_index identifies
the parent-branch node to which its first node attaches.

Coordinates and radius remain attached to their original node throughout this
transformation.

## Inference without ground-truth topology
Decode only from predicted rows: de-normalize and quantize topology fields;
discard invalid/padded rows; group rows by predicted branch_id; sort each group
by within_branch_index; connect consecutive nodes in each group; then attach
each non-root branch's first node to the predicted node indicated by
(parent_branch_id, parent_attach_index).

Invalid references, cycles, duplicate IDs, and empty branches are resolved only
by this fixed deterministic decoder: retain the root component and discard
unresolvable non-root branches. Ground-truth topology is never supplied at
inference.

## Frozen controls
The existing architecture, epsilon target, optimizer, schedule, batch size,
50k endpoint, conditioning, precision, DDIM sampler, guidance, five validation
seeds, and locked TEST split are unchanged. No alternative Stage 2
representation will be evaluated.
