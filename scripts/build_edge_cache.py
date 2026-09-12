"""Precompute GT vessel-graph edges for the coherence loss.

Uses `src/coronarycl/metrics.py:topology_tree_edges(gt_mm, iso_mm)` -- the SAME
function the evaluator uses for edge_continuity_5x and
largest_connected_component_fraction_5x. Optimising a different edge set than
the gate measures would repeat, at one remove, the Chamfer-convention mistake.

Node indices are into the UNPADDED valid rows, in packaged array order:
    gt_mm = z["centerline"][z["centerline_mask"]][:, :3]
Since the packaged padding is a contiguous tail, those indices are also valid
indices into the padded array -- so no re-indexing is needed at training time.
That invariant is asserted per sample, not assumed.

DO NOT WRITE THIS INTO THE DATASET DIRECTORY.
`dataset_v3_1.list_samples` globs every *.npz in packaged_dir and reads
`z["split"]`; edges_v1.npz has no such key, so a cache written there makes the
loader raise on the next run. `--out` is REQUIRED and is refused if it resolves
to the dataset directory.

Output: <out>/edges_v1.npz containing one (E, 2) int32 array per sample id plus
`_meta`, a JSON record holding per-sample node counts, edge counts and a
content fingerprint of the exact GT coordinates the edges were built from. The
trainer re-checks that fingerprint, so a cache built against different data or
a different point ordering is rejected rather than silently mis-indexed.

Validation performed per sample (all fatal):
  * every index in [0, n)
  * no self-loops, no duplicate edges
  * E == n - 1                      (a spanning tree)
  * the graph is connected          (single component by union-find)

Run:
    python scripts/build_edge_cache.py --data /path/ds105_full --out /path/edge_cache
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from src.coronarycl.metrics import topology_tree_edges        # noqa: E402
from src.coronarycl.dataset_v3_1 import list_samples          # noqa: E402


def as_edge_array(edges, n_nodes, sample):
    """Normalise whatever topology_tree_edges returns into (E, 2) int32.

    Accepts a list of pairs, an (E,2) array, or a (2,E) array. Validates every
    index, rejects self-loops, and canonicalises each edge to (min, max) so a
    later dedupe or comparison is order-independent.
    """
    a = np.asarray(edges)
    if a.ndim != 2:
        raise ValueError(f"{sample}: expected 2-D edges, got shape {a.shape}")
    if a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T                                  # (2,E) -> (E,2)
    if a.shape[1] != 2:
        raise ValueError(f"{sample}: cannot read shape {a.shape} as (E,2)")
    a = a.astype(np.int64)
    if a.size:
        if a.min() < 0 or a.max() >= n_nodes:
            raise ValueError(
                f"{sample}: edge index out of range [0,{n_nodes}): "
                f"min {a.min()} max {a.max()}")
        if (a[:, 0] == a[:, 1]).any():
            raise ValueError(f"{sample}: self-loop in edge list")
        a = np.sort(a, axis=1)
    return a.astype(np.int32)


def check_tree(edges, n, sample, strict_tree):
    """No duplicates, spanning-tree edge count, and connectivity."""
    if len(edges):
        pairs = {(int(a), int(b)) for a, b in edges}
        if len(pairs) != len(edges):
            raise ValueError(f"{sample}: {len(edges) - len(pairs)} duplicate edges")

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    n_components = len({find(k) for k in range(n)})

    message = None
    if len(edges) != n - 1:
        message = f"{sample}: E={len(edges)} but N-1={n - 1} (not a spanning tree)"
    elif n_components != 1:
        message = f"{sample}: graph has {n_components} components, expected 1"
    if message:
        if strict_tree:
            raise ValueError(message)
        print("  WARNING:", message)
    return n_components


def fingerprint(gt_mm):
    """Content hash of the exact coordinates, in order, the edges refer to."""
    return hashlib.sha256(np.ascontiguousarray(gt_mm, dtype=np.float64)
                          .tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True,
                    help="MUST NOT be the dataset directory -- see module docstring")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--allow_non_tree", action="store_true",
                    help="downgrade the E==N-1 / connectivity checks to warnings")
    args = ap.parse_args()

    data = Path(args.data).resolve()
    out = Path(args.out).resolve()
    if out == data:
        raise SystemExit(
            "REFUSED: --out equals --data. list_samples() globs *.npz in the "
            "dataset directory and reads z['split']; edges_v1.npz has no such "
            "key and would break the loader. Use a separate directory.")
    out.mkdir(parents=True, exist_ok=True)

    store, rows = {}, []
    for split in args.splits.split(","):
        for sid in list_samples(data, split.strip()):
            with np.load(data / f"{sid}.npz", allow_pickle=True) as z:
                mask = z["centerline_mask"].astype(bool)
                # the packaged padding is a contiguous tail -- assert, don't assume
                n = int(mask.sum())
                assert mask[:n].all() and not mask[n:].any(), \
                    f"{sid}: centerline_mask is not a contiguous prefix"
                gt_mm = z["centerline"][mask][:, :3].astype(np.float64)
                iso_mm = float(z["iso_mm"])
                vessel = str(z["vessel"])

            e = as_edge_array(topology_tree_edges(gt_mm, iso_mm), n, sid)
            n_comp = check_tree(e, n, sid, strict_tree=not args.allow_non_tree)
            store[sid] = e
            rows.append({"sample": sid, "split": split.strip(), "vessel": vessel,
                         "n_nodes": n, "n_edges": int(len(e)),
                         "n_components": int(n_comp),
                         "fingerprint": fingerprint(gt_mm)})
            if len(rows) % 200 == 0:
                print(f"  {len(rows)} samples", flush=True)

    ne = np.array([r["n_edges"] for r in rows])
    nn = np.array([r["n_nodes"] for r in rows])
    path = out / "edges_v1.npz"
    meta = {"source_data": str(data), "n_samples": len(rows),
            "builder": "topology_tree_edges", "rows": rows}
    np.savez_compressed(path, _meta=json.dumps(meta), **store)

    print(f"\nwrote {path}  ({path.stat().st_size / 1e6:.1f} MB, {len(rows)} samples)")
    print(f"nodes  median {int(np.median(nn))}  min {nn.min()}  max {nn.max()}")
    print(f"edges  median {int(np.median(ne))}  min {ne.min()}  max {ne.max()}")
    print(f"E == N-1 on all samples: {bool((ne == nn - 1).all())}")
    print(f"single connected component: "
          f"{bool(all(r['n_components'] == 1 for r in rows))}")
    print("\nNEXT, AND DO NOT SKIP IT: compare this cache against the evaluator "
          "edge-for-edge on a few samples. Matching counts is NOT the same as "
          "matching edges, and optimising a different edge set than the gate "
          "measures is the whole mistake this file exists to avoid.")
    if (ne == 0).any():
        bad = [r["sample"] for r in rows if r["n_edges"] == 0]
        print(f"\nWARNING: {len(bad)} samples have NO edges: {bad[:10]}")


if __name__ == "__main__":
    main()
