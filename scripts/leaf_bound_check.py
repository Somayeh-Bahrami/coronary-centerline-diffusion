"""Build exact constructive optimal orderings for every packaged sample.

This script consumes the evaluator-aligned spanning-tree edge cache and writes:

1. ``optimal_ordering_v1.npz``: one exact node permutation per sample;
2. ``leaf_bound_check_results.json``: per-sample DFS-versus-optimal statistics;
3. ``leaf_bound_check_summary.json``: aggregate statistics and provenance.

It does not modify the dataset. The saved permutation maps new row positions to
old packaged row indices, so a later derived-dataset builder must apply the same
permutation to all five valid centerline columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.coronarycl.dataset_v3_1 import list_samples  # noqa: E402
from src.coronarycl.edge_coherence import load_edge_cache  # noqa: E402
from src.coronarycl.optimal_ordering import (  # noqa: E402
    count_consecutive_tree_edges,
    optimal_tree_ordering,
)


def sha256_file(path: Path) -> str:
    """Return the full SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    """Return the source commit recorded with the output."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def distribution(values) -> dict:
    """Return stable descriptive statistics for a numeric sequence."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def canonical_edge_set(edges) -> set[tuple[int, int]]:
    """Represent an edge array as canonical undirected integer pairs."""
    return {
        tuple(sorted((int(left), int(right))))
        for left, right in np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    }


def validate_permutation(permutation, n_nodes, edges, optimum, sample):
    """Fail if a constructive result is incomplete or misses its optimum."""
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (n_nodes,):
        raise AssertionError(
            f"{sample}: permutation shape {permutation.shape}, expected {(n_nodes,)}")
    if not np.array_equal(np.sort(permutation), np.arange(n_nodes)):
        raise AssertionError(f"{sample}: result is not a complete permutation")
    achieved = count_consecutive_tree_edges(permutation, edges)
    if achieved != optimum:
        raise AssertionError(
            f"{sample}: constructed order achieved {achieved}, optimum is {optimum}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Packaged builder-v3 dataset directory")
    parser.add_argument("--edge-cache", required=True,
                        help="Evaluator-aligned edge-cache directory or NPZ")
    parser.add_argument("--out", required=True,
                        help="Separate output directory; never the dataset directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = Path(args.data).resolve()
    edge_cache_argument = Path(args.edge_cache).resolve()
    edge_cache_path = (
        edge_cache_argument / "edges_v1.npz"
        if edge_cache_argument.is_dir()
        else edge_cache_argument
    )
    out = Path(args.out).resolve()
    if out == data:
        raise SystemExit("REFUSED: --out must not equal --data")
    if not data.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {data}")
    if not edge_cache_path.is_file():
        raise FileNotFoundError(f"edge cache not found: {edge_cache_path}")
    out.mkdir(parents=True, exist_ok=True)

    permutation_path = out / "optimal_ordering_v1.npz"
    results_path = out / "leaf_bound_check_results.json"
    summary_path = out / "leaf_bound_check_summary.json"
    existing = [path for path in (permutation_path, results_path, summary_path)
                if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing outputs: {existing}; pass --overwrite")

    split_ids = {
        split: list_samples(data, split)
        for split in ("train", "val", "test")
    }
    all_ids = [sample for split in ("train", "val", "test")
               for sample in split_ids[split]]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("a sample appears in more than one split")

    edge_map, edge_meta = load_edge_cache(
        edge_cache_path, data, required_ids=all_ids)
    if int(edge_meta["n_verified"]) != len(all_ids):
        raise AssertionError("edge cache was not verified for every sample")

    sample_to_split = {
        sample: split for split, samples in split_ids.items() for sample in samples
    }
    rows = []
    permutations = {}
    edge_rows = {row["sample"]: row for row in edge_meta["rows"]}

    for index, sample in enumerate(all_ids, start=1):
        with np.load(data / f"{sample}.npz", allow_pickle=True) as packaged:
            mask = packaged["centerline_mask"].astype(bool)
            n_nodes = int(mask.sum())
            if not mask[:n_nodes].all() or mask[n_nodes:].any():
                raise AssertionError(f"{sample}: valid mask is not a contiguous prefix")
            vessel = str(packaged["vessel"])

        edges = np.asarray(edge_map[sample], dtype=np.int64).reshape(-1, 2)
        if len(edges) != n_nodes - 1:
            raise AssertionError(
                f"{sample}: expected N-1 tree edges, got N={n_nodes}, E={len(edges)}")
        result = optimal_tree_ordering(n_nodes, edges)
        validate_permutation(
            result.permutation,
            n_nodes,
            edges,
            result.max_consecutive_edges,
            sample,
        )

        current_count = count_consecutive_tree_edges(np.arange(n_nodes), edges)
        n_edges = len(edges)
        current_fraction = 1.0 - current_count / n_edges
        optimal_fraction = 1.0 - result.max_consecutive_edges / n_edges
        gap = current_fraction - optimal_fraction
        if gap < -1e-12:
            raise AssertionError(f"{sample}: current ordering beats exact optimum")

        degree = np.bincount(edges.ravel(), minlength=n_nodes)
        permutation = result.permutation.astype(np.int32)
        permutations[sample] = permutation
        rows.append({
            "sample": sample,
            "split": sample_to_split[sample],
            "vessel": vessel,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "leaves": int((degree <= 1).sum()),
            "branches": int((degree >= 3).sum()),
            "max_degree": int(degree.max()),
            "current_consecutive_edges": int(current_count),
            "optimal_consecutive_edges": int(result.max_consecutive_edges),
            "current_nonconsecutive_fraction": float(current_fraction),
            "optimal_nonconsecutive_fraction": float(optimal_fraction),
            "recoverable_gap": float(gap),
            "within_half_percentage_point_of_floor": bool(gap <= 0.005),
            "dataset_coordinate_fingerprint": edge_rows[sample]["fingerprint"],
            "permutation_sha256": hashlib.sha256(
                np.ascontiguousarray(permutation).tobytes()).hexdigest(),
        })
        if index % 200 == 0:
            print(f"  processed {index}/{len(all_ids)} samples", flush=True)

    current = [row["current_nonconsecutive_fraction"] for row in rows]
    optimal = [row["optimal_nonconsecutive_fraction"] for row in rows]
    gaps = [row["recoverable_gap"] for row in rows]
    close_count = sum(row["within_half_percentage_point_of_floor"] for row in rows)
    summary = {
        "schema_version": "optimal_ordering_v1",
        "source_commit": git_commit(REPO),
        "source_dataset": str(data),
        "edge_cache": str(edge_cache_path),
        "edge_cache_sha256": sha256_file(edge_cache_path),
        "n_samples": len(rows),
        "split_counts": {split: len(ids) for split, ids in split_ids.items()},
        "current_nonconsecutive_fraction": distribution(current),
        "optimal_nonconsecutive_fraction": distribution(optimal),
        "recoverable_gap": distribution(gaps),
        "within_half_percentage_point_of_floor": {
            "count": int(close_count),
            "fraction": float(close_count / len(rows)),
        },
        "objective": (
            "maximize the number of evaluator-tree edges whose endpoints are "
            "adjacent in one node permutation"
        ),
        "permutation_semantics": (
            "new valid row position -> old packaged valid row index"
        ),
    }

    np.savez_compressed(
        permutation_path,
        _meta=json.dumps(summary, sort_keys=True),
        **permutations,
    )
    results_path.write_text(json.dumps(rows, indent=2) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nedge cache verified on {edge_meta['n_verified']} samples")
    print(f"permutations written to {permutation_path}")
    print(f"per-sample results written to {results_path}")
    print(f"summary written to {summary_path}")
    print("\nnon-consecutive edge fraction")
    print("                          mean      median")
    print(f"current DFS          {100*np.mean(current):8.3f}%  "
          f"{100*np.median(current):8.3f}%")
    print(f"exact optimum        {100*np.mean(optimal):8.3f}%  "
          f"{100*np.median(optimal):8.3f}%")
    print(f"recoverable gap      {100*np.mean(gaps):8.3f}%  "
          f"{100*np.median(gaps):8.3f}%")
    print(f"\n{close_count}/{len(rows)} samples "
          f"({100*close_count/len(rows):.1f}%) are within 0.5 percentage point "
          "of the theoretical floor")
    print("\nNEXT: run the Stage 1.5 rescued/sacrificed edge analysis. "
          "Do not build a derived dataset or start GPU training yet.")


if __name__ == "__main__":
    main()

