"""Build and verify a derived optimal-order dataset without regenerating data.

Only valid centerline rows are permuted. Every other sample field is copied
unchanged. The canonical source graph is re-indexed through the permutation;
it is never reconstructed from floating-point coordinates.

The output dataset and edge-cache directories must not already exist. They
are published atomically only after every sample passes all checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.coronarycl.dataset_v3_1 import list_samples  # noqa: E402
from src.coronarycl.optimal_ordering import (  # noqa: E402
    count_consecutive_tree_edges,
    optimal_tree_ordering,
)


ORDERING_VERSION = "optimal_linear_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_fingerprint(points_xyz: np.ndarray) -> str:
    raw = np.ascontiguousarray(points_xyz, dtype=np.float64).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def canonical_edges(edges: np.ndarray) -> np.ndarray:
    result = np.asarray(edges, dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"edge array must have shape (E,2), got {result.shape}")
    result = np.sort(result, axis=1)
    if len(result):
        result = result[np.lexsort((result[:, 1], result[:, 0]))]
    return result.astype(np.int32)


def assert_tree(edges: np.ndarray, n_nodes: int, sample: str) -> None:
    if len(edges) != n_nodes - 1:
        raise AssertionError(f"{sample}: E={len(edges)} but N-1={n_nodes - 1}")
    if len(edges):
        if int(edges.min()) < 0 or int(edges.max()) >= n_nodes:
            raise AssertionError(f"{sample}: remapped edge index is out of range")
        if np.any(edges[:, 0] == edges[:, 1]):
            raise AssertionError(f"{sample}: remapped graph contains a self-loop")
        if len({tuple(row) for row in edges.tolist()}) != len(edges):
            raise AssertionError(f"{sample}: remapped graph contains duplicate edges")
    parent = np.arange(n_nodes, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for left, right in edges:
        root_left, root_right = find(int(left)), find(int(right))
        if root_left != root_right:
            parent[root_left] = root_right
    if len({find(node) for node in range(n_nodes)}) != 1:
        raise AssertionError(f"{sample}: remapped graph is disconnected")


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def transform_sample(
    source_arrays: dict[str, np.ndarray],
    permutation: np.ndarray,
    source_edges: np.ndarray,
    sample: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    required = {"centerline", "centerline_mask", "n_points", "split"}
    missing = required - source_arrays.keys()
    if missing:
        raise KeyError(f"{sample}: missing fields {sorted(missing)}")

    centerline = source_arrays["centerline"]
    mask = source_arrays["centerline_mask"].astype(bool)
    n_nodes = int(mask.sum())
    if n_nodes != int(source_arrays["n_points"]):
        raise AssertionError(f"{sample}: n_points disagrees with centerline_mask")
    if not mask[:n_nodes].all() or mask[n_nodes:].any():
        raise AssertionError(f"{sample}: valid rows are not a contiguous prefix")
    if np.any(centerline[n_nodes:] != 0):
        raise AssertionError(f"{sample}: source padding is not exactly zero")

    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (n_nodes,):
        raise AssertionError(
            f"{sample}: permutation shape {permutation.shape}, expected {(n_nodes,)}"
        )
    if not np.array_equal(np.sort(permutation), np.arange(n_nodes)):
        raise AssertionError(f"{sample}: ordering is not a complete permutation")

    source_edges = canonical_edges(source_edges)
    assert_tree(source_edges, n_nodes, sample)
    inverse = np.empty(n_nodes, dtype=np.int64)
    inverse[permutation] = np.arange(n_nodes)
    derived_edges = canonical_edges(inverse[source_edges])
    assert_tree(derived_edges, n_nodes, sample)

    optimum = optimal_tree_ordering(n_nodes, source_edges).max_consecutive_edges
    achieved = count_consecutive_tree_edges(permutation, source_edges)
    if achieved != optimum:
        raise AssertionError(
            f"{sample}: permutation achieves {achieved}, exact optimum is {optimum}"
        )

    derived_centerline = centerline.copy()
    derived_centerline[:n_nodes] = centerline[:n_nodes][permutation]
    if np.any(derived_centerline[n_nodes:] != 0):
        raise AssertionError(f"{sample}: derived padding is not exactly zero")
    if not np.array_equal(derived_centerline[:n_nodes][inverse], centerline[:n_nodes]):
        raise AssertionError(f"{sample}: inverse permutation does not recover source")

    output = {key: value.copy() for key, value in source_arrays.items()}
    output["centerline"] = derived_centerline
    output["ordering_version"] = np.asarray(ORDERING_VERSION)
    output["source_builder_version"] = np.asarray(str(source_arrays["builder_version"]))

    for key, value in source_arrays.items():
        if key == "centerline":
            continue
        if not arrays_equal(value, output[key]):
            raise AssertionError(f"{sample}: non-centerline field changed: {key}")

    old_edge_set = {tuple(row) for row in source_edges.tolist()}
    restored_edge_set = {
        tuple(sorted((int(permutation[a]), int(permutation[b]))))
        for a, b in derived_edges
    }
    if restored_edge_set != old_edge_set:
        raise AssertionError(f"{sample}: graph topology changed during remapping")

    row = {
        "sample": sample,
        "split": str(source_arrays["split"]),
        "n_nodes": n_nodes,
        "n_edges": int(len(derived_edges)),
        "optimal_consecutive_edges": int(optimum),
        "optimal_nonconsecutive_fraction": float((len(derived_edges) - optimum) / len(derived_edges)),
        "source_coordinate_fingerprint": coordinate_fingerprint(centerline[:n_nodes, :3]),
        "derived_coordinate_fingerprint": coordinate_fingerprint(derived_centerline[:n_nodes, :3]),
        "permutation_sha256": hashlib.sha256(
            np.ascontiguousarray(permutation, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    return output, derived_edges, row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--permutations", required=True)
    parser.add_argument("--source-edge-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--edge-out", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    permutation_path = Path(args.permutations).resolve()
    source_cache_path = Path(args.source_edge_cache).resolve()
    if source_cache_path.is_dir():
        source_cache_path = source_cache_path / "edges_v1.npz"
    output = Path(args.out).resolve()
    edge_output = Path(args.edge_out).resolve()

    if not source.is_dir():
        raise SystemExit(f"source dataset not found: {source}")
    for path, label in ((permutation_path, "permutation bank"), (source_cache_path, "edge cache")):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    for path in (output, edge_output):
        if path.exists():
            raise SystemExit(f"REFUSED: output already exists: {path}")
        if path == source:
            raise SystemExit("REFUSED: derived output equals source dataset")

    sample_ids = list_samples(source)
    with np.load(permutation_path, allow_pickle=True) as bank:
        permutation_meta = json.loads(str(bank["_meta"]))
        permutations = {key: bank[key] for key in bank.files if key != "_meta"}
    with np.load(source_cache_path, allow_pickle=True) as cache:
        source_edge_meta = json.loads(str(cache["_meta"]))
        source_edges = {key: cache[key] for key in cache.files if key != "_meta"}

    expected = set(sample_ids)
    if set(permutations) != expected:
        raise SystemExit("permutation-bank sample IDs do not exactly match the dataset")
    if set(source_edges) != expected:
        raise SystemExit("source edge-cache sample IDs do not exactly match the dataset")

    output.parent.mkdir(parents=True, exist_ok=True)
    edge_output.parent.mkdir(parents=True, exist_ok=True)
    temp_dataset = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    temp_edges = Path(tempfile.mkdtemp(prefix=f".{edge_output.name}.", dir=edge_output.parent))
    rows = []
    derived_edge_store: dict[str, np.ndarray] = {}
    try:
        for index, sample in enumerate(sample_ids, 1):
            source_path = source / f"{sample}.npz"
            with np.load(source_path, allow_pickle=False) as archive:
                source_arrays = {key: archive[key] for key in archive.files}
            transformed, remapped_edges, row = transform_sample(
                source_arrays, permutations[sample], source_edges[sample], sample
            )
            target_path = temp_dataset / source_path.name
            np.savez_compressed(target_path, **transformed)
            row.update({
                "source_npz_sha256": sha256_file(source_path),
                "derived_npz_sha256": sha256_file(target_path),
            })
            rows.append(row)
            derived_edge_store[sample] = remapped_edges
            if index % 200 == 0:
                print(f"  transformed and verified {index}/{len(sample_ids)} samples", flush=True)

        json_hashes = {}
        for source_json in sorted(source.glob("*.json")):
            target_json = temp_dataset / source_json.name
            shutil.copy2(source_json, target_json)
            if source_json.read_bytes() != target_json.read_bytes():
                raise AssertionError(f"JSON copy differs: {source_json.name}")
            json_hashes[source_json.name] = sha256_file(source_json)

        edge_rows = []
        for row in rows:
            sample = row["sample"]
            with np.load(temp_dataset / f"{sample}.npz", allow_pickle=False) as archive:
                mask = archive["centerline_mask"].astype(bool)
                points = archive["centerline"][mask, :3]
                vessel = str(archive["vessel"])
            edge_rows.append({
                "sample": sample,
                "split": row["split"],
                "vessel": vessel,
                "n_nodes": row["n_nodes"],
                "n_edges": row["n_edges"],
                "n_components": 1,
                "fingerprint": coordinate_fingerprint(points),
            })
        derived_edge_meta = {
            "schema_version": "canonical_remapped_edges_v1",
            "source_edge_cache": str(source_cache_path),
            "source_edge_cache_sha256": sha256_file(source_cache_path),
            "ordering_version": ORDERING_VERSION,
            "n_samples": len(rows),
            "builder": "canonical source edges re-indexed by optimal permutation",
            "rows": edge_rows,
        }
        derived_cache_path = temp_edges / "edges_v1.npz"
        np.savez_compressed(
            derived_cache_path,
            _meta=json.dumps(derived_edge_meta, sort_keys=True),
            **derived_edge_store,
        )

        split_counts = {}
        for row in rows:
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        manifest = {
            "schema_version": "optimal_order_dataset_v1",
            "ordering_version": ORDERING_VERSION,
            "source_dataset": str(source),
            "source_edge_cache_sha256": sha256_file(source_cache_path),
            "source_edge_cache_metadata_sha256": hashlib.sha256(
                json.dumps(source_edge_meta, sort_keys=True).encode()
            ).hexdigest(),
            "permutation_bank": str(permutation_path),
            "permutation_bank_sha256": sha256_file(permutation_path),
            "permutation_metadata": permutation_meta,
            "n_samples": len(rows),
            "split_counts": split_counts,
            "json_files_copied_byte_identically": json_hashes,
            "transformation": (
                "Only valid centerline rows were permuted; all five columns moved "
                "together. Padding, masks, images, poses, split labels, and all "
                "geometric metadata were preserved. Canonical edges were remapped."
            ),
            "rows": rows,
        }
        (temp_dataset / "optimal_order_manifest_v1.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        qa = {
            "status": "PASS",
            "n_samples": len(rows),
            "split_counts": split_counts,
            "all_permutations_complete": True,
            "all_unordered_node_sets_exactly_preserved": True,
            "all_non_centerline_fields_exactly_preserved": True,
            "all_padding_zero": True,
            "all_graphs_connected_trees": True,
            "all_orderings_reach_exact_dp_optimum": True,
            "canonical_edges_remapped_not_reconstructed": True,
        }
        (temp_dataset / "optimal_order_qa_v1.json").write_text(
            json.dumps(qa, indent=2) + "\n"
        )

        os.replace(temp_dataset, output)
        os.replace(temp_edges, edge_output)
    except BaseException:
        shutil.rmtree(temp_dataset, ignore_errors=True)
        shutil.rmtree(temp_edges, ignore_errors=True)
        raise

    print("DERIVED DATASET BUILD PASSED")
    print(f"dataset: {output}")
    print(f"edge cache: {edge_output / 'edges_v1.npz'}")
    print(f"samples: {len(rows)}  splits: {split_counts}")
    print(f"manifest: {output / 'optimal_order_manifest_v1.json'}")
    print(f"QA: {output / 'optimal_order_qa_v1.json'}")
    print("NEXT: run the independent derived-dataset verification. Do not train yet.")


if __name__ == "__main__":
    main()
