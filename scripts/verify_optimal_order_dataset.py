"""Independently verify the derived optimal-order dataset and edge cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_METADATA = (
    "case_splits_v3.json",
    "norm_stats_v3.json",
    "pilot_report_v3.json",
)


def fingerprint(points: np.ndarray) -> str:
    raw = np.ascontiguousarray(points, dtype=np.float64).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in np.asarray(edges)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--derived", required=True)
    parser.add_argument("--permutations", required=True)
    parser.add_argument("--source-edge-cache", required=True)
    parser.add_argument("--derived-edge-cache", required=True)
    parser.add_argument(
        "--metadata-dir",
        help="authoritative v3 JSON directory; defaults to --source",
    )
    args = parser.parse_args()

    source, derived = Path(args.source), Path(args.derived)
    metadata_dir = Path(args.metadata_dir or args.source)
    source_cache = Path(args.source_edge_cache)
    derived_cache = Path(args.derived_edge_cache)
    if source_cache.is_dir():
        source_cache = source_cache / "edges_v1.npz"
    if derived_cache.is_dir():
        derived_cache = derived_cache / "edges_v1.npz"

    source_ids = {path.stem for path in source.glob("*.npz")}
    derived_ids = {path.stem for path in derived.glob("*.npz")}
    if source_ids != derived_ids or len(source_ids) != 1694:
        raise AssertionError("source and derived sample IDs do not match exactly")

    with np.load(args.permutations, allow_pickle=True) as archive:
        permutations = {key: archive[key] for key in archive.files if key != "_meta"}
    with np.load(source_cache, allow_pickle=True) as archive:
        source_edges = {key: archive[key] for key in archive.files if key != "_meta"}
    with np.load(derived_cache, allow_pickle=True) as archive:
        derived_meta = json.loads(str(archive["_meta"]))
        derived_edges = {key: archive[key] for key in archive.files if key != "_meta"}
    if not (set(permutations) == set(source_edges) == set(derived_edges) == source_ids):
        raise AssertionError("dataset, permutation, and edge-cache IDs differ")
    recorded_fingerprints = {
        row["sample"]: row["fingerprint"] for row in derived_meta["rows"]
    }

    split_counts: dict[str, int] = {}
    changed = 0
    for index, sample in enumerate(sorted(source_ids), 1):
        with np.load(source / f"{sample}.npz", allow_pickle=False) as archive:
            old = {key: archive[key] for key in archive.files}
        with np.load(derived / f"{sample}.npz", allow_pickle=False) as archive:
            new = {key: archive[key] for key in archive.files}

        expected_keys = set(old) | {"ordering_version", "source_builder_version"}
        if set(new) != expected_keys:
            raise AssertionError(f"{sample}: unexpected derived fields")
        for key in old:
            if key != "centerline" and not equal(old[key], new[key]):
                raise AssertionError(f"{sample}: field changed: {key}")

        mask = old["centerline_mask"].astype(bool)
        n = int(mask.sum())
        permutation = np.asarray(permutations[sample], dtype=np.int64)
        if not np.array_equal(np.sort(permutation), np.arange(n)):
            raise AssertionError(f"{sample}: invalid permutation")
        if not np.array_equal(new["centerline"][:n], old["centerline"][:n][permutation]):
            raise AssertionError(f"{sample}: valid centerline rows were not permuted exactly")
        if not np.array_equal(new["centerline"][n:], old["centerline"][n:]):
            raise AssertionError(f"{sample}: padding changed")
        if str(new["ordering_version"]) != "optimal_linear_v1":
            raise AssertionError(f"{sample}: incorrect ordering version")
        if not np.array_equal(permutation, np.arange(n)):
            changed += 1

        inverse = np.empty(n, dtype=np.int64)
        inverse[permutation] = np.arange(n)
        expected_edges = edge_set(inverse[np.asarray(source_edges[sample])])
        if edge_set(derived_edges[sample]) != expected_edges:
            raise AssertionError(f"{sample}: canonical graph was not remapped exactly")
        current_fp = fingerprint(new["centerline"][:n, :3])
        if recorded_fingerprints.get(sample) != current_fp:
            raise AssertionError(f"{sample}: derived edge-cache fingerprint mismatch")

        split = str(new["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
        if index % 200 == 0:
            print(f"  independently verified {index}/{len(source_ids)} samples", flush=True)

    for name in REQUIRED_METADATA:
        authoritative = metadata_dir / name
        target = derived / name
        if not authoritative.is_file():
            raise AssertionError(f"required authoritative metadata missing: {authoritative}")
        if not target.is_file() or authoritative.read_bytes() != target.read_bytes():
            raise AssertionError(f"JSON was not copied byte-identically: {name}")

    expected_splits = {"train": 1350, "val": 177, "test": 167}
    if split_counts != expected_splits:
        raise AssertionError(f"split counts differ: {split_counts}")
    qa = json.loads((derived / "optimal_order_qa_v1.json").read_text())
    if qa.get("status") != "PASS":
        raise AssertionError("builder QA report is not PASS")

    print("INDEPENDENT DERIVED-DATASET VERIFICATION PASSED")
    print(f"samples: {len(source_ids)}  reordered samples: {changed}")
    print(f"splits: {split_counts}")
    print("all five centerline columns moved together: PASS")
    print("all non-centerline fields and source JSON files unchanged: PASS")
    print("canonical graph remapping and fingerprints: PASS")
    print("NEXT: freeze dataset hashes, then prepare the paired 50k pilot.")


if __name__ == "__main__":
    main()
