"""Build a separate Stage-2 branch-token dataset without regenerating images."""

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

from src.coronarycl.branch_token_tree import decode_branch_tokens, encode_branch_tokens
from src.coronarycl.dataset_v3_1 import list_samples
from src.coronarycl.edge_coherence import load_edge_cache

REPRESENTATION_VERSION = "explicit_branch_token_tree_v1"
REQUIRED_METADATA = ("case_splits_v3.json", "norm_stats_v3.json", "pilot_report_v3.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _edge_set(edges):
    return {tuple(sorted(map(int, edge))) for edge in np.asarray(edges)}


def transform_sample(source_arrays, edges, sample):
    centerline = source_arrays["centerline"]
    mask = source_arrays["centerline_mask"].astype(bool)
    n = int(mask.sum())
    if n != int(source_arrays["n_points"]):
        raise AssertionError(f"{sample}: n_points disagrees with mask")
    if not mask[:n].all() or mask[n:].any() or np.any(centerline[n:] != 0):
        raise AssertionError(f"{sample}: invalid mask or nonzero padding")

    tokens = encode_branch_tokens(centerline, mask, edges)
    if not np.array_equal(tokens[:n, :4], centerline[:n, :4]):
        raise AssertionError(f"{sample}: geometry changed")
    if np.any(tokens[n:] != 0):
        raise AssertionError(f"{sample}: token padding is nonzero")
    if _edge_set(decode_branch_tokens(tokens, mask)) != _edge_set(edges):
        raise AssertionError(f"{sample}: token decoder does not recover tree")

    output = {key: value.copy() for key, value in source_arrays.items()}
    output["centerline"] = tokens
    output["representation_version"] = np.asarray(REPRESENTATION_VERSION)
    return output, {
        "sample": sample,
        "split": str(source_arrays["split"]),
        "n_nodes": n,
        "representation_version": REPRESENTATION_VERSION,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-edge-cache", required=True)
    parser.add_argument("--metadata-dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--edge-out", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    metadata_dir = Path(args.metadata_dir or source).resolve()
    out, edge_out = Path(args.out).resolve(), Path(args.edge_out).resolve()
    cache = Path(args.source_edge_cache).resolve()
    if cache.is_dir():
        cache = cache / "edges_v1.npz"
    if not source.is_dir() or not cache.is_file():
        raise SystemExit("source dataset or source edge cache not found")
    if out.exists() or edge_out.exists() or out == source:
        raise SystemExit("REFUSED: output already exists or equals source")
    missing = [name for name in REQUIRED_METADATA if not (metadata_dir / name).is_file()]
    if missing:
        raise SystemExit(f"required metadata missing: {missing}")

    sample_ids = list_samples(source)
    edge_map, edge_meta = load_edge_cache(cache, source, sample_ids)
    out.parent.mkdir(parents=True, exist_ok=True)
    edge_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    tmp_edges = Path(tempfile.mkdtemp(prefix=f".{edge_out.name}.", dir=edge_out.parent))
    rows = []
    try:
        for sample in sample_ids:
            source_path = source / f"{sample}.npz"
            with np.load(source_path, allow_pickle=False) as archive:
                source_arrays = {key: archive[key] for key in archive.files}
            output, row = transform_sample(source_arrays, edge_map[sample], sample)
            target = tmp_out / source_path.name
            np.savez_compressed(target, **output)
            row["source_npz_sha256"] = sha256_file(source_path)
            row["derived_npz_sha256"] = sha256_file(target)
            rows.append(row)

        for path in metadata_dir.glob("*.json"):
            shutil.copy2(path, tmp_out / path.name)
        shutil.copy2(cache, tmp_edges / "edges_v1.npz")
        manifest = {
            "schema_version": "branch_token_dataset_v1",
            "representation_version": REPRESENTATION_VERSION,
            "source_dataset": str(source),
            "source_edge_cache_sha256": sha256_file(cache),
            "source_edge_cache_metadata": edge_meta,
            "rows": rows,
        }
        (tmp_out / "branch_token_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(tmp_out, out)
        os.replace(tmp_edges, edge_out)
    except Exception:
        shutil.rmtree(tmp_out, ignore_errors=True)
        shutil.rmtree(tmp_edges, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
