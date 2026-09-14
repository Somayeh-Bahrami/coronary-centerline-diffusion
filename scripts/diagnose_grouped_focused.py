"""VAL-only paired group diagnostic using the reviewed diagnostic implementations.

Keep diagnose_edge_timesteps.py and diagnose_ddim_trajectory.py beside this file.
No model training, checkpoint writes, or TEST evaluation is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def group_metrics(pred, gt, edges):
    """Measure real GT edges; array adjacency only defines the two groups."""
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    dg = gt[edges[:, 1], :3] - gt[edges[:, 0], :3]
    dp = pred[edges[:, 1], :3] - pred[edges[:, 0], :3]
    result = {}
    for name, mask in (
        ("consecutive", np.abs(edges[:, 1] - edges[:, 0]) == 1),
        ("nonconsecutive", np.abs(edges[:, 1] - edges[:, 0]) > 1),
    ):
        result[name + "_n_edges"] = int(mask.sum())
        if not mask.any():
            for metric in ("error_mm", "broken_5x", "length_ratio"):
                result[name + "_" + metric] = None
            continue
        lengths_gt = np.linalg.norm(dg[mask], axis=1)
        lengths_pred = np.linalg.norm(dp[mask], axis=1)
        if np.any(lengths_gt <= 0):
            raise ValueError("GT graph contains a zero-length edge")
        result[name + "_error_mm"] = float(np.linalg.norm(dp[mask] - dg[mask], axis=1).mean())
        result[name + "_broken_5x"] = float(np.mean(lengths_pred > 5 * lengths_gt))
        result[name + "_length_ratio"] = float(lengths_pred.sum() / lengths_gt.sum())
    return result


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Diagnostic source changed: expected one anchor {old!r}")
    return source.replace(old, new, 1)


def run_reviewed(filename, argv, reference):
    path = Path(__file__).resolve().parent / filename
    source = path.read_text()
    # Rename the third arm consistently, including its CLI and reference lookup.
    source = source.replace('"coherence"', '"grouped"').replace('"--coherence"', '"--grouped"')
    if filename == "diagnose_edge_timesteps.py":
        source = replace_once(source, 'ds = CoronaryCenterlineDatasetV31',
            'if REFERENCE is not None:\n        assert ids == REFERENCE["protocol"]["samples"], "VAL selection differs from reference"\n    ds = CoronaryCenterlineDatasetV31')
        source = replace_once(source, '"abar": a.item(),',
            '**group_metrics(pred[:, :3]*std, gt[:, :3]*std, edges[sid]),\n                                "abar": a.item(),')
    else:
        source = replace_once(source, '"edge_vector_error_mm":float(np.linalg.norm(error,axis=1).mean()),',
            '**group_metrics(pred, gt[:, :3], e),\n                        "edge_vector_error_mm":float(np.linalg.norm(error,axis=1).mean()),')
    old_argv = sys.argv
    try:
        sys.argv = [str(path), *map(str, argv)]
        exec(compile(source, str(path), "exec"), {
            "__name__": "__main__", "__file__": str(path),
            "group_metrics": group_metrics, "REFERENCE": reference,
        })
    finally:
        sys.argv = old_argv


def summarize(path, dimensions):
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    metrics = [g + "_" + m for g in ("consecutive", "nonconsecutive")
               for m in ("error_mm", "broken_5x", "length_ratio")]
    summary = []
    for key in sorted({tuple(r[d] for d in dimensions) for r in rows}):
        group = [r for r in rows if tuple(r[d] for d in dimensions) == key]
        record = dict(zip(dimensions, key))
        for metric in metrics:
            patients = {}
            for row in group:
                if row[metric] is not None:
                    patients.setdefault(row["patient"], []).append(row[metric])
            record[metric] = float(np.mean([np.mean(v) for v in patients.values()])) if patients else None
            record[metric + "_n_patients"] = len(patients)
        summary.append(record)
    payload["group_summary"] = summary
    payload["protocol"]["group_definition"] = "Real GT edges split by abs(i-j)==1 versus >1; not anatomical branch labels"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "data", "baseline", "control", "grouped", "out-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--reference", type=Path,
        help="Optional prior timestep report used to lock samples and checkpoints")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    ref = json.loads(args.reference.read_text()) if args.reference else None
    if ref is not None:
        assert ref["protocol"]["split"] == "val"
        assert ref["protocol"]["seed"] == 104729
        for arm in ("baseline", "control"):
            with getattr(args, arm).open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            assert checksum == ref["checkpoints"][arm]["sha256"], (
                f"Wrong {arm} checkpoint")
    assert args.batch > 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    noised = args.out_dir / "grouped_noised_gt.json"
    trajectory = args.out_dir / "grouped_trajectory.json"
    if noised.exists() or trajectory.exists():
        raise FileExistsError("Keep existing results; use a new output directory for a new run")
    common = ["--repo", args.repo, "--data", args.data, "--baseline", args.baseline,
              "--control", args.control, "--grouped", args.grouped, "--batch", args.batch]
    patient_count = len(ref["protocol"]["patients"]) if ref else 16
    run_reviewed("diagnose_edge_timesteps.py", common + ["--patients",
        patient_count, "--out", noised], ref)
    summarize(noised, ["arm", "t", "self_cond"])
    trajectory_reference = json.loads(noised.read_text())
    run_reviewed("diagnose_ddim_trajectory.py", common + ["--reference", noised,
        "--out", trajectory], trajectory_reference)
    summarize(trajectory, ["arm", "iteration", "t", "stage"])
    print("COMPLETE: upload grouped_noised_gt.json and grouped_trajectory.json", flush=True)


if __name__ == "__main__":
    main()
