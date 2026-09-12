#!/usr/bin/env python3
"""Predeclared VAL-only checkpoint/sampler selection rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-continuity", type=float, default=0.95)
    parser.add_argument("--min-lcc", type=float, default=0.90)
    parser.add_argument("--min-length-ratio", type=float, default=0.75)
    parser.add_argument("--max-length-ratio", type=float, default=1.25)
    return parser.parse_args()


def selection_metrics(payload):
    bootstrap = payload["patient_bootstrap"]
    return {
        key: float(value["equal_patient_mean"])
        for key, value in bootstrap.items()}


def main():
    args = parse_args()
    candidates = []
    reference_samples = None
    for path in args.results:
        payload = json.loads(path.read_text())
        protocol = payload["protocol"]
        if protocol["split"] != "val":
            raise RuntimeError(f"Selection input is not VAL: {path}")
        if payload["mode"] != "conditional":
            raise RuntimeError(f"Selection input is a control: {path}")
        samples = tuple(row["sample"] for row in payload["per_sample"])
        if reference_samples is None:
            reference_samples = samples
        elif samples != reference_samples:
            raise RuntimeError("Candidates do not contain identical VAL samples")
        metrics = selection_metrics(payload)
        feasible = (
            metrics["edge_continuity_5x"] >= args.min_continuity
            and metrics["largest_connected_component_fraction_5x"] >= args.min_lcc
            and args.min_length_ratio <= metrics["tree_length_ratio"]
            <= args.max_length_ratio
        )
        candidates.append({
            "result_json": str(path.resolve()),
            "checkpoint": payload["checkpoint"],
            "ddim_steps": protocol["ddim_steps"],
            "guidance": protocol["guidance"],
            "bounds": protocol["bounds"],
            "seeds": protocol["seeds"],
            "metrics_equal_patient": metrics,
            "feasible": feasible,
        })

    candidates.sort(key=lambda row: (
        not row["feasible"],
        row["metrics_equal_patient"]["chamfer_l2"],
        row["metrics_equal_patient"]["hd95_mm"],
    ))
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    result = {
        "selection_split": "val",
        "rule": {
            "feasibility": {
                "edge_continuity_5x_min": args.min_continuity,
                "largest_connected_component_fraction_5x_min": args.min_lcc,
                "tree_length_ratio_range": [
                    args.min_length_ratio, args.max_length_ratio],
            },
            "objective_within_feasible_set": (
                "minimum equal-patient Chamfer L2; HD95 tie-break"),
        },
        "selected": feasible[0] if feasible else None,
        "decision": (
            "selected" if feasible else
            "no candidate met curve-coherence gates; do not run TEST"),
        "candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for candidate in candidates:
        metrics = candidate["metrics_equal_patient"]
        checkpoint = candidate["checkpoint"]
        print(
            f"step={checkpoint['step']:>6} "
            f"DDIM={candidate['ddim_steps']:>3} g={candidate['guidance']:<3} "
            f"CD={metrics['chamfer_l2']:.2f} HD95={metrics['hd95_mm']:.2f} "
            f"continuity={metrics['edge_continuity_5x']:.3f} "
            f"LCC={metrics['largest_connected_component_fraction_5x']:.3f} "
            f"length={metrics['tree_length_ratio']:.3f} "
            f"{'PASS' if candidate['feasible'] else 'FAIL'}")
    print(f"decision: {result['decision']}\nwrote {args.out}")
    if not feasible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
