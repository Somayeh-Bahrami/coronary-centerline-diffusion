#!/usr/bin/env python3
"""Equal-axis, graph-valid visualizations from ``ddim_eval.py`` outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.coronarycl.visualize import (  # noqa: E402
    plot_given_topology_comparison)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample", action="append",
                        help="repeat for specific sample ids")
    parser.add_argument("--seed", type=int,
                        help="default: first fixed protocol seed")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def representative_samples(per_sample):
    ordered = sorted(per_sample, key=lambda row: row["chamfer_l2"])
    indices = [
        round((len(ordered) - 1) * quantile)
        for quantile in (0.1, 0.5, 0.9)]
    return [ordered[index]["sample"] for index in dict.fromkeys(indices)]


def main():
    args = parse_args()
    results = json.loads(args.results.read_text())
    if results["protocol"]["topology_metrics"] != "given/oracle GT topology":
        raise RuntimeError("Unexpected topology protocol")
    seed = args.seed
    if seed is None:
        seed = int(results["protocol"]["seeds"][0])
    samples = args.sample or representative_samples(results["per_sample"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.pred) as archive:
        for sample in samples:
            required = [
                f"pred__{seed}__{sample}", f"gt__{sample}",
                f"edges__{sample}", f"bounds__{sample}"]
            missing = [key for key in required if key not in archive]
            if missing:
                raise KeyError(f"Missing archive keys: {missing}")
            prediction = archive[required[0]]
            ground_truth = archive[required[1]]
            edges = archive[required[2]]
            lower, upper = archive[required[3]]
            figure = plot_given_topology_comparison(
                ground_truth, prediction, edges, lower, upper,
                sample=sample, seed=seed)
            output = args.out_dir / f"{sample}_seed{seed}.png"
            figure.savefig(output, dpi=args.dpi)
            import matplotlib.pyplot as plt
            plt.close(figure)
            print(f"wrote {output}")


if __name__ == "__main__":
    main()
