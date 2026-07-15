"""Entry point for Quantitative evaluation, Run locally.

Usage:
    python evaluate.py --pred path/to/pred_centerline.npy --gt path/to/gt_centerline.npy
"""

import argparse

import numpy as np

from src.coronarycl.metrics import evaluate_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="(N,3) or (N,4) predicted centerline .npy")
    parser.add_argument("--gt", required=True, help="(N,3) or (N,4) ground-truth centerline .npy")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[1.0, 2.0, 5.0],
                         help="Ot(d) thresholds in mm.")
    args = parser.parse_args()

    pred = np.load(args.pred)[:, :3]
    gt = np.load(args.gt)[:, :3]
    results = evaluate_case(pred, gt, thresholds=args.thresholds)

    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
