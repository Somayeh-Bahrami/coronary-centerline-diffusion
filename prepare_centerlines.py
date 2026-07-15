"""Entry point for,centerline ground-truth extraction, Runs locally.

Usage:
    python prepare_centerlines.py --config configs/default.yaml
    python prepare_centerlines.py --config configs/default.yaml --case-ids 1 2 3
"""

import argparse
from pathlib import Path

from src.coronarycl.centerline import extract_all
from src.coronarycl.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--case-ids", nargs="*", type=int, default=None,
                        help="Optional subset of case IDs for a quick local test.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    extract_all(
        raw_dir=Path(cfg["data"]["raw_dir"]),
        out_dir=Path(cfg["data"]["centerline_dir"]),
        case_ids=args.case_ids,
    )


if __name__ == "__main__":
    main()
