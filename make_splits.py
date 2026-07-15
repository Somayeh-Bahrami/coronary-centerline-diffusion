"""Entry point for Train / val / test split & data packaging, Runs locally.

Usage:
    python make_splits.py --config configs/default.yaml --n-cases 1000
"""

import argparse
from pathlib import Path

from src.coronarycl.config import load_config
from src.coronarycl.splits import make_case_level_split, write_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    case_ids = list(range(1, args.n_cases + 1))
    splits = make_case_level_split(case_ids, seed=args.seed)
    out_path = Path(cfg["data"]["splits_file"])
    write_splits(splits, out_path)

    print(f"train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])} -> {out_path}")


if __name__ == "__main__":
    main()
