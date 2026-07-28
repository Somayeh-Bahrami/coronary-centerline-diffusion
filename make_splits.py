"""Entry point for Step 1.3. Runs locally on M4, no GPU needed.

Usage:
    python make_splits.py --config configs/default.yaml
"""

import argparse
from pathlib import Path

from src.coronarycl.config import load_config
from src.coronarycl.splits import get_verified_case_ids, make_case_level_split, write_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--centerline-dir",
                        default="data/processed/centerlines")
    parser.add_argument(
        "--drr-dir", default="data/processed/DRR_Generation/Total")
    args = parser.parse_args()

    cfg = load_config(args.config)

    case_ids = get_verified_case_ids(args.centerline_dir, args.drr_dir)
    print(f"ID consistency check passed: {len(case_ids)} verified cases")

    splits = make_case_level_split(case_ids, seed=args.seed)
    out_path = Path(cfg["data"]["splits_file"])
    write_splits(splits, out_path)

    print(f"train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])} -> {out_path}")


if __name__ == "__main__":
    main()
