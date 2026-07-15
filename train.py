"""Entry point for Model training Full runs need Colab (CUDA); quick-test
is safe on laptop locally.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --quick-test   # M4-safe sanity check
"""

import argparse

from src.coronarycl.config import load_config
from src.coronarycl.trainer import train as run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick-test", action="store_true",
                        help="Tiny local run on M4 to sanity-check the training loop.")
    parser.add_argument("--device", default=None,
                        help="Override auto-detected device.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.setdefault("train", {})["device"] = args.device

    run_training(cfg, quick_test=args.quick_test)


if __name__ == "__main__":
    main()
