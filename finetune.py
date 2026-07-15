"""Entry point for Fine-tuning / domain-gap analysis, Needs the same CUDA GPU setup as train.py
(Colab). Analysis/plots done locally.

Usage:
    python finetune.py --checkpoint outputs/model.pt --real-data-dir data/real_ica/
"""

import argparse


def finetune(checkpoint_path: str, real_data_dir: str):
    raise NotImplementedError(
        "TODO once real ICA data is available. Run on Colab (same "
        "requirements as train.py). See docs/work_breakdown.md Step 3.3."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--real-data-dir", required=True)
    args = parser.parse_args()
    finetune(args.checkpoint, args.real_data_dir)
