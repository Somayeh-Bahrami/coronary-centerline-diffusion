"""Step 2.3 — model training. Full-dataset run needs a CUDA GPU (Colab)
for reasonable speed. M4 is used only for a local unit test on a tiny
subset before committing a full Colab run.

TODO: standard diffusion training loop (noise prediction loss) over
the packaged dataset from Step 1.3, using CenterlineDenoiser
(src/coronarycl/models/diffusion.py).
"""

from .config import resolve_device


def train(config: dict, quick_test: bool = False):
    """
    Args:
        config: parsed YAML config (see configs/default.yaml).
        quick_test: if True, runs a handful of steps on a tiny subset —
                    intended for local M4 sanity-checking, not real training.
    """
    device = resolve_device(config.get("train", {}).get("device", "auto"))
    print(f"Training on device: {device}")
    if device != "cuda" and not quick_test:
        print("WARNING: no CUDA GPU detected. Full training should run on "
              "Colab, not locally. Use --quick-test for a local sanity check.")

    raise NotImplementedError("Training loop TODO — see docs/work_breakdown.md Step 2.3.")
